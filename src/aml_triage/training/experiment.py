"""The experiment: feature-set ladder (paper replication) + augmentation arms (GAN question),
champion selection on VALIDATION, frozen-threshold evaluation on TEST with day-block CIs,
then policy routing and reason codes for the test period.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_triage.augment.synthesizers import augment, fidelity_auc, fill_for_synthesis
from aml_triage.config import Config
from aml_triage.evaluation.metrics import (
    calibration_curve,
    day_block_bootstrap,
    operating_point,
    recall_vs_review_share,
    summary,
)
from aml_triage.evaluation.monitoring import feature_psi
from aml_triage.features.build import columns_for
from aml_triage.pipeline import git_sha, log
from aml_triage.serving.policy import btl_recall_estimate, policy_version, route
from aml_triage.serving.reasons import reason_codes
from aml_triage.training.train import TrainedModel, _objective_metric, fit_final, select_features, split, tune


def _report_dir(cfg: Config) -> Path:
    p = Path(cfg.paths.reports_dir) / cfg.run_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _evaluate(tm: TrainedModel, va: pd.DataFrame, te: pd.DataFrame, cfg: Config) -> dict:
    raw_va, raw_te = tm.raw(va), tm.raw(te)
    s = summary(te["label"], raw_te, tm.tau_raw, cfg.training.target_fpr)
    s["brier"] = float(np.mean((tm.calibrator.predict(raw_te) - te["label"].astype(int)) ** 2))
    df = pd.DataFrame({"day": te["day"].to_numpy(), "y": te["label"].astype(int).to_numpy(), "s": raw_te})
    reps = cfg.evaluation.bootstrap_reps
    s["recall_ci"] = day_block_bootstrap(df, lambda d: operating_point(d.y, d.s, tm.tau_raw)["recall"], reps, cfg.seed)
    s["fp_reduction_ci"] = day_block_bootstrap(df, lambda d: operating_point(d.y, d.s, tm.tau_raw)["fp_reduction"], reps, cfg.seed)
    s["val_pauc"] = _objective_metric(va["label"].astype(int), raw_va)
    return s


def _pretest_folds(table: pd.DataFrame, cfg: Config) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin folds whose test windows end before the real test period starts. Selecting on these uses
    several hundred positives instead of the validation split's few dozen, and never touches test."""
    from aml_triage.evaluation.backtest import _fold_split

    b, out = cfg.evaluation.backtest, []
    for cut in b.cuts:
        if cut + b.test_days > cfg.split.test_start_day or cut - b.val_days <= cfg.split.warmup_days:
            continue
        sp = _fold_split(table, cfg.split.warmup_days, cut - b.val_days, cut, cut + b.test_days)
        folds = tuple(table[sp == k] for k in ("train", "val", "test"))
        if folds[0]["label"].sum() >= 20 and folds[1]["label"].sum() >= 5 and folds[2]["label"].sum() >= 5:
            out.append(folds)
    return out


def _selection_score(folds, cols, params, arm: str, cfg: Config) -> float | None:
    """Mean recall at the frozen threshold over the pre-test folds (FPR is fixed at the target by construction)."""
    if not folds:
        return None
    scores = []
    for ftr, fva, fte in folds:
        p = dict(params)
        Xtr, ytr, Xva, Xte = ftr[cols], ftr["label"].astype(int), fva[cols], fte[cols]
        if arm != "class_weight":
            p["scale_pos_weight"] = 1.0
            Xtr, ytr, _ = augment(ftr[cols], ftr["label"], arm, cfg.augmentation.target_positive_share, cfg.seed,
                                  cfg.augmentation.ctgan)
            Xva, Xte = fill_for_synthesis(Xva), fill_for_synthesis(Xte)
        m, _, tau_raw = fit_final(Xtr, ytr, Xva, fva["label"].astype(int), p, cfg)
        scores.append(operating_point(fte["label"].to_numpy(), m.predict_proba(Xte)[:, 1], tau_raw)["recall"])
    return float(np.mean(scores))


def _fit_arm(tr, va, cols, params, arm: str, cfg: Config) -> tuple[TrainedModel, dict]:
    p = dict(params)
    extra: dict = {}
    if arm == "class_weight":
        Xtr, ytr, Xva = tr[cols], tr["label"].astype(int), va[cols]
        fill = False
    else:
        p["scale_pos_weight"] = 1.0
        t0 = time.time()
        Xtr, ytr, synth_flag = augment(tr[cols], tr["label"], arm, cfg.augmentation.target_positive_share,
                                       cfg.seed, cfg.augmentation.ctgan)
        Xva, fill = fill_for_synthesis(va[cols]), True
        extra["n_synthetic"] = int(synth_flag.sum())
        extra["augment_seconds"] = round(time.time() - t0, 1)
        if synth_flag.any():
            real_pos = Xtr[(~synth_flag) & (ytr.to_numpy() == 1)]
            extra["fidelity_auc_real_vs_synth"] = round(fidelity_auc(real_pos, Xtr[synth_flag], cfg.seed), 3)
    m, cal, tau_raw = fit_final(Xtr, ytr, Xva, va["label"].astype(int), p, cfg)
    tm = TrainedModel(model=m, calibrator=cal, features=cols, params=p, tau_raw=tau_raw, augmentation=arm,
                      fill_nan=fill, extra=extra)
    return tm, extra


def stage_train(cfg: Config) -> None:
    rep = _report_dir(cfg)
    table = pd.read_parquet(cfg.work("features.parquet"))
    table["split"] = split(table, cfg)
    tr, va, te = (table[table["split"] == s] for s in ("train", "val", "test"))
    log(f"split sizes train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}; positives "
        f"{int(tr.label.sum())}/{int(va.label.sum())}/{int(te.label.sum())}; "
        f"dropped (label not yet decided at training time): {int((table.split == 'drop').sum()):,}")
    all_cols = list(table.columns)

    # rules-only baseline: every alert is reviewed
    rules_only = operating_point(te["label"], np.ones(len(te)), 0.5)

    folds = _pretest_folds(table, cfg)
    log(f"model selection on {len(folds)} pre-test folds "
        f"({sum(int(f[2]['label'].sum()) for f in folds)} positives) + validation pAUC as tie-break")
    ladder, models = [], {}
    for fs in cfg.evaluation.feature_sets:
        if "gwd" in fs and not cfg.features.gwd.enabled:
            continue
        cols = columns_for(fs, all_cols)
        t0 = time.time()
        params = tune(tr[cols], tr["label"].astype(int), va[cols], va["label"].astype(int), cfg)
        m0, _, _ = fit_final(tr[cols], tr["label"].astype(int), va[cols], va["label"].astype(int), params, cfg)
        sel = select_features(m0, cols, cfg.training.feature_selection_importance_share)
        m, cal, tau_raw = fit_final(tr[sel], tr["label"].astype(int), va[sel], va["label"].astype(int), params, cfg)
        tm = TrainedModel(model=m, calibrator=cal, features=sel, params=params, tau_raw=tau_raw, feature_set=fs)
        ev = _evaluate(tm, va, te, cfg)
        ev.update({"feature_set": fs, "n_features_total": len(cols), "n_features_selected": len(sel),
                   "seconds": round(time.time() - t0, 1)})
        ev["selection_recall"] = _selection_score(folds, sel, params, "class_weight", cfg)
        ladder.append(ev)
        models[("ladder", fs)] = tm
        log(f"[{fs}] val pAUC {ev['val_pauc']:.3f} | test recall@tau {ev['recall']:.3f} "
            f"CI {ev['recall_ci'][0]:.2f}-{ev['recall_ci'][1]:.2f} | FP reduction {ev['fp_reduction']:.1%} | "
            f"oracle recall@{cfg.training.target_fpr:.0%}FPR {ev['recall_at_target_fpr_oracle']:.3f} | {len(sel)}/{len(cols)} feats | "
            f"selection recall {ev['selection_recall'] if ev['selection_recall'] is None else round(ev['selection_recall'], 3)}")

    def _key(r):
        return (r["selection_recall"] if r.get("selection_recall") is not None else -1, r["val_pauc"])

    # Parsimony: walk the ladder in order and move to a richer feature set only if it beats the current choice by more
    # than `selection_min_gain` (default 0.01 ≈ the seed-to-seed spread). Smaller gains are noise, not evidence.
    gain = cfg.training.selection_min_gain
    best = ladder[0]
    for r in ladder[1:]:
        if r.get("selection_recall") is None or best.get("selection_recall") is None:
            if _key(r) > _key(best):
                best = r
        elif r["selection_recall"] > best["selection_recall"] + gain:
            best = r
    base_fs = best["feature_set"]
    base = models[("ladder", base_fs)]
    log(f"augmentation arms on feature set '{base_fs}' (chosen on pre-test folds, never on test)")
    arms = []
    for arm in cfg.augmentation.methods:
        if arm == "class_weight":
            tm, extra = base, {}
        else:
            try:
                tm, extra = _fit_arm(tr, va, base.features, base.params, arm, cfg)
            except ImportError as e:
                log(f"  skipping {arm}: {e} (pip install -e '.[gan]')")
                continue
        tm.feature_set = base_fs
        ev = _evaluate(tm, va, te, cfg)
        ev.update({"augmentation": arm, "feature_set": base_fs, **extra})
        if arm == "class_weight":
            ev["selection_recall"] = next(r["selection_recall"] for r in ladder if r["feature_set"] == base_fs)
        elif arm != "ctgan" or cfg.augmentation.select_ctgan_in_folds:
            ev["selection_recall"] = _selection_score(folds, base.features, base.params, arm, cfg)
        else:
            ev["selection_recall"] = None   # CTGAN is refit per fold only if select_ctgan_in_folds (minutes each)
        arms.append(ev)
        models[("arm", arm)] = tm
        log(f"  [{arm}] val pAUC {ev['val_pauc']:.3f} | test recall@tau {ev['recall']:.3f} "
            f"CI {ev['recall_ci'][0]:.2f}-{ev['recall_ci'][1]:.2f} | FP reduction {ev['fp_reduction']:.1%} | {extra}")

    # champion chosen on pre-test folds (validation pAUC breaks ties); test numbers are reported, never used to choose.
    # Arms without a fold score (CTGAN by default) can't be champion: one small validation split is too noisy to promote on.
    # The base (class weights) stays champion unless an augmentation arm beats it by more than the same margin.
    champ_key = ("ladder", base_fs)
    base_score = best.get("selection_recall")
    better = [r for r in arms if r["augmentation"] != "class_weight" and r.get("selection_recall") is not None
              and base_score is not None and r["selection_recall"] > base_score + gain]
    if better:
        champ_key = ("arm", max(better, key=lambda r: r["selection_recall"])["augmentation"])
    champ = models[champ_key]
    champ_eval = next(r for r in (ladder if champ_key[0] == "ladder" else arms)
                      if r.get("feature_set" if champ_key[0] == "ladder" else "augmentation") == champ_key[1])
    run_id = f"{cfg.run_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _save_model(champ, champ_eval, run_id, cfg)

    raw_te = champ.raw(te)
    imp = pd.Series(champ.model.booster_.feature_importance("gain"), index=champ.features)
    per_rule = []
    for r in [c[len("a_rule_"):] for c in all_cols if c.startswith("a_rule_")]:
        mask = te[f"a_rule_{r}"].to_numpy() == 1
        if mask.sum():
            op = operating_point(te["label"].to_numpy()[mask], raw_te[mask], champ.tau_raw)
            per_rule.append({"rule": r, **{k: op[k] for k in ("n_alerts", "n_positives", "recall", "fp_reduction")}})
    per_typ = []
    for t, g in te[te["label"]].groupby("typology"):
        kept = (champ.raw(g) >= champ.tau_raw).mean()
        per_typ.append({"typology": t, "n_positive_alerts": len(g), "recall": float(kept)})

    metrics = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": json.loads(cfg.work("data_meta.json").read_text()),
        "alert_stats": json.loads(cfg.work("alert_stats.json").read_text()),
        "split": {s: {"alerts": int((table.split == s).sum()), "positives": int(table.loc[table.split == s, "label"].sum()),
                      "days": [int(table.loc[table.split == s, "day"].min()), int(table.loc[table.split == s, "day"].max())]}
                  for s in ("warmup", "train", "val", "test") if (table.split == s).any()},
        "target_fpr": cfg.training.target_fpr,
        "rules_only_baseline": rules_only,
        "ladder": ladder,
        "augmentation": arms,
        "champion": {"key": list(champ_key), "feature_set": champ.feature_set, "augmentation": champ.augmentation,
                     "calibration": type(champ.calibrator).__name__.replace("Regression", "").replace("Calibrator", "").lower(),
                     "tau_raw": champ.tau_raw, "tau_calibrated": champ.tau, "n_features": len(champ.features),
                     "params": champ.params, **{k: v for k, v in champ_eval.items() if k not in ("feature_set", "augmentation")}},
        "curves": {
            "recall_vs_review_share": recall_vs_review_share(te["label"], raw_te),
            "calibration": calibration_curve(te["label"], champ.calibrator.predict(raw_te)),
        },
        "per_rule": per_rule,
        "per_typology": per_typ,
        "feature_importance": [{"feature": f, "gain": float(g)} for f, g in imp.sort_values(ascending=False).head(25).items()],
        "drift_psi_train_vs_test": feature_psi(tr, te, list(imp.sort_values(ascending=False).head(15).index)),
    }
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    (rep / "ablation.md").write_text(_ablation_md(metrics))
    log(f"champion {champ_key} -> {rep / 'metrics.json'}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _save_model(tm: TrainedModel, ev: dict, run_id: str, cfg: Config) -> Path:
    d = Path(cfg.paths.artifacts_dir) / "models" / run_id
    d.mkdir(parents=True, exist_ok=True)
    tm.model.booster_.save_model(str(d / "model.txt"))
    joblib.dump(tm, d / "trained_model.joblib")
    data_meta = json.loads(cfg.work("data_meta.json").read_text())
    manifest = {
        "run_id": run_id, "run_name": cfg.run_name, "status": "candidate", "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(), "data_fingerprint": data_meta["sha256_16"], "data_source": data_meta["source"],
        "feature_set": tm.feature_set, "augmentation": tm.augmentation, "features": tm.features, "params": tm.params,
        "tau_raw": tm.tau_raw, "tau_calibrated": tm.tau, "target_fpr": cfg.training.target_fpr,
        "test_metrics": {k: ev.get(k) for k in ("recall", "recall_ci", "fp_reduction", "fp_reduction_ci", "pr_auc", "val_pauc")},
        "approvals": [],
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    reg_path = Path(cfg.paths.artifacts_dir) / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"models": []}
    reg["models"].append({"run_id": run_id, "status": "candidate", "path": str(d), "created_at": manifest["created_at"]})
    reg_path.write_text(json.dumps(reg, indent=2))
    (Path(cfg.paths.reports_dir) / cfg.run_name / "champion_run_id.txt").write_text(run_id)
    return d


def _fmt_sel(r: dict) -> str:
    v = r.get("selection_recall")
    return "not run" if v is None else f"{v:.3f}"


def _fmt_ci(ci, pct: bool = False) -> str:
    if not ci or ci[0] != ci[0]:
        return "n/a"
    return f"{ci[0]:.1%}–{ci[1]:.1%}" if pct else f"{ci[0]:.2f}–{ci[1]:.2f}"


def _ablation_md(m: dict) -> str:
    lines = [f"# Ablation — {m['run_id']}", "",
             f"Test period alerts: {m['split']['test']['alerts']:,} ({m['split']['test']['positives']} suspicious). "
             f"Threshold frozen on validation at {m['target_fpr']:.0%} FPR.", "",
             "## Feature-set ladder (class-weighted LightGBM)", "",
             "Selection recall = mean recall at the frozen threshold over rolling folds that end before the test period;",
             "the champion is picked on it (validation pAUC breaks ties). Test columns are reported only.", "",
             "| Feature set | Features (selected/total) | Selection recall | Val pAUC | Test recall @ frozen τ (95% CI) | FP reduction (95% CI) | PR-AUC | Oracle recall@20%FPR |",
             "|---|---|---|---|---|---|---|---|"]
    for r in m["ladder"]:
        lines.append(f"| {r['feature_set']} | {r['n_features_selected']}/{r['n_features_total']} | {_fmt_sel(r)} | {r['val_pauc']:.3f} | "
                     f"{r['recall']:.3f} ({_fmt_ci(r['recall_ci'])}) | {r['fp_reduction']:.1%} ({_fmt_ci(r['fp_reduction_ci'], pct=True)}) | "
                     f"{r['pr_auc']:.3f} | {r['recall_at_target_fpr_oracle']:.3f} |")
    lines += ["", "## Augmentation arms (training set only)", "",
              "| Arm | Synthetic rows | Real-vs-synthetic AUC | Selection recall | Val pAUC | Test recall (95% CI) | FP reduction | PR-AUC |",
              "|---|---|---|---|---|---|---|---|"]
    for r in m["augmentation"]:
        lines.append(f"| {r['augmentation']} | {r.get('n_synthetic', 0):,} | {r.get('fidelity_auc_real_vs_synth', '—')} | {_fmt_sel(r)} | "
                     f"{r['val_pauc']:.3f} | {r['recall']:.3f} ({_fmt_ci(r['recall_ci'])}) | {r['fp_reduction']:.1%} | {r['pr_auc']:.3f} |")
    c = m["champion"]
    lines += ["", f"**Champion (chosen on pre-test folds):** {c['feature_set']} / {c['augmentation']} — "
              f"test recall {c['recall']:.3f} ({_fmt_ci(c['recall_ci'])}), FP reduction {c['fp_reduction']:.1%}.", ""]
    return "\n".join(lines)


def stage_score(cfg: Config) -> None:
    """Score the test period as if it were live: policy routing, BTL sample, reason codes."""
    rep = _report_dir(cfg)
    run_id = (rep / "champion_run_id.txt").read_text().strip()
    tm: TrainedModel = joblib.load(Path(cfg.paths.artifacts_dir) / "models" / run_id / "trained_model.joblib")
    table = pd.read_parquet(cfg.work("features.parquet"))
    live = table[table["day"] >= cfg.split.test_start_day].reset_index(drop=True)
    raw = tm.raw(live)
    p = tm.calibrator.predict(raw)
    routed = route(live, raw, tm.tau_raw, cfg.policy, cfg.seed)
    reasons = reason_codes(tm.model, live, tm.features, top_k=4, fill_nan=tm.fill_nan)
    keep = ["alert_id", "account", "day", "rules", "label", "decision_day", "typology",
            "a_sum_in", "a_sum_out", "a_max_out", "a_max_in", "a_n_src", "a_n_dst", "gw_prior_confirmed"]
    out = live[keep].copy()
    out["score_raw"], out["probability"] = raw, p
    out = out.merge(routed, on="alert_id")
    out["priority_rank"] = out.groupby("day")["score_raw"].rank(ascending=False, method="first").astype(int)
    out["reasons_json"] = [json.dumps(r) for r in reasons]
    pol = policy_version(cfg.policy, tm.tau_raw)
    out["policy_version"], out["model_run_id"] = pol, run_id
    out.to_parquet(rep / "scored_alerts.parquet")

    # the transactions the UI needs to draw each alert's neighbourhood
    txns = pd.read_parquet(cfg.work("transactions.parquet"))
    lo = cfg.split.test_start_day - cfg.features.graph.tws_days
    txns[txns["day"] >= lo][["txn_id", "ts", "day", "src", "dst", "amount_usd", "pay_format", "currency"]].to_parquet(
        rep / "live_transactions.parquet")
    pd.read_parquet(cfg.work("labels.parquet")).to_parquet(rep / "labels.parquet")

    est = btl_recall_estimate(routed, raw, tm.tau_raw, live["label"].to_numpy(), cfg.policy)
    counts = routed["decision"].value_counts().to_dict()
    days = live["day"].nunique()
    m = json.loads((rep / "metrics.json").read_text())
    m["policy"] = {"version": pol, "mode": cfg.policy.mode, "config": cfg.policy.model_dump(), "decisions": counts,
                   "reasons": routed["decision_reason"].value_counts().to_dict(),
                   "reviewed_per_day": round((routed["decision"] != "SUPPRESS").sum() / days, 1),
                   "alerts_per_day": round(len(live) / days, 1), **est}
    (rep / "metrics.json").write_text(json.dumps(m, indent=2, default=_json_default))
    log(f"scored {len(out):,} live alerts with {pol}: {counts}; BTL-estimated recall "
        f"{est['recall_estimated_from_btl']:.3f} vs hindsight {est['recall_true_hindsight']:.3f}")
