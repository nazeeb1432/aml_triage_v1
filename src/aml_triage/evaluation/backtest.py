"""Robustness checks (BUILD_PLAN M8). One test period can flatter a model; these ask whether the result holds.

1. Rolling-origin backtest: re-run the whole train -> calibrate -> freeze threshold -> test procedure at several cut
   dates, each with only the labels that would have existed at that date.
2. Feature families across folds: does each family help in most periods, or only in the headline one?
3. Held-out typology: train without any positive from one laundering pattern, then test on that pattern.
   Answers "does the model generalise to a scheme it has never seen, or does it memorise the ones it was shown?"
4. Seeds: refit the champion with different random seeds to see how much of a difference is just noise.

Hyperparameters are the champion's (tuned once on the main validation split) to keep the cost down; that is stated
in the report. Re-tuning per fold is the stricter version and is a config switch away.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aml_triage.config import Config
from aml_triage.evaluation.metrics import day_block_bootstrap, operating_point
from aml_triage.features.build import columns_for
from aml_triage.pipeline import log
from aml_triage.training.train import fit_final


def _fold_split(table: pd.DataFrame, warmup: int, val_start: int, test_start: int, test_end: int) -> pd.Series:
    d, dec = table["day"], table["decision_day"]
    s = pd.Series("drop", index=table.index)
    s[(d >= warmup) & (d < val_start) & (dec < val_start)] = "train"
    s[(d >= val_start) & (d < test_start) & (dec < test_start)] = "val"
    s[(d >= test_start) & (d < test_end)] = "test"
    return s


def _fit_eval(tr, va, te, cols, params, cfg: Config, reps: int) -> dict:
    m, cal, tau_raw = fit_final(tr[cols], tr["label"].astype(int), va[cols], va["label"].astype(int), params, cfg)
    s = m.predict_proba(te[cols])[:, 1]
    op = operating_point(te["label"].to_numpy(), s, tau_raw)
    df = pd.DataFrame({"day": te["day"].to_numpy(), "y": te["label"].astype(int).to_numpy(), "s": s})
    ci = day_block_bootstrap(df, lambda x: operating_point(x.y, x.s, tau_raw)["recall"], reps, cfg.seed) if reps else (np.nan, np.nan)
    return {"recall": op["recall"], "recall_ci": ci, "fp_reduction": op["fp_reduction"],
            "n_positives": op["n_positives"], "n_alerts": op["n_alerts"], "model": m, "tau_raw": tau_raw}


def stage_backtest(cfg: Config) -> dict:
    rep = Path(cfg.paths.reports_dir) / cfg.run_name
    run_id = (rep / "champion_run_id.txt").read_text().strip()
    man = json.loads((Path(cfg.paths.artifacts_dir) / "models" / run_id / "manifest.json").read_text())
    params, feats, champ_set = man["params"], man["features"], man["feature_set"]
    table = pd.read_parquet(cfg.work("features.parquet"))
    b = cfg.evaluation.backtest
    all_cols = list(table.columns)
    out: dict = {"run_id": run_id, "params_note": "champion hyperparameters reused in every fold (not re-tuned)",
                 "folds": [], "families": [], "held_out_typology": [], "seeds": []}

    # 1 + 2: rolling origin, champion features and every ladder family
    family_rows = []
    for cut in b.cuts:
        val_start, test_end = cut - b.val_days, min(cut + b.test_days, int(table["day"].max()) + 1)
        sp = _fold_split(table, cfg.split.warmup_days, val_start, cut, test_end)
        tr, va, te = (table[sp == k] for k in ("train", "val", "test"))
        if tr["label"].sum() < 20 or va["label"].sum() < 5 or te["label"].sum() < 5:
            log(f"backtest cut {cut}: too few positives, skipped")
            continue
        r = _fit_eval(tr, va, te, feats, params, cfg, b.bootstrap_reps)
        out["folds"].append({"cut_day": cut, "test_days": [cut, test_end - 1], "train_positives": int(tr["label"].sum()),
                             **{k: v for k, v in r.items() if k not in ("model", "tau_raw")}})
        log(f"backtest cut {cut}: recall {r['recall']:.3f} ({r['recall_ci'][0]:.2f}-{r['recall_ci'][1]:.2f}), "
            f"FP reduction {r['fp_reduction']:.1%}, {r['n_positives']} positives")
        for fs in cfg.evaluation.feature_sets:
            cols = columns_for(fs, all_cols)
            fr = _fit_eval(tr, va, te, cols, params, cfg, 0)
            family_rows.append({"cut_day": cut, "feature_set": fs, "recall": fr["recall"], "fp_reduction": fr["fp_reduction"]})
    fam = pd.DataFrame(family_rows)
    if not fam.empty:
        order = cfg.evaluation.feature_sets
        piv = fam.pivot(index="cut_day", columns="feature_set", values="recall")[order]
        for i, fs in enumerate(order):
            row = {"feature_set": fs, "mean_recall": float(piv[fs].mean()), "sd_recall": float(piv[fs].std(ddof=0)),
                   "mean_fp_reduction": float(fam[fam.feature_set == fs]["fp_reduction"].mean())}
            if i:
                diff = piv[fs] - piv[order[i - 1]]
                row["folds_better_than_previous"] = f"{int((diff > 0).sum())}/{len(diff)}"
                row["mean_gain_vs_previous"] = float(diff.mean())
            out["families"].append(row)

    # main split for 3 + 4
    from aml_triage.training.train import split

    sp = split(table, cfg)
    tr, va, te = (table[sp == k] for k in ("train", "val", "test"))

    # 3: held-out typology
    for typ, n in te.loc[te["label"], "typology"].value_counts().items():
        if n < b.min_typology_positives:
            continue
        mask_te = (te["typology"] == typ).to_numpy() & te["label"].to_numpy()
        tr_wo = tr[~((tr["typology"] == typ) & tr["label"])]
        va_wo = va[~((va["typology"] == typ) & va["label"])]
        seen = _fit_eval(tr, va, te, feats, params, cfg, 0)
        unseen = _fit_eval(tr_wo, va_wo, te, feats, params, cfg, 0)
        def rec(r, mask=mask_te):
            return float((r["model"].predict_proba(te[feats])[:, 1][mask] >= r["tau_raw"]).mean())

        out["held_out_typology"].append({"typology": typ, "test_positives": int(n),
                                         "train_positives_removed": int(((tr["typology"] == typ) & tr["label"]).sum()),
                                         "recall_when_seen": rec(seen), "recall_when_unseen": rec(unseen)})
        log(f"held-out {typ}: recall seen {rec(seen):.2f} -> unseen {rec(unseen):.2f} (n={n})")

    # 4: seeds
    for seed in b.seeds:
        r = _fit_eval(tr, va, te, feats, params, cfg.model_copy(update={"seed": seed}), 0)
        out["seeds"].append({"seed": seed, "recall": r["recall"], "fp_reduction": r["fp_reduction"]})
    rs = [s["recall"] for s in out["seeds"]]
    out["seed_spread"] = {"min": float(min(rs)), "max": float(max(rs)), "sd": float(np.std(rs))} if rs else {}

    (rep / "backtest.json").write_text(json.dumps(out, indent=2, default=float))
    (rep / "backtest.md").write_text(_md(out, champ_set))
    m = json.loads((rep / "metrics.json").read_text())
    m["robustness"] = out
    (rep / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    return out


def _md(o: dict, champ_set: str) -> str:
    L = [f"# Robustness — {o['run_id']}", "", f"Model: {champ_set}. {o['params_note'].capitalize()}.", "",
         "## Rolling-origin backtest", "", "| Test days | Train positives | Test positives | Recall (95% CI) | FP reduction |",
         "|---|---|---|---|---|"]
    for f in o["folds"]:
        L.append(f"| {f['test_days'][0]}–{f['test_days'][1]} | {f['train_positives']} | {f['n_positives']} | "
                 f"{f['recall']:.3f} ({f['recall_ci'][0]:.2f}–{f['recall_ci'][1]:.2f}) | {f['fp_reduction']:.1%} |")
    L += ["", "## Feature families across folds", "", "| Feature set | Mean recall | SD | Folds better than previous rung | Mean gain |",
          "|---|---|---|---|---|"]
    for r in o["families"]:
        L.append(f"| {r['feature_set']} | {r['mean_recall']:.3f} | {r['sd_recall']:.3f} | {r.get('folds_better_than_previous', '—')} | "
                 f"{r.get('mean_gain_vs_previous', float('nan')):+.3f} |" if "mean_gain_vs_previous" in r else
                 f"| {r['feature_set']} | {r['mean_recall']:.3f} | {r['sd_recall']:.3f} | — | — |")
    L += ["", "## Held-out laundering pattern", "", "Training positives of the pattern removed; recall on that pattern in test.", "",
          "| Pattern | Test positives | Removed from training | Recall when seen | Recall when never seen |", "|---|---|---|---|---|"]
    for r in o["held_out_typology"]:
        L.append(f"| {r['typology']} | {r['test_positives']} | {r['train_positives_removed']} | {r['recall_when_seen']:.2f} | {r['recall_when_unseen']:.2f} |")
    if o.get("seed_spread"):
        s = o["seed_spread"]
        L += ["", f"## Seeds\n\nRecall across {len(o['seeds'])} seeds: {s['min']:.3f}–{s['max']:.3f} (sd {s['sd']:.3f}). "
              "Differences between variants smaller than this are noise.", ""]
    return "\n".join(L)
