"""End-to-end pipeline. Each stage reads the previous stage's files, so stages can be rerun alone.

    ingest    raw IBM-format CSV (real or synthetic)        -> transactions.parquet
    alerts    rule engine + bitemporal labels                -> alerts.parquet, labels.parquet
    features  alert/profile/degree/GW/GWd feature table     -> features.parquet, stage1.joblib
    train     feature-set ladder + augmentation arms         -> artifacts/models/<run_id>/, reports/<run>/
    score     champion model + policy on the test period     -> reports/<run>/scored_alerts.parquet
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aml_triage.config import Config, dump_config
from aml_triage.ingest.ibm import attach_typology, read_ibm_patterns, read_ibm_transactions
from aml_triage.ingest.synth import write_synthetic
from aml_triage.labels.store import make_labels
from aml_triage.rules.engine import run_rules


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _raw_paths(cfg: Config) -> tuple[Path, Path]:
    raw = Path(cfg.data.raw_dir)
    if cfg.data.source == "synthetic":
        s = cfg.data.synthetic
        return raw / s.out_trans_file, raw / s.out_patterns_file
    if cfg.data.source == "amlsim":
        d = Path(cfg.data.amlsim_dir)
        return d / "transactions.csv", d / "alerts.csv"
    return raw / cfg.data.ibm_trans_file, raw / cfg.data.ibm_patterns_file


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def stage_ingest(cfg: Config) -> pd.DataFrame:
    trans, pats = _raw_paths(cfg)
    if cfg.data.source == "synthetic" and not trans.exists():
        log(f"generating synthetic IBM-format data -> {trans}")
        write_synthetic(cfg, cfg.data.raw_dir)
    if not trans.exists():
        raise FileNotFoundError(f"{trans} not found. See docs/DATA.md for how to download the dataset.")
    log(f"reading {trans}")
    if cfg.data.source == "amlsim":
        from aml_triage.ingest.amlsim import read_amlsim

        txns = read_amlsim(trans.parent)
    else:
        txns = read_ibm_transactions(trans)
        if pats.exists():
            txns = attach_typology(txns, read_ibm_patterns(pats))
        else:
            txns["typology"] = None
    txns.to_parquet(cfg.work("transactions.parquet"))
    meta = {"source": cfg.data.source, "file": str(trans), "sha256_16": file_fingerprint(trans),
            "n_txns": len(txns), "n_accounts": int(pd.concat([txns["src"], txns["dst"]]).nunique()),
            "n_laundering": int(txns["is_laundering"].sum()), "days": int(txns["day"].max() + 1),
            "start_date": txns.attrs.get("start_date")}
    cfg.work("data_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"transactions: {meta}")
    return txns


def stage_alerts(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    txns = pd.read_parquet(cfg.work("transactions.parquet"))
    alerts = run_rules(txns, cfg.rules)
    labels = make_labels(alerts, txns, cfg.labels, cfg.seed)
    alerts.to_parquet(cfg.work("alerts.parquet"))
    labels.to_parquet(cfg.work("labels.parquet"))
    # how good are the rules on their own? (the baseline the triage model must improve on)
    touched = labels["label"].sum()
    bad_acct_days = txns.loc[txns["is_laundering"], ["src", "day"]].drop_duplicates().shape[0]
    stats = {
        "n_alerts": len(alerts), "alerts_per_day": round(len(alerts) / max(alerts["day"].nunique(), 1), 1),
        "positive_alerts": int(touched), "alert_precision": round(float(labels["label"].mean()), 4),
        "false_positive_rate_of_rules": round(1 - float(labels["label"].mean()), 4),
        "laundering_sender_days": int(bad_acct_days),
        "per_rule": {r: {"alerts": int(alerts[f"r_{r}"].sum()),
                         "precision": round(float(labels.loc[alerts[f"r_{r}"].to_numpy(), "label"].mean()), 4)}
                     for r in [c[2:] for c in alerts.columns if c.startswith("r_")]},
    }
    cfg.work("alert_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"alerts: {len(alerts):,} ({stats['alerts_per_day']}/day), positives {touched:,} "
        f"-> rules alone are {stats['false_positive_rate_of_rules']:.1%} false positives")
    return alerts, labels


def stage_features(cfg: Config) -> pd.DataFrame:
    from aml_triage.features.build import build_features

    txns = pd.read_parquet(cfg.work("transactions.parquet"))
    alerts = pd.read_parquet(cfg.work("alerts.parquet"))
    labels = pd.read_parquet(cfg.work("labels.parquet"))
    t0 = time.time()
    table, stage1 = build_features(txns, alerts, labels, cfg)
    table.to_parquet(cfg.work("features.parquet"))
    if stage1 is not None:
        joblib.dump(stage1, cfg.work("stage1.joblib"))
    n_feat = sum(1 for c in table.columns if c.split("_")[0] in ("a", "p", "g", "gw", "gwd"))
    log(f"features: {table.shape[0]:,} alerts x {n_feat} features in {time.time() - t0:.0f}s")
    return table


def run_all(cfg: Config, stages: list[str]) -> None:
    def stage_train(c):
        from aml_triage.training.experiment import stage_train as f
        return f(c)

    def stage_score(c):
        from aml_triage.training.experiment import stage_score as f
        return f(c)

    def stage_backtest(c):
        from aml_triage.evaluation.backtest import stage_backtest as f
        return f(c)

    Path(cfg.paths.reports_dir, cfg.run_name).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.paths.reports_dir, cfg.run_name, "config_resolved.yaml"))
    fns = {"ingest": stage_ingest, "alerts": stage_alerts, "features": stage_features,
           "train": stage_train, "score": stage_score, "backtest": stage_backtest}
    for s in stages:
        t0 = time.time()
        log(f"=== stage {s}")
        fns[s](cfg)
        log(f"=== stage {s} done in {time.time() - t0:.0f}s")
    np.random.seed(cfg.seed)
