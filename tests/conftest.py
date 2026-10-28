"""A small, deterministic world shared by the tests (generated once per session, ~20s)."""

from __future__ import annotations

import pandas as pd
import pytest

from aml_triage.config import load_config

SMALL = [
    "run_name=pytest",
    "data.synthetic.n_accounts=1500",
    "data.synthetic.n_days=70",
    "data.synthetic.laundering_patterns=90",
    "data.synthetic.out_trans_file=PYTEST_Trans.csv",
    "data.synthetic.out_patterns_file=PYTEST_Patterns.txt",
    "features.profile_windows_days=[1,3,7]",
    "features.graph.tws_days=10",
    "features.guilty_walker.n_walks=20",
    "split.warmup_days=20",
    "split.val_start_day=45",
    "split.test_start_day=55",
    "labels.delay_median_days_pos=5",
    "labels.delay_median_days_neg=2",
    "training.n_trials=2",
    "training.timeout_seconds=30",
    "evaluation.bootstrap_reps=20",
    "augmentation.methods=[none,class_weight,smote]",
]


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    return load_config("configs/default.yaml", overrides=SMALL + [
        f"data.raw_dir={root / 'raw'}", f"paths.work_dir={root / 'work'}",
        f"paths.artifacts_dir={root / 'artifacts'}", f"paths.reports_dir={root / 'reports'}",
    ])


@pytest.fixture(scope="session")
def small_world(small_cfg):
    from aml_triage.pipeline import stage_alerts, stage_ingest

    stage_ingest(small_cfg)
    stage_alerts(small_cfg)
    txns = pd.read_parquet(small_cfg.work("transactions.parquet"))
    alerts = pd.read_parquet(small_cfg.work("alerts.parquet"))
    labels = pd.read_parquet(small_cfg.work("labels.parquet"))
    return txns, alerts, labels


@pytest.fixture(scope="session")
def trained_run(small_cfg, small_world):
    """Run features -> train -> score on the small world; returns the reports dir."""
    from aml_triage.pipeline import run_all

    run_all(small_cfg, ["features", "train", "score"])
    from pathlib import Path

    return Path(small_cfg.paths.reports_dir) / small_cfg.run_name
