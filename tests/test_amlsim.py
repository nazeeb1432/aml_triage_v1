"""AMLSim reader and an end-to-end run of the AMLSim profile on a small world written in AMLSim's format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aml_triage.config import load_config
from aml_triage.ingest.amlsim import read_amlsim

# first rows of the real Kaggle files (CRLF line endings, as published)
TXNS = ("TX_ID,SENDER_ACCOUNT_ID,RECEIVER_ACCOUNT_ID,TX_TYPE,TX_AMOUNT,TIMESTAMP,IS_FRAUD,ALERT_ID\r\n"
        "1,6456,9069,TRANSFER,465.05,0,False,-1\r\n2,7516,9543,TRANSFER,564.64,0,False,-1\r\n"
        "82,6976,9739,TRANSFER,4.85,0,True,193\r\n949,5776,2570,TRANSFER,10.27,0,True,377\r\n"
        "6280,9999,9530,TRANSFER,2.74,1,True,189\r\n")
ALERTS = ("ALERT_ID,ALERT_TYPE,IS_FRAUD,TX_ID,SENDER_ACCOUNT_ID,RECEIVER_ACCOUNT_ID,TX_TYPE,TX_AMOUNT,TIMESTAMP\r\n"
          "193,fan_in,True,82,6976,9739,TRANSFER,4.85,0\r\n377,cycle,True,949,5776,2570,TRANSFER,10.27,0\r\n"
          "189,fan_in,True,6280,9999,9530,TRANSFER,2.74,1\r\n")


def test_amlsim_reader(tmp_path):
    (tmp_path / "transactions.csv").write_text(TXNS)
    (tmp_path / "alerts.csv").write_text(ALERTS)
    t = read_amlsim(tmp_path)
    assert list(t["day"]) == [0, 0, 0, 0, 1]
    assert t["ts"].is_monotonic_increasing and t["ts"].is_unique
    assert t["is_laundering"].sum() == 3
    assert t.set_index("txn_id").loc[949, "typology"] == "CYCLE"
    assert t.set_index("txn_id").loc[82, "typology"] == "FAN-IN"
    assert t.set_index("txn_id").loc[1, "typology"] is None
    assert t.loc[0, "src"] == "A6456" and t.loc[0, "pay_format"] == "Transfer"


def _write_amlsim_world(small_world, out: Path) -> None:
    """Re-express the shared small world in AMLSim's format, with laundering amounts shrunk under $25 as in AMLSim."""
    txns, _, _ = small_world
    rng = np.random.default_rng(0)
    amt = txns["amount_usd"].to_numpy().copy()
    bad = txns["is_laundering"].to_numpy()
    amt[bad] = rng.uniform(2, 20, bad.sum())
    ids = {a: i for i, a in enumerate(pd.unique(pd.concat([txns["src"], txns["dst"]])))}
    scheme = np.where(bad, rng.integers(0, 60, len(txns)), -1)
    t = pd.DataFrame({"TX_ID": np.arange(1, len(txns) + 1), "SENDER_ACCOUNT_ID": txns["src"].map(ids),
                      "RECEIVER_ACCOUNT_ID": txns["dst"].map(ids), "TX_TYPE": "TRANSFER", "TX_AMOUNT": amt.round(2),
                      "TIMESTAMP": txns["day"], "IS_FRAUD": bad, "ALERT_ID": scheme})
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "transactions.csv", index=False)
    a = t[t["IS_FRAUD"]].assign(ALERT_TYPE=np.where(t.loc[t["IS_FRAUD"], "ALERT_ID"] % 2 == 0, "cycle", "fan_in"))
    a[["ALERT_ID", "ALERT_TYPE", "IS_FRAUD", "TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_TYPE",
       "TX_AMOUNT", "TIMESTAMP"]].to_csv(out / "alerts.csv", index=False)


def test_amlsim_profile_end_to_end(small_world, tmp_path):
    from aml_triage.pipeline import run_all

    from .conftest import SMALL

    _write_amlsim_world(small_world, tmp_path / "amlsim")
    cfg = load_config("configs/default.yaml", "configs/amlsim.yaml", SMALL + [
        "run_name=pytest-amlsim", f"data.amlsim_dir={tmp_path / 'amlsim'}", f"paths.work_dir={tmp_path / 'work'}",
        f"paths.artifacts_dir={tmp_path / 'artifacts'}", f"paths.reports_dir={tmp_path / 'reports'}",
        "evaluation.backtest.cuts=[45,55]", "evaluation.backtest.val_days=10", "evaluation.backtest.test_days=10",
        "evaluation.backtest.seeds=[1]", "evaluation.backtest.bootstrap_reps=10",
    ])
    assert cfg.data.source == "amlsim" and cfg.rules.low_value_usd == 25
    run_all(cfg, ["ingest", "alerts", "features", "train", "score", "backtest"])
    stats = json.loads(cfg.work("alert_stats.json").read_text())
    assert stats["per_rule"]["low_value"]["alerts"] > 0 and stats["per_rule"]["pass_through"]["alerts"] == 0
    m = json.loads((tmp_path / "reports" / "pytest-amlsim" / "metrics.json").read_text())
    assert m["champion"]["recall"] > 0 and "robustness" in m
