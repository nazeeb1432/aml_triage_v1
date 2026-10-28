"""Pipeline + API smoke test on the small world."""

from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient


def test_pipeline_and_api(trained_run, monkeypatch, small_cfg):
    m = json.loads((trained_run / "metrics.json").read_text())
    assert m["ladder"] and m["champion"]["tau_raw"] > 0
    scored = pd.read_parquet(trained_run / "scored_alerts.parquet")
    assert (scored["day"] >= small_cfg.split.test_start_day).all()
    assert set(scored["decision"]) <= {"REVIEW", "SUPPRESS", "BTL_REVIEW"}

    monkeypatch.setenv("AML_TRIAGE_REPORTS", str(trained_run.parent))
    monkeypatch.setenv("AML_TRIAGE_RUN", trained_run.name)
    monkeypatch.setenv("AML_TRIAGE_ARTIFACTS", small_cfg.paths.artifacts_dir)
    monkeypatch.setenv("AML_TRIAGE_DB", str(trained_run / "audit.sqlite"))
    from aml_triage.api.app import create_app

    c = TestClient(create_app())
    assert c.get("/api/health").json()["status"] == "ok"
    q = c.get("/api/alerts", params={"view": "queue"}).json()
    assert q["items"] and "SUPPRESS" not in {i["decision"] for i in q["items"]}
    assert "label" not in q["items"][0], "ground truth must not leak into the analyst queue"
    aid = q["items"][0]["alert_id"]
    d = c.get(f"/api/alerts/{aid}").json()
    assert d["reasons"] and "graph" in d
    r = c.post(f"/api/alerts/{aid}/disposition", json={"action": "escalate", "analyst": "test", "note": "x"})
    assert r.status_code == 200
    assert c.get("/api/audit").json()["items"][0]["alert_id"] == aid
    sim = c.post("/api/policy/simulate", json={"review_share": 0.3}).json()
    assert 0 <= sim["recall"] <= 1
    assert c.get("/api/model").json()["ladder"]


def test_backtest_runs_and_reports(trained_run, small_cfg):
    from aml_triage.evaluation.backtest import stage_backtest

    cfg = small_cfg.model_copy(deep=True)
    cfg.evaluation.backtest.cuts = [45, 55]
    cfg.evaluation.backtest.val_days = 10
    cfg.evaluation.backtest.test_days = 10
    cfg.evaluation.backtest.bootstrap_reps = 20
    cfg.evaluation.backtest.seeds = [1, 2]
    cfg.evaluation.backtest.min_typology_positives = 3
    out = stage_backtest(cfg)
    assert out["seeds"] and len(out["seeds"]) == 2
    for f in out["folds"]:
        assert 0 <= f["recall"] <= 1
    assert (trained_run / "backtest.md").exists()
    assert "robustness" in __import__("json").loads((trained_run / "metrics.json").read_text())
