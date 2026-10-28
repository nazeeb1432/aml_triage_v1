"""Unit tests for the reader, rules, metrics, policy and augmentation guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml_triage.augment.synthesizers import augment
from aml_triage.config import PolicyCfg, RulesCfg
from aml_triage.evaluation.metrics import operating_point, recall_vs_review_share, threshold_at_fpr
from aml_triage.ingest.ibm import attach_typology, read_ibm_patterns, read_ibm_transactions
from aml_triage.rules.engine import run_rules
from aml_triage.serving.policy import route
from aml_triage.serving.reasons import describe

IBM_CSV = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,10,8000EBD30,10,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0
2022/09/01 00:20,3208,8000F4580,1,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0
2022/09/02 05:14,011,8139F54E0,0111632,8062C56E0,5000.00,Euro,5000.00,Euro,ACH,1
"""
IBM_PAT = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/02 05:14,011,8139F54E0,0111632,8062C56E0,5000.00,Euro,5000.00,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT
"""


def test_ibm_reader(tmp_path):
    (tmp_path / "t.csv").write_text(IBM_CSV)
    (tmp_path / "p.txt").write_text(IBM_PAT)
    t = read_ibm_transactions(tmp_path / "t.csv")
    assert list(t["day"]) == [0, 0, 1]
    assert t.loc[2, "src"] == "11_8139F54E0" and t.loc[2, "dst"] == "111632_8062C56E0"
    assert t.loc[2, "amount_usd"] == pytest.approx(5000 * 1.08)
    assert t["is_laundering"].tolist() == [False, False, True]
    t = attach_typology(t, read_ibm_patterns(tmp_path / "p.txt"))
    assert t.loc[2, "typology"] == "FAN-OUT" and t.loc[0, "typology"] is None


def _txns(rows):
    df = pd.DataFrame(rows, columns=["day", "src", "dst", "amount_usd", "pay_format"])
    df["ts"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["day"], unit="D")
    df["is_laundering"] = False
    df["cross_currency"] = False
    df["txn_id"] = np.arange(len(df))
    return df


def test_rules_fire_on_their_patterns():
    rows = [(0, "A", "B", 12000, "Wire")]                                  # large
    rows += [(1, "C", f"X{i}", 100, "ACH") for i in range(7)]               # fan-out
    rows += [(2, "D", "E", 9000, "ACH"), (2, "D", "F", 9500, "ACH")]        # structuring
    rows += [(3, "G", "H", 5000, "ACH"), (3, "H", "I", 4500, "ACH")]        # pass-through at H
    a = run_rules(_txns(rows), RulesCfg())
    fired = {(r.account, r.day): set(r.rules.split("|")) for r in a.itertuples()}
    assert "large_txn" in fired[("A", 0)]
    assert "fan_out" in fired[("C", 1)]
    assert "structuring" in fired[("D", 2)]
    assert "pass_through" in fired[("H", 3)]


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.02
    s = rng.random(5000) + y
    tau = threshold_at_fpr(y, s, 0.2)
    assert operating_point(y, s, tau)["fpr"] <= 0.2 + 1e-9
    curve = recall_vs_review_share(y, s)
    assert all(b["recall"] >= a["recall"] for a, b in zip(curve, curve[1:], strict=False))


def test_policy_overrides_btl_and_determinism():
    alerts = pd.DataFrame({"alert_id": np.arange(2000), "rules": ["spike"] * 1999 + ["structuring"],
                           "gw_prior_confirmed": [0] * 1998 + [2, 0]})
    raw = np.linspace(0, 1, 2000)
    cfg = PolicyCfg()
    r1 = route(alerts, raw, 0.8, cfg, seed=1)
    r2 = route(alerts, raw, 0.8, cfg, seed=1)
    assert r1.equals(r2), "routing must be replayable for audit"
    assert r1.loc[1999, "decision_reason"] == "hard_override"   # never-suppress rule
    assert r1.loc[1998, "decision_reason"] == "hard_override"   # prior confirmed suspicious
    assert (r1.loc[raw >= 0.8, "decision"] == "REVIEW").all()
    btl_share = (r1.loc[(raw >= 0.4) & (raw < 0.8), "decision"] == "BTL_REVIEW").mean()
    assert 0.05 < btl_share < 0.16
    prio = route(alerts, raw, 0.8, PolicyCfg(mode="prioritise"), seed=1)
    assert "SUPPRESS" not in set(prio["decision"])


def test_augment_refuses_non_training_rows():
    X = pd.DataFrame({"x": np.arange(10.0), "split": ["train"] * 9 + ["test"]})
    with pytest.raises(ValueError):
        augment(X, pd.Series([0] * 10), "smote", 0.2, 0)


def test_smote_rows_are_flagged():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    X.iloc[0, 0] = np.nan
    y = pd.Series((rng.random(400) < 0.05).astype(int))
    Xa, ya, flag = augment(X, y, "smote", 0.2, 0)
    assert flag.sum() == len(Xa) - len(X) > 0
    assert ya[flag].eq(1).all() and not Xa.isna().any().any()
    assert ya.mean() == pytest.approx(0.2, abs=0.02)


def test_reason_text():
    assert "random walks" in describe("gw_hit_rate", 0.4)
    assert "No random walk" in describe("gw_min", float("nan"))
    assert describe("p_s_sum_7d", 1234.0) == "Sent $1,234 over the last 7 day(s)"
