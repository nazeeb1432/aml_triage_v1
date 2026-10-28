"""The most important tests in the repo: nothing computed for day d may depend on anything after d.

`test_features_do_not_see_the_future` rebuilds the world as it looked at the end of day d
(transactions up to d, only decisions recorded by d) and checks every feature of every alert
up to d is identical to the full-history computation. It catches label leakage, look-ahead
windows, and shared state between days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml_triage.features.build import FAMILIES, build_features
from aml_triage.labels.store import LabelIndex, illicit_as_of
from aml_triage.rules.engine import run_rules
from aml_triage.training.train import split


def _feature_cols(df):
    return [c for c in df.columns if any(c.startswith(p) for p in FAMILIES.values())]


@pytest.mark.parametrize("d", [35, 52])
def test_features_do_not_see_the_future(small_cfg, small_world, d):
    txns, alerts, labels = small_world
    full, stage1 = build_features(txns, alerts, labels, small_cfg)

    txns_d = txns[txns["day"] <= d]
    alerts_d = run_rules(txns_d, small_cfg.rules)
    # what the bank knows at end of day d: decisions recorded by d; the rest is pending (no decision yet)
    labels_d = labels[labels["event_day"] <= d].copy()
    pending = labels_d["decision_day"] > d
    labels_d["decision_day"] = labels_d["decision_day"].astype(float)
    labels_d.loc[pending, "decision_day"] = np.nan
    labels_d.loc[pending, "label"] = False
    cut, _ = build_features(txns_d, alerts_d, labels_d, small_cfg, stage1=stage1)

    a = full[full["day"] <= d].set_index(["account", "day"]).sort_index()
    b = cut.set_index(["account", "day"]).sort_index()
    assert a.index.equals(b.index), "rule engine produced different alerts once the future was removed"
    cols = _feature_cols(a)
    for c in cols:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        assert np.allclose(x, y, equal_nan=True, rtol=1e-6, atol=1e-6), f"feature {c} changes when data after day {d} is removed"


def test_illicit_set_is_as_of(small_world):
    _, _, labels = small_world
    accounts = pd.unique(labels["account"])
    idx = LabelIndex(labels, {a: i for i, a in enumerate(accounts)})
    for d in (10, 30, 60):
        expected = illicit_as_of(labels, d)
        got = {accounts[i] for i in np.flatnonzero(idx.illicit_mask_as_of(d))}
        assert got == expected
        known = labels[labels["account"].isin(got) & labels["label"]]
        assert known["decision_day"].min() <= d


def test_own_label_is_never_known_on_alert_day(small_world):
    _, _, labels = small_world
    assert (labels["decision_day"] > labels["event_day"]).all()


def test_split_respects_label_delay(small_cfg, small_world):
    _, _, labels = small_world
    t = labels.rename(columns={"event_day": "day"})
    s = split(t, small_cfg)
    train = t[s == "train"]
    assert (train["decision_day"] < small_cfg.split.val_start_day).all()
    assert (t.loc[s == "val", "decision_day"] < small_cfg.split.test_start_day).all()
    late = t[(t["day"] < small_cfg.split.val_start_day) & (t["day"] >= small_cfg.split.warmup_days)
             & (t["decision_day"] >= small_cfg.split.val_start_day)]
    assert len(late) > 0 and (s[late.index] == "drop").all()
