"""Bitemporal label store.

Every label has two times:
  event_day     the alert day it describes
  decision_day  the day the analyst decision was recorded (knowledge time)

Anything computed for day d may only use labels with decision_day <= d. That single rule
is what the original replication broke (GuiltyWalker used every label from the whole year).

Real banks get labels from case dispositions. Public data (IBM AMLworld) only has
transaction-level `Is Laundering`, so we derive alert labels from it and *simulate*
decision delays: positives (escalation -> L2 -> SAR) take longer than L1 closures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml_triage.config import LabelsCfg
from aml_triage.contracts import LABELS, validate


def laundering_touch_days(txns: pd.DataFrame) -> pd.DataFrame:
    """(account, day, typology) for every account that sent or received a laundering transaction."""
    bad = txns.loc[txns["is_laundering"]]
    typ = bad["typology"] if "typology" in bad else pd.Series([None] * len(bad), index=bad.index)
    a = pd.DataFrame({"account": bad["src"], "day": bad["day"], "typology": typ})
    b = pd.DataFrame({"account": bad["dst"], "day": bad["day"], "typology": typ})
    return pd.concat([a, b]).drop_duplicates(["account", "day"])


def make_labels(alerts: pd.DataFrame, txns: pd.DataFrame, cfg: LabelsCfg, seed: int) -> pd.DataFrame:
    touch = laundering_touch_days(txns)
    hits = []
    for lag in range(cfg.lookback_days + 1):
        t = touch.assign(day=touch["day"] + lag)
        hits.append(alerts[["alert_id", "account", "day"]].merge(t, on=["account", "day"], how="inner"))
    hit = pd.concat(hits).drop_duplicates("alert_id").set_index("alert_id")
    lab = alerts[["alert_id", "account", "day"]].rename(columns={"day": "event_day"}).copy()
    lab["label"] = lab["alert_id"].isin(hit.index)
    lab["typology"] = lab["alert_id"].map(hit["typology"])

    rng = np.random.default_rng(seed + 7)
    med = np.where(lab["label"], cfg.delay_median_days_pos, cfg.delay_median_days_neg)
    delay = np.ceil(rng.lognormal(np.log(np.maximum(med, 0.5)), cfg.delay_sigma)).astype("int64")
    lab["decision_day"] = lab["event_day"].astype("int64") + delay
    return validate(lab, LABELS, "labels")


class LabelIndex:
    """Fast as-of lookups: which accounts are confirmed suspicious, as known on day d."""

    def __init__(self, labels: pd.DataFrame, account_ids: dict[str, int]):
        pos = labels.loc[labels["label"]].sort_values("decision_day")
        self.pos_days = pos["decision_day"].to_numpy()
        self.pos_nodes = pos["account"].map(account_ids).fillna(-1).astype("int64").to_numpy()
        self.n_nodes = len(account_ids)

    def illicit_mask_as_of(self, day: int, memory_days: int | None = None) -> np.ndarray:
        """Accounts with a positive decision recorded on or before `day` (and, with memory_days, not older than that).
        A bounded memory keeps GuiltyWalker stationary: otherwise the known-bad set only ever grows and hit rates drift up."""
        k = np.searchsorted(self.pos_days, day, side="right")   # decision_day <= day
        k0 = 0 if memory_days is None else np.searchsorted(self.pos_days, day - memory_days, side="right")
        mask = np.zeros(self.n_nodes, dtype=np.bool_)
        nodes = self.pos_nodes[k0:k]
        mask[nodes[nodes >= 0]] = True
        return mask

    def prior_positive_count_as_of(self, day: int) -> np.ndarray:
        k = np.searchsorted(self.pos_days, day, side="right")
        nodes = self.pos_nodes[:k]
        return np.bincount(nodes[nodes >= 0], minlength=self.n_nodes)


def illicit_as_of(labels: pd.DataFrame, day: int) -> set[str]:
    """Readable version of LabelIndex.illicit_mask_as_of, used in tests and docs."""
    known = labels[(labels["decision_day"] <= day) & labels["label"]]
    return set(known["account"])
