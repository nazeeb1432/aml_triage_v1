"""Assemble the alert feature table: one row per alert, families distinguished by prefix.

    a_    alert / rule features            (no labels)
    p_    profiles                         (no labels)
    g_    degree features                  (no labels)
    gw_   GuiltyWalker + prior confirmed   (labels as of the alert day)
    gwd_  GuiltyWalker-delay               (labels as of the day + stage-1 pseudo labels)
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from aml_triage.config import Config
from aml_triage.features.graph import graph_features
from aml_triage.features.profiles import alert_features, profile_features
from aml_triage.rules.engine import daily_account_stats

FAMILIES = {"alert": "a_", "profiles": "p_", "degree": "g_", "gw": "gw_", "gwd": "gwd_"}
META = ["alert_id", "account", "day", "rules", "label", "decision_day", "typology"]


def node_index(txns: pd.DataFrame) -> dict[str, int]:
    """Account -> int id in order of first appearance in time. Append-only: an account's id never changes when
    later data arrives, which keeps walk seeds and CSR order (and so GW features) stable day to day."""
    t = txns.sort_values("ts", kind="stable") if "ts" in txns else txns
    accts = pd.unique(np.column_stack([t["src"].to_numpy(), t["dst"].to_numpy()]).ravel())
    return {a: i for i, a in enumerate(accts)}


def columns_for(feature_set: str, columns: list[str]) -> list[str]:
    """'alert+profiles+degree' -> every column whose prefix belongs to one of those families."""
    fams = feature_set.split("+")
    prefixes = [FAMILIES[f] for f in fams]
    out = []
    for c in columns:
        # gw_ is a prefix of gwd_: match the longest family prefix
        fam = max((p for p in FAMILIES.values() if c.startswith(p)), key=len, default=None)
        if fam in prefixes:
            out.append(c)
    return out


def train_stage1(table: pd.DataFrame, cfg: Config) -> lgb.LGBMClassifier:
    """Pseudo-labeller for GWd. Trained ONLY on warm-up alerts whose labels were decided
    before the warm-up ended; those rows are never used again for training or evaluation."""
    w = cfg.split.warmup_days
    rows = table[(table["day"] < w) & (table["decision_day"] < w)]
    cols = columns_for("alert+profiles+degree", list(table.columns))
    if rows["label"].sum() < 3:
        raise ValueError(f"warm-up period has only {int(rows['label'].sum())} decided positives; widen split.warmup_days")
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=cfg.seed, verbosity=-1)
    m.fit(rows[cols], rows["label"].astype(int))
    m.feature_cols_ = cols
    return m


def build_features(txns: pd.DataFrame, alerts: pd.DataFrame, labels: pd.DataFrame, cfg: Config,
                   stage1: lgb.LGBMClassifier | None = None) -> tuple[pd.DataFrame, lgb.LGBMClassifier | None]:
    f = cfg.features
    nodes = node_index(txns)
    daily = daily_account_stats(txns, cfg.rules).to_pandas()
    a = alert_features(alerts, daily)
    p = profile_features(txns, alerts, f.profile_windows_days, cfg.rules.high_risk_formats)
    g = graph_features(txns, alerts, labels, nodes, f, cfg.seed)
    meta = alerts[["alert_id", "account", "day", "rules"]].merge(
        labels[["alert_id", "label", "decision_day", "typology"]], on="alert_id", how="left")
    table = meta.merge(a, on="alert_id").merge(p, on="alert_id").merge(g, on="alert_id")

    if f.gwd.enabled:
        if stage1 is None:
            stage1 = train_stage1(table, cfg)
        score = stage1.predict_proba(table[stage1.feature_cols_])[:, 1]
        # warm-up alerts were the stage-1 training data, so their scores are in-sample: never pseudo-label them
        pseudo = (score >= f.gwd.pseudo_label_threshold) & (table["day"].to_numpy() >= cfg.split.warmup_days)
        extra = table[["account", "day", "decision_day"]].assign(pseudo_pos=pseudo)
        gd = graph_features(txns, alerts, labels, nodes, f, cfg.seed, hybrid_extra=extra, base=False)
        table = table.merge(gd, on="alert_id")
    return table.sort_values(["day", "alert_id"]).reset_index(drop=True), stage1
