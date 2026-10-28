"""Rule engine: the existing, regulator-approved monitoring system the triage model sits behind.

Rules run end-of-day on (account, day) aggregates and are intentionally simple and noisy.
The triage model never creates alerts; it only ranks / suppresses the alerts produced here,
so every alert stays explainable by the rule that raised it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from aml_triage.config import RulesCfg
from aml_triage.contracts import ALERTS, validate

RULE_NAMES = ["large_txn", "structuring", "fan_out", "fan_in", "spike", "high_risk_channel", "pass_through", "low_value"]


def daily_account_stats(txns: pd.DataFrame, rules: RulesCfg) -> pl.DataFrame:
    t = pl.from_pandas(txns[["day", "src", "dst", "amount_usd", "pay_format"]])
    out = t.group_by(["src", "day"]).agg(
        pl.len().alias("n_out"),
        pl.col("amount_usd").sum().alias("sum_out"),
        pl.col("amount_usd").max().alias("max_out"),
        pl.col("dst").n_unique().alias("n_dst"),
        ((pl.col("amount_usd") >= rules.structuring_low_usd) & (pl.col("amount_usd") < rules.structuring_high_usd))
        .sum().alias("n_struct"),
        pl.when(pl.col("pay_format").is_in(rules.high_risk_formats)).then(pl.col("amount_usd")).otherwise(0.0)
        .sum().alias("sum_high_risk"),
        (pl.col("amount_usd") < rules.low_value_usd).sum().alias("n_low_out"),
    ).rename({"src": "account"})
    inn = t.group_by(["dst", "day"]).agg(
        pl.len().alias("n_in"),
        pl.col("amount_usd").sum().alias("sum_in"),
        pl.col("amount_usd").max().alias("max_in"),
        pl.col("src").n_unique().alias("n_src"),
        (pl.col("amount_usd") < rules.low_value_usd).sum().alias("n_low_in"),
    ).rename({"dst": "account"})
    daily = out.join(inn, on=["account", "day"], how="full", coalesce=True).fill_null(0)
    daily = daily.sort(["account", "day"])
    trailing = (
        daily.filter(pl.col("sum_out") > 0)
        .rolling(index_column="day", period=f"{rules.spike_window_days}i", closed="left", group_by="account")
        .agg(pl.col("sum_out").mean().alias("trail_mean_out"))
    )
    return daily.join(trailing, on=["account", "day"], how="left")


def run_rules(txns: pd.DataFrame, rules: RulesCfg) -> pd.DataFrame:
    d = daily_account_stats(txns, rules)
    flags = d.with_columns(
        r_large_txn=(pl.max_horizontal("max_out", "max_in") >= rules.large_txn_usd),
        r_structuring=(pl.col("n_struct") >= rules.structuring_min_count),
        r_fan_out=(pl.col("n_dst") >= rules.fan_out_min_counterparties),
        r_fan_in=(pl.col("n_src") >= rules.fan_in_min_counterparties),
        r_spike=((pl.col("sum_out") > rules.spike_multiplier * pl.col("trail_mean_out"))
                 & (pl.col("sum_out") >= rules.spike_min_usd)).fill_null(False),
        r_high_risk_channel=(pl.col("sum_high_risk") >= rules.high_risk_channel_min_usd),
        r_pass_through=((pl.col("sum_in") >= rules.pass_through_min_in_usd)
                        & (pl.col("sum_out") >= rules.pass_through_ratio * pl.col("sum_in"))),
        r_low_value=((pl.col("n_low_out") + pl.col("n_low_in")) >= rules.low_value_min_count),
    )
    rcols = [f"r_{r}" for r in RULE_NAMES]
    alerts = flags.filter(pl.any_horizontal(rcols)).to_pandas()
    alerts = alerts.sort_values(["day", "account"]).reset_index(drop=True)
    alerts.insert(0, "alert_id", np.arange(len(alerts), dtype="int64"))
    alerts["n_rules"] = alerts[rcols].sum(axis=1).astype("int64")
    names = np.array(RULE_NAMES)
    alerts["rules"] = ["|".join(names[row]) for row in alerts[rcols].to_numpy()]
    keep = ["alert_id", "account", "day", "rules", "n_rules", *rcols]
    return validate(alerts[keep], ALERTS, "alerts")
