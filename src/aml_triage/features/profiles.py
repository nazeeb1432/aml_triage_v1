"""Entity-centric features (Eddin §2.2 "profiles") and alert features (Bakry et al. 2023).

Profiles: rolling sent/received aggregates per account over several day windows that END
on the alert day (end-of-day batch, so the alert day's own transactions are known), plus
ratios/differences between the short and long windows.

Alert features: which rules fired and by how much, and how often the account alerted before.
Nothing here reads labels.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from aml_triage.rules.engine import RULE_NAMES


def _daily(txns: pd.DataFrame, high_risk_formats: list[str]) -> pl.DataFrame:
    t = pl.from_pandas(txns[["day", "src", "dst", "amount_usd", "pay_format", "cross_currency"]])
    hr = pl.col("pay_format").is_in(high_risk_formats)
    out = t.group_by(["src", "day"]).agg(
        pl.len().alias("s_cnt"), pl.col("amount_usd").sum().alias("s_sum"),
        pl.col("amount_usd").max().alias("s_max"), pl.col("amount_usd").min().alias("s_min"),
        hr.sum().alias("s_hr_cnt"), pl.col("cross_currency").sum().alias("s_xcur_cnt"),
    ).rename({"src": "account"})
    inn = t.group_by(["dst", "day"]).agg(
        pl.len().alias("r_cnt"), pl.col("amount_usd").sum().alias("r_sum"),
        pl.col("amount_usd").max().alias("r_max"), pl.col("amount_usd").min().alias("r_min"),
        hr.sum().alias("r_hr_cnt"),
    ).rename({"dst": "account"})
    d = out.join(inn, on=["account", "day"], how="full", coalesce=True)
    zero_cols = ["s_cnt", "s_sum", "s_hr_cnt", "s_xcur_cnt", "r_cnt", "r_sum", "r_hr_cnt"]
    return d.with_columns([pl.col(c).fill_null(0) for c in zero_cols]).sort(["account", "day"])


def profile_features(txns: pd.DataFrame, alerts: pd.DataFrame, windows: list[int],
                     high_risk_formats: list[str]) -> pd.DataFrame:
    daily = _daily(txns, high_risk_formats)
    targets = pl.from_pandas(alerts[["alert_id", "account", "day"]])
    first_seen = daily.group_by("account").agg(pl.col("day").min().alias("first_day"))
    feats = targets.join(first_seen, on="account", how="left").with_columns(
        # capped at the longest window: "new account" is a risk signal, but uncapped age just counts calendar days
        pl.min_horizontal(pl.col("day") - pl.col("first_day"), pl.lit(max(windows))).alias("p_account_age_days")
    ).drop("first_day")

    for w in windows:
        roll = daily.rolling(index_column="day", period=f"{w}i", closed="right", group_by="account").agg(
            pl.col("s_sum").sum().alias(f"p_s_sum_{w}d"), pl.col("s_cnt").sum().alias(f"p_s_cnt_{w}d"),
            pl.col("s_max").max().alias(f"p_s_max_{w}d"), pl.col("s_min").min().alias(f"p_s_min_{w}d"),
            pl.col("r_sum").sum().alias(f"p_r_sum_{w}d"), pl.col("r_cnt").sum().alias(f"p_r_cnt_{w}d"),
            pl.col("r_max").max().alias(f"p_r_max_{w}d"), pl.col("r_min").min().alias(f"p_r_min_{w}d"),
            pl.col("s_hr_cnt").sum().alias(f"p_s_hr_cnt_{w}d"), pl.col("r_hr_cnt").sum().alias(f"p_r_hr_cnt_{w}d"),
            pl.col("s_xcur_cnt").sum().alias(f"p_s_xcur_cnt_{w}d"),
            pl.len().alias(f"p_active_days_{w}d"),
        )
        feats = feats.join(roll, on=["account", "day"], how="left")
        # rolling float sums drift (e.g. 1e-12 left after values leave the window): zero them when the count is 0
        feats = feats.with_columns(
            pl.when(pl.col(f"p_s_cnt_{w}d") > 0).then(pl.col(f"p_s_sum_{w}d")).otherwise(0.0).alias(f"p_s_sum_{w}d"),
            pl.when(pl.col(f"p_r_cnt_{w}d") > 0).then(pl.col(f"p_r_sum_{w}d")).otherwise(0.0).alias(f"p_r_sum_{w}d"),
        )
        feats = feats.with_columns(
            pl.when(pl.col(f"p_s_cnt_{w}d") > 0).then(pl.col(f"p_s_sum_{w}d") / pl.col(f"p_s_cnt_{w}d")).alias(f"p_s_mean_{w}d"),
            pl.when(pl.col(f"p_r_cnt_{w}d") > 0).then(pl.col(f"p_r_sum_{w}d") / pl.col(f"p_r_cnt_{w}d")).alias(f"p_r_mean_{w}d"),
            (pl.col(f"p_r_sum_{w}d") - pl.col(f"p_s_sum_{w}d")).alias(f"p_net_flow_{w}d"),
            (pl.col(f"p_s_sum_{w}d") / (pl.col(f"p_r_sum_{w}d") + 1.0)).alias(f"p_pass_ratio_{w}d"),
        )
    short = windows[0]
    for w in windows[1:]:
        feats = feats.with_columns(
            (pl.col(f"p_s_sum_{short}d") / (pl.col(f"p_s_sum_{w}d") + 1.0)).alias(f"p_s_sum_ratio_{short}_{w}"),
            (pl.col(f"p_r_sum_{short}d") / (pl.col(f"p_r_sum_{w}d") + 1.0)).alias(f"p_r_sum_ratio_{short}_{w}"),
            (pl.col(f"p_s_cnt_{short}d") / (pl.col(f"p_s_cnt_{w}d") + 1.0)).alias(f"p_s_cnt_ratio_{short}_{w}"),
            (pl.col(f"p_s_mean_{short}d") - pl.col(f"p_s_mean_{w}d")).alias(f"p_s_mean_diff_{short}_{w}"),
            (pl.col(f"p_r_mean_{short}d") - pl.col(f"p_r_mean_{w}d")).alias(f"p_r_mean_diff_{short}_{w}"),
        )
    out = feats.drop(["account", "day"]).to_pandas()
    num = out.select_dtypes("number").columns
    out[num] = out[num].replace([float("inf"), float("-inf")], float("nan"))
    return out


def alert_features(alerts: pd.DataFrame, daily_stats: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """Rule flags + rule intensities + the account's alert history (past alerts only)."""
    a = alerts[["alert_id", "account", "day", "n_rules"] + [f"r_{r}" for r in RULE_NAMES]].copy()
    s = daily_stats[["account", "day", "max_out", "max_in", "n_dst", "n_src", "n_struct",
                     "sum_out", "sum_in", "sum_high_risk", "trail_mean_out", "n_low_out", "n_low_in"]].copy()
    s["n_low"] = s["n_low_out"] + s["n_low_in"]
    s = s.drop(columns=["n_low_out", "n_low_in"])
    a = a.merge(s, on=["account", "day"], how="left")
    a["a_spike_ratio"] = a["sum_out"] / (a["trail_mean_out"] + 1.0)
    a["a_pass_ratio"] = a["sum_out"] / (a["sum_in"] + 1.0)
    # prior alerts on the same account in (day - lookback, day): strictly earlier days only
    pa = pl.from_pandas(alerts[["account", "day"]]).sort(["account", "day"]).with_columns(pl.lit(1).alias("one"))
    prior = pa.rolling(index_column="day", period=f"{lookback_days}i", closed="left", group_by="account").agg(
        pl.col("one").sum().alias("a_prior_alerts_30d")
    ).to_pandas()
    a = a.merge(prior, on=["account", "day"], how="left")
    a["a_prior_alerts_30d"] = a["a_prior_alerts_30d"].fillna(0)
    rename = {c: f"a_{c}" for c in ["n_rules", "max_out", "max_in", "n_dst", "n_src", "n_struct",
                                     "sum_out", "sum_in", "sum_high_risk", "n_low"]}
    rename.update({f"r_{r}": f"a_rule_{r}" for r in RULE_NAMES})
    a = a.rename(columns=rename).drop(columns=["trail_mean_out"])
    for c in [f"a_rule_{r}" for r in RULE_NAMES]:
        a[c] = a[c].astype("int8")
    return a.drop(columns=["account", "day"])
