"""Reason codes: turn per-alert SHAP contributions into sentences an analyst can check.

LightGBM computes exact TreeSHAP contributions itself (`pred_contrib=True`), so no extra
dependency is needed. Each feature name maps to a template; the value shown is the
alert's own value, so every reason can be verified against the transactions.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

RULE_TEXT = {
    "large_txn": "a single payment over the large-transaction limit",
    "structuring": "several payments just under the reporting threshold",
    "fan_out": "payments to many different accounts in one day",
    "fan_in": "payments from many different accounts in one day",
    "spike": "a sudden jump in outgoing volume",
    "high_risk_channel": "large cash or crypto movements",
    "pass_through": "money paid out almost as fast as it came in",
    "low_value": "very small transfers (account testing or micro-structuring)",
}


def _money(v: float) -> str:
    return f"${v:,.0f}"


def describe(feature: str, v: float) -> str:  # noqa: C901 - a flat lookup reads better than a dispatch table here
    nan = v is None or (isinstance(v, float) and math.isnan(v))
    if feature.startswith("a_rule_"):
        r = feature[len("a_rule_"):]
        return f"Rule fired: {RULE_TEXT.get(r, r)}" if v else f"Rule did not fire: {RULE_TEXT.get(r, r)}"
    fixed = {
        "a_n_rules": lambda: f"{v:.0f} rule(s) fired on this alert",
        "a_max_out": lambda: f"Largest outgoing payment today: {_money(v)}",
        "a_max_in": lambda: f"Largest incoming payment today: {_money(v)}",
        "a_n_dst": lambda: f"Paid {v:.0f} different accounts today",
        "a_n_src": lambda: f"Received from {v:.0f} different accounts today",
        "a_n_struct": lambda: f"{v:.0f} payment(s) just under the reporting threshold today",
        "a_n_low": lambda: f"{v:.0f} very small transfer(s) today",
        "a_sum_out": lambda: f"Sent {_money(v)} today",
        "a_sum_in": lambda: f"Received {_money(v)} today",
        "a_sum_high_risk": lambda: f"{_money(v)} moved by cash or crypto today",
        "a_spike_ratio": lambda: f"Outgoing volume is {v:.1f}x its recent daily average",
        "a_pass_ratio": lambda: f"Paid out {v:.0%} of what came in today",
        "a_prior_alerts_30d": lambda: f"{v:.0f} earlier alert(s) on this account in the last 30 days",
        "p_account_age_days": lambda: f"Account first seen {v:.0f} days ago",
        "g_in_deg": lambda: f"Received from {v:.0f} accounts in the recent graph window",
        "g_out_deg": lambda: f"Paid {v:.0f} accounts in the recent graph window",
        "g_in_w": lambda: f"Received {_money(v)} in the recent graph window",
        "g_out_w": lambda: f"Sent {_money(v)} in the recent graph window",
        "g_n_neighbours": lambda: f"{v:.0f} direct counterparties in the graph window",
        "g_nb_in_deg_max": lambda: (f"A counterparty receives from {v:.0f} accounts (collector pattern)" if v > 1
                                    else "No counterparty collects from other accounts"),
        "g_nb_out_deg_max": lambda: (f"A counterparty pays {v:.0f} accounts (distributor pattern)" if v > 1
                                     else "No counterparty passes money on to other accounts"),
        "g_nb_in_deg_mean": lambda: f"Counterparties receive from {v:.1f} accounts on average",
        "g_nb_out_deg_mean": lambda: f"Counterparties pay {v:.1f} accounts on average",
    }
    if feature in fixed and not nan:
        return fixed[feature]()
    m = re.match(r"(gwd?)_(.+)", feature)
    if m:
        who = "a confirmed or pending-suspicious account" if m.group(1) == "gwd" else "a confirmed-suspicious account"
        stat = m.group(2)
        if stat == "prior_confirmed":
            return f"This account had {v:.0f} earlier alert(s) confirmed suspicious"
        if nan or (stat == "hit_rate" and v == 0):
            return f"No random walk reached {who} within the step limit"
        if stat == "hit_rate":
            return f"{v:.0%} of random walks reach {who}"
        if stat == "n_distinct":
            return f"Walks reach {v:.0f} different suspicious account(s)"
        if stat == "min":
            return f"Nearest {who.split(' ', 1)[1]} is {v:.0f} hop(s) away"
        return f"Typical distance to {who}: {v:.1f} hops ({stat})"
    m = re.match(r"p_(s|r)_(sum|cnt|max|min|mean|hr_cnt)_(\d+)d", feature)
    if m and not nan:
        verb = "Sent" if m.group(1) == "s" else "Received"
        w, stat = m.group(3), m.group(2)
        return {
            "sum": f"{verb} {_money(v)} over the last {w} day(s)",
            "cnt": f"{verb} {v:.0f} payment(s) over the last {w} day(s)",
            "max": f"Largest payment {verb.lower()} in {w} day(s): {_money(v)}",
            "min": f"Smallest payment {verb.lower()} in {w} day(s): {_money(v)}",
            "mean": f"Average payment {verb.lower()} over {w} day(s): {_money(v)}",
            "hr_cnt": f"{v:.0f} cash/crypto payment(s) {verb.lower()} over {w} day(s)",
        }[stat]
    m = re.match(r"p_net_flow_(\d+)d", feature)
    if m and not nan:
        return f"Net {'inflow' if v >= 0 else 'outflow'} of {_money(abs(v))} over {m.group(1)} day(s)"
    m = re.match(r"p_pass_ratio_(\d+)d", feature)
    if m and not nan:
        return f"Sent out {v:.0%} of what was received over {m.group(1)} day(s)"
    m = re.match(r"p_(s|r)_(sum|cnt)_ratio_(\d+)_(\d+)", feature)
    if m and not nan:
        verb = "sending" if m.group(1) == "s" else "receiving"
        return f"Last {m.group(3)} day(s) of {verb} = {v:.0%} of the {m.group(4)}-day total"
    m = re.match(r"p_(s|r)_mean_diff_(\d+)_(\d+)", feature)
    if m and not nan:
        verb = "sent" if m.group(1) == "s" else "received"
        return f"Average payment {verb} is {_money(abs(v))} {'above' if v >= 0 else 'below'} its {m.group(3)}-day average"
    m = re.match(r"p_active_days_(\d+)d", feature)
    if m and not nan:
        return f"Active on {v:.0f} of the last {m.group(1)} day(s)"
    m = re.match(r"p_s_xcur_cnt_(\d+)d", feature)
    if m and not nan:
        return f"{v:.0f} cross-currency payment(s) over {m.group(1)} day(s)"
    return f"{feature} = {'missing' if nan else f'{v:.3g}'}"


def reason_codes(model, X: pd.DataFrame, features: list[str], top_k: int = 4, fill_nan: bool = False) -> list[list[dict]]:
    Xf = X[features].astype("float64")
    if fill_nan:
        Xf = Xf.fillna(-1.0)
    contrib = model.booster_.predict(Xf, pred_contrib=True)[:, :-1]  # last column is the bias
    vals = X[features].to_numpy(dtype="float64")
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(contrib[i]))[:top_k]
        out.append([
            {"feature": features[j], "value": None if np.isnan(vals[i, j]) else float(vals[i, j]),
             "impact": float(contrib[i, j]), "direction": "raises" if contrib[i, j] > 0 else "lowers",
             "text": describe(features[j], vals[i, j])}
            for j in order
        ])
    return out
