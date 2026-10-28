"""Data contracts: the columns and types each stage promises to the next.

A broken upstream feed must fail loudly here instead of turning into NaN features
that silently suppress real suspicious alerts.
"""

from __future__ import annotations

import pandas as pd


class ContractError(ValueError):
    pass


TRANSACTIONS = {
    "txn_id": "int",
    "ts": "datetime",
    "day": "int",
    "src": "object",
    "dst": "object",
    "amount_usd": "float",
    "pay_format": "object",
    "currency": "object",
    "is_laundering": "bool",
}

ALERTS = {
    "alert_id": "int",
    "account": "object",
    "day": "int",
    "rules": "object",
    "n_rules": "int",
}

LABELS = {
    "alert_id": "int",
    "account": "object",
    "event_day": "int",
    "decision_day": "int",
    "label": "bool",
}


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "float":
        return pd.api.types.is_float_dtype(series)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(series)
    if kind == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    return True  # object / string: anything goes


def validate(df: pd.DataFrame, contract: dict[str, str], name: str) -> pd.DataFrame:
    missing = [c for c in contract if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")
    bad = [c for c, k in contract.items() if not _kind_ok(df[c], k)]
    if bad:
        raise ContractError(f"{name}: wrong dtypes for {bad}: {df[bad].dtypes.to_dict()}")
    if name == "transactions":
        if (df["amount_usd"] < 0).any():
            raise ContractError("transactions: negative amounts")
        if df["day"].min() < 0:
            raise ContractError("transactions: negative day index")
    if name == "labels" and (df["decision_day"] < df["event_day"]).any():
        raise ContractError("labels: a decision cannot be recorded before the event it describes")
    return df
