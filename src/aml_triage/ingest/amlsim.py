"""Read IBM AMLSim output (Kaggle: anshankul/ibm-amlsim-example-dataset) into the canonical transaction table.

AMLSim (Weber et al., 2018; github.com/IBM/AMLSim) is IBM's earlier agent-based simulator. The example dataset has:
  transactions.csv  TX_ID, SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_TYPE, TX_AMOUNT, TIMESTAMP (a day index 0..199),
                    IS_FRAUD, ALERT_ID (-1, or the laundering scheme the transaction belongs to)
  alerts.csv        one row per laundering transaction, with ALERT_TYPE (the typology: cycle, fan_in, ...)
  accounts.csv      ACCOUNT_ID, CUSTOMER_ID, INIT_BALANCE, COUNTRY, ACCOUNT_TYPE, IS_FRAUD, TX_BEHAVIOR_ID

Despite its name, alerts.csv is not rule-engine output: it is the ground-truth list of laundering transactions.
The rule engine in this repo creates alerts, exactly as for AMLworld.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aml_triage.contracts import TRANSACTIONS, validate

TXN_DTYPES = {"TX_ID": "int64", "SENDER_ACCOUNT_ID": "int64", "RECEIVER_ACCOUNT_ID": "int64", "TX_TYPE": "string",
              "TX_AMOUNT": "float64", "TIMESTAMP": "int64", "ALERT_ID": "int64"}


def read_amlsim(directory: str | Path, start_date: str = "2024-01-01") -> pd.DataFrame:
    d = Path(directory)
    t = pd.read_csv(d / "transactions.csv", dtype=TXN_DTYPES)
    t.columns = [c.strip() for c in t.columns]
    fraud = t["IS_FRAUD"]
    if fraud.dtype != bool:
        fraud = fraud.astype(str).str.strip().str.lower().isin(["true", "1"])
    typ = pd.Series([None] * len(t), dtype=object)
    alerts_path = d / "alerts.csv"
    if alerts_path.exists():
        a = pd.read_csv(alerts_path)
        a.columns = [c.strip() for c in a.columns]
        kinds = a.drop_duplicates("ALERT_ID").set_index("ALERT_ID")["ALERT_TYPE"].astype(str).str.upper().str.replace("_", "-")
        typ = t["ALERT_ID"].map(kinds).where(fraud, None)
    day = t["TIMESTAMP"].astype("int32")
    start = pd.Timestamp(start_date)
    # AMLSim only records the day; spread transactions through it in file order so timestamps are unique and stable
    within = t.groupby(day).cumcount().to_numpy()
    per_day = np.maximum(t.groupby(day)["TX_ID"].transform("size").to_numpy(), 1)
    seconds = (within * (86_399 / per_day)).astype("int64")
    out = pd.DataFrame({
        "txn_id": t["TX_ID"].astype("int64").to_numpy(),
        "ts": start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(seconds, unit="s"),
        "day": day.to_numpy(),
        "src": "A" + t["SENDER_ACCOUNT_ID"].astype(str),
        "dst": "A" + t["RECEIVER_ACCOUNT_ID"].astype(str),
        "amount_usd": t["TX_AMOUNT"].astype("float64").to_numpy(),
        "pay_format": t["TX_TYPE"].astype(str).str.title().to_numpy(),
        "currency": "US Dollar",
        "cross_currency": False,
        "is_laundering": fraud.astype(bool).to_numpy(),
        "typology": typ.to_numpy(),
    })
    out = out[out["src"] != out["dst"]].sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True)
    out.attrs["start_date"] = str(start.date())
    return validate(out, TRANSACTIONS, "transactions")
