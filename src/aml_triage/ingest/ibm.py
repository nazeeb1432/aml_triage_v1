"""Read IBM AMLworld files (Altman et al., NeurIPS 2023) into the canonical transaction table.

Files (Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml):
  *_Trans.csv     Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
                  Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering
  *_Patterns.txt  blocks of laundering transactions, one block per laundering attempt:
                  BEGIN LAUNDERING ATTEMPT - FAN-OUT: ...
                  <same CSV columns>
                  END LAUNDERING ATTEMPT - FAN-OUT

The synthetic generator in `synth.py` writes exactly this format, so this reader is the
single entry point for both real and synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from aml_triage.contracts import TRANSACTIONS, validate

# Approximate fixed USD rates. IBM simulates with fixed rates too; exact values only shift
# amount scales per currency, which the model sees consistently in train and test.
FX_TO_USD = {
    "US Dollar": 1.0,
    "Euro": 1.08,
    "UK Pound": 1.27,
    "Swiss Franc": 1.12,
    "Yuan": 0.14,
    "Yen": 0.0068,
    "Rupee": 0.012,
    "Ruble": 0.011,
    "Australian Dollar": 0.66,
    "Canadian Dollar": 0.74,
    "Brazil Real": 0.20,
    "Mexican Peso": 0.058,
    "Saudi Riyal": 0.267,
    "Shekel": 0.27,
    "Bitcoin": 30000.0,
}

IBM_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def _account_key(bank: pl.Expr, account: pl.Expr) -> pl.Expr:
    # banks appear zero-padded in some files ("0011") and not in others ("11")
    stripped = bank.cast(pl.Utf8).str.strip_chars_start("0")
    bank_norm = pl.when(stripped == "").then(pl.lit("0")).otherwise(stripped)
    return pl.concat_str([bank_norm, account], separator="_")


def read_ibm_transactions(path: str | Path) -> pd.DataFrame:
    lf = pl.scan_csv(
        path,
        has_header=True,
        new_columns=IBM_COLUMNS,
        schema_overrides={"From Bank": pl.Utf8, "To Bank": pl.Utf8, "Account": pl.Utf8, "Account.1": pl.Utf8},
    )
    fx = pl.DataFrame({"Payment Currency": list(FX_TO_USD), "fx": list(FX_TO_USD.values())})
    df = (
        lf.with_columns(pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M").alias("ts"))
        .join(fx.lazy(), on="Payment Currency", how="left")
        .with_columns(
            _account_key(pl.col("From Bank"), pl.col("Account")).alias("src"),
            _account_key(pl.col("To Bank"), pl.col("Account.1")).alias("dst"),
            (pl.col("Amount Paid") * pl.col("fx").fill_null(1.0)).alias("amount_usd"),
            pl.col("Payment Format").alias("pay_format"),
            pl.col("Payment Currency").alias("currency"),
            (pl.col("Payment Currency") != pl.col("Receiving Currency")).alias("cross_currency"),
            (pl.col("Is Laundering") == 1).alias("is_laundering"),
        )
        .select(["ts", "src", "dst", "amount_usd", "pay_format", "currency", "cross_currency", "is_laundering"])
        .sort("ts")
        .collect()
        .to_pandas()
    )
    start = df["ts"].min().normalize()
    df["day"] = ((df["ts"].dt.normalize() - start).dt.days).astype("int32")
    df.insert(0, "txn_id", np.arange(len(df), dtype="int64"))
    df.attrs["start_date"] = str(start.date())
    return validate(df, TRANSACTIONS, "transactions")


def read_ibm_patterns(path: str | Path) -> pd.DataFrame:
    """Return one row per laundering transaction listed in the patterns file, with its typology."""
    rows: list[list[str]] = []
    typology, attempt = None, -1
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            typology = line.split("-", 1)[1].split(":")[0].strip()
            attempt += 1
        elif line.startswith("END LAUNDERING ATTEMPT"):
            typology = None
        elif line and typology:
            parts = line.split(",")
            if len(parts) >= 11:
                rows.append(parts[:11] + [typology, str(attempt)])
    if not rows:
        return pd.DataFrame(columns=["ts", "src", "dst", "amount_paid", "typology", "attempt_id"])
    p = pd.DataFrame(rows, columns=IBM_COLUMNS + ["typology", "attempt_id"])
    p["ts"] = pd.to_datetime(p["Timestamp"], format="%Y/%m/%d %H:%M")
    norm = lambda b: b.str.lstrip("0").replace("", "0")  # noqa: E731
    p["src"] = norm(p["From Bank"]) + "_" + p["Account"]
    p["dst"] = norm(p["To Bank"]) + "_" + p["Account.1"]
    p["amount_paid"] = p["Amount Paid"].astype(float).round(2)
    p["attempt_id"] = p["attempt_id"].astype(int)
    return p[["ts", "src", "dst", "amount_paid", "typology", "attempt_id"]]


def attach_typology(txns: pd.DataFrame, patterns: pd.DataFrame, fx_amount_col: str = "amount_usd") -> pd.DataFrame:
    """Best-effort join of typology onto laundering transactions (used for per-typology recall only)."""
    txns = txns.copy()
    txns["typology"] = None
    if patterns.empty:
        return txns
    key = patterns.drop_duplicates(["ts", "src", "dst"]).set_index(["ts", "src", "dst"])["typology"]
    idx = pd.MultiIndex.from_frame(txns[["ts", "src", "dst"]])
    txns["typology"] = key.reindex(idx).to_numpy()
    txns.loc[~txns["is_laundering"], "typology"] = None
    return txns
