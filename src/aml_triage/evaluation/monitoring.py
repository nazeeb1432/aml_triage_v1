"""Drift and workload monitoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index. <0.1 stable, 0.1-0.2 watch, >0.2 investigate."""
    e = pd.Series(expected).replace([np.inf, -np.inf], np.nan).fillna(-1e12).to_numpy()
    a = pd.Series(actual).replace([np.inf, -np.inf], np.nan).fillna(-1e12).to_numpy()
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_psi(train: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[dict]:
    rows = [{"feature": f, "psi": round(psi(train[f].to_numpy(), current[f].to_numpy()), 4)} for f in features]
    return sorted(rows, key=lambda r: -r["psi"])
