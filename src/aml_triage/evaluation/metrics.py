"""Metrics that map to how an AML team actually works.

recall_at_fpr        the paper's metric: share of suspicious alerts kept at a given FPR
threshold_at_fpr     the score threshold that produces that FPR (fit on validation, then frozen)
operating_point      what a frozen threshold does on new data: recall, FPR, workload, FP reduction
day_block_bootstrap  confidence intervals that respect the fact that alerts on one day are correlated
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def recall_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose FPR on (y, s) is <= target_fpr."""
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(target_fpr * len(neg)))
    if k >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[k], np.inf))


def operating_point(y: np.ndarray, s: np.ndarray, tau: float) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    review = np.asarray(s) >= tau
    tp, fp = (review & y).sum(), (review & ~y).sum()
    pos, neg = y.sum(), (~y).sum()
    return {
        "recall": tp / pos if pos else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
        "fp_reduction": 1 - fp / neg if neg else float("nan"),
        "review_share": review.mean(),
        "precision_of_queue": tp / max(review.sum(), 1),
        "missed_positives": int(pos - tp),
        "n_alerts": int(len(y)),
        "n_positives": int(pos),
    }


def f_beta(y: np.ndarray, s: np.ndarray, tau: float, beta: float = 2.0) -> float:
    """F-beta with beta>1 weights recall (Bakry et al. report this)."""
    op = operating_point(y, s, tau)
    p, r = op["precision_of_queue"], op["recall"]
    return float((1 + beta**2) * p * r / max(beta**2 * p + r, 1e-12))


def summary(y: np.ndarray, s: np.ndarray, tau: float, target_fpr: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    out = operating_point(y, s, tau)
    out.update({
        "recall_at_target_fpr_oracle": recall_at_fpr(y, s, target_fpr),
        "pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
        "f2": f_beta(y, s, tau, 2.0),
    })
    return out


def day_block_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], reps: int,
                        seed: int, day_col: str = "day") -> tuple[float, float]:
    """95% CI by resampling whole days with replacement."""
    rng = np.random.default_rng(seed)
    groups = {d: g for d, g in df.groupby(day_col)}
    days = np.array(list(groups))
    vals = []
    for _ in range(reps):
        pick = rng.choice(days, size=len(days), replace=True)
        sample = pd.concat([groups[d] for d in pick], ignore_index=True)
        v = stat(sample)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def recall_vs_review_share(y: np.ndarray, s: np.ndarray, points: int = 51) -> list[dict[str, float]]:
    """The capacity curve: if analysts can review X% of alerts (highest scores first), what recall do we get?"""
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(s), kind="stable")
    cum = np.cumsum(y[order])
    n, pos = len(y), max(y.sum(), 1)
    out = []
    for share in np.linspace(0, 1, points):
        k = int(round(share * n))
        out.append({"review_share": float(share), "recall": float(cum[k - 1] / pos) if k else 0.0})
    return out


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=bins, labels=False)
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True).to_dict(orient="records")
