"""Decision policy: turns a model score into an action. Versioned separately from the model.

Order of precedence (first match wins):
  1. hard override  -> REVIEW   (rules that are never suppressed, accounts with prior confirmed alerts)
  2. score >= tau   -> REVIEW
  3. BTL sample     -> BTL_REVIEW (a random, stratified share of below-threshold alerts goes to
                       analysts blind; this is how missed-suspicious volume is measured)
  4. otherwise      -> SUPPRESS (mode=suppress) or REVIEW at low priority (mode=prioritise)

Sampling is a deterministic hash of alert_id + policy seed, so any decision can be replayed for an audit.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from aml_triage.config import PolicyCfg


def _uniform(alert_ids: np.ndarray, seed: int) -> np.ndarray:
    x = (alert_ids.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(2**32)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x45D9F3B)) % np.uint64(2**32)
    x ^= x >> np.uint64(16)
    return x.astype(np.float64) / 2**32


def policy_version(cfg: PolicyCfg, tau_raw: float) -> str:
    blob = json.dumps({"cfg": cfg.model_dump(), "tau_raw": round(tau_raw, 10)}, sort_keys=True)
    return "pol-" + hashlib.sha256(blob.encode()).hexdigest()[:10]


def route(alerts: pd.DataFrame, raw: np.ndarray, tau_raw: float, cfg: PolicyCfg, seed: int) -> pd.DataFrame:
    """alerts needs: alert_id, rules ('a|b'), gw_prior_confirmed (optional)."""
    n = len(alerts)
    decision = np.full(n, "SUPPRESS" if cfg.mode == "suppress" else "REVIEW_LOW", dtype=object)
    reason = np.full(n, "below_threshold", dtype=object)

    rules = alerts["rules"].fillna("").str.split("|")
    never = set(cfg.never_suppress_rules)
    override = rules.apply(lambda rs: bool(never.intersection(rs))).to_numpy()
    if cfg.override_prior_positive and "gw_prior_confirmed" in alerts:
        override |= alerts["gw_prior_confirmed"].fillna(0).to_numpy() > 0

    above = raw >= tau_raw
    frac = np.where(tau_raw > 0, raw / tau_raw, 0.0)
    u = _uniform(alerts["alert_id"].to_numpy(), seed)
    btl = np.zeros(n, bool)
    for st in cfg.btl.strata:
        in_st = (~above) & (frac >= st.lo_frac_of_tau) & (frac < st.hi_frac_of_tau)
        btl |= in_st & (u < st.rate)

    decision[btl], reason[btl] = "BTL_REVIEW", "below_the_line_sample"
    decision[above], reason[above] = "REVIEW", "model_score"
    decision[override], reason[override] = "REVIEW", "hard_override"
    return pd.DataFrame({"alert_id": alerts["alert_id"].to_numpy(), "decision": decision, "decision_reason": reason})


def btl_recall_estimate(routed: pd.DataFrame, raw: np.ndarray, tau_raw: float, labels: np.ndarray,
                        cfg: PolicyCfg) -> dict[str, float]:
    """Estimate recall the way production must: only labels of reviewed alerts (incl. BTL) are visible.
    missed ~= sum over strata (positives found in BTL sample / sampled) * suppressed in stratum."""
    labels = np.asarray(labels).astype(bool)
    reviewed = routed["decision"].isin(["REVIEW"]).to_numpy()
    found_above = int((reviewed & labels).sum())
    frac = np.where(tau_raw > 0, raw / tau_raw, 0.0)
    below = ~reviewed
    missed_est = 0.0
    for st in cfg.btl.strata:
        in_st = below & (frac >= st.lo_frac_of_tau) & (frac < st.hi_frac_of_tau)
        sampled = in_st & (routed["decision"] == "BTL_REVIEW").to_numpy()
        if sampled.sum():
            missed_est += labels[sampled].mean() * in_st.sum()
    # bootstrap the sampled alerts within each stratum: with a handful of positives found, the estimate is wide
    rng = np.random.default_rng(0)
    sims = []
    strata = []
    for st in cfg.btl.strata:
        in_st = below & (frac >= st.lo_frac_of_tau) & (frac < st.hi_frac_of_tau)
        sampled = in_st & (routed["decision"] == "BTL_REVIEW").to_numpy()
        strata.append((labels[sampled], int(in_st.sum())))
    for _ in range(2000):
        miss = sum((rng.choice(y, size=len(y)).mean() * n) if len(y) else 0.0 for y, n in strata)
        sims.append(found_above / max(found_above + miss, 1e-9))
    ci = (float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5)))
    true_missed = int((below & labels).sum())
    return {
        "recall_estimated_from_btl": found_above / max(found_above + missed_est, 1e-9),
        "recall_estimated_from_btl_ci": ci,
        "btl_positives_found": int(sum(int(y.sum()) for y, _ in strata)),
        "missed_estimated": float(missed_est),
        "recall_true_hindsight": found_above / max(int(labels.sum()), 1),
        "missed_true_hindsight": true_missed,
        "btl_sampled": int((routed["decision"] == "BTL_REVIEW").sum()),
    }
