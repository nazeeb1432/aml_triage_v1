"""Graph features (Eddin §2.3): degree features and GuiltyWalker, computed point-in-time.

For every day d that has alerts we build a snapshot of the transaction graph:
  * edges from the last `twl_days` days are always kept
  * edges from the last `tws_days` days are kept only if they touch an account that is
    known-suspicious *as of d* (decision_day <= d). The replication chose the window from
    the row's own label, which is not available in production.
The snapshot is stored as CSR arrays and the per-target loops run in numba, so a day with
thousands of alerts takes milliseconds instead of minutes.

GuiltyWalker walks to the as-of illicit set. GuiltyWalker-delay (GWd) walks to a hybrid
set: as-of confirmed accounts plus recent, still-undecided alerts that a stage-1 model
(trained only on the warm-up period) scores as likely suspicious.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
import pandas as pd

from aml_triage.config import FeaturesCfg
from aml_triage.labels.store import LabelIndex

DEGREE_COLS = [
    "g_in_deg", "g_out_deg", "g_in_w", "g_out_w",
    "g_nb_in_deg_mean", "g_nb_in_deg_min", "g_nb_in_deg_max",
    "g_nb_out_deg_mean", "g_nb_out_deg_min", "g_nb_out_deg_max",
    "g_nb_in_w_mean", "g_nb_out_w_mean", "g_n_neighbours",
]
GW_STATS = ["min", "max", "mean", "median", "std", "p25", "p75", "hit_rate", "n_distinct"]


@dataclass
class Snapshot:
    indptr: np.ndarray       # undirected CSR (walk graph, hubs removed)
    indices: np.ndarray
    full_indptr: np.ndarray  # undirected CSR including hubs (degree features)
    full_indices: np.ndarray
    in_deg: np.ndarray
    out_deg: np.ndarray
    in_w: np.ndarray
    out_w: np.ndarray


def _csr(a: np.ndarray, b: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order].astype(np.int64)


class GraphBuilder:
    def __init__(self, txns: pd.DataFrame, node_ids: dict[str, int], cfg: FeaturesCfg):
        order = np.argsort(txns["day"].to_numpy(), kind="stable")
        self.day = txns["day"].to_numpy()[order]
        self.s = txns["src"].map(node_ids).to_numpy()[order].astype(np.int64)
        self.t = txns["dst"].map(node_ids).to_numpy()[order].astype(np.int64)
        self.amt = txns["amount_usd"].to_numpy()[order]
        self.n = len(node_ids)
        self.cfg = cfg.graph

    def snapshot(self, d: int, suspicious: np.ndarray) -> Snapshot:
        lo = np.searchsorted(self.day, d - self.cfg.tws_days + 1, side="left")
        hi = np.searchsorted(self.day, d, side="right")
        day, s, t, amt = self.day[lo:hi], self.s[lo:hi], self.t[lo:hi], self.amt[lo:hi]
        keep = (day >= d - self.cfg.twl_days + 1) | suspicious[s] | suspicious[t]
        s, t, amt = s[keep], t[keep], amt[keep]
        key = s * self.n + t
        uniq, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=amt)
        us, ut = uniq // self.n, uniq % self.n
        out_deg = np.bincount(us, minlength=self.n)
        in_deg = np.bincount(ut, minlength=self.n)
        out_w = np.bincount(us, weights=w, minlength=self.n)
        in_w = np.bincount(ut, weights=w, minlength=self.n)
        full_indptr, full_indices = _csr(us, ut, self.n)
        deg = np.diff(full_indptr)
        active = deg[deg > 0]
        cap = np.quantile(active, self.cfg.hub_degree_quantile) if len(active) else np.inf
        hub = deg > cap
        ok = ~(hub[us] | hub[ut])
        indptr, indices = _csr(us[ok], ut[ok], self.n)
        return Snapshot(indptr, indices, full_indptr, full_indices, in_deg, out_deg, in_w, out_w)


@nb.njit(cache=True)
def _degree_kernel(targets, indptr, indices, in_deg, out_deg, in_w, out_w):
    out = np.full((len(targets), 9), np.nan)
    for i in range(len(targets)):
        v = targets[i]
        a, b = indptr[v], indptr[v + 1]
        if b == a:
            out[i, 8] = 0.0
            continue
        nb_ = indices[a:b]
        ind = in_deg[nb_].astype(np.float64)
        outd = out_deg[nb_].astype(np.float64)
        out[i, 0], out[i, 1], out[i, 2] = ind.mean(), ind.min(), ind.max()
        out[i, 3], out[i, 4], out[i, 5] = outd.mean(), outd.min(), outd.max()
        out[i, 6], out[i, 7] = in_w[nb_].mean(), out_w[nb_].mean()
        out[i, 8] = b - a
    return out


def degree_features(snap: Snapshot, targets: np.ndarray) -> np.ndarray:
    nbr = _degree_kernel(targets, snap.full_indptr, snap.full_indices, snap.in_deg, snap.out_deg,
                         snap.in_w, snap.out_w)
    own = np.column_stack([snap.in_deg[targets], snap.out_deg[targets], snap.in_w[targets], snap.out_w[targets]])
    return np.column_stack([own, nbr])


@nb.njit(cache=True)
def _walk_kernel(targets, indptr, indices, illicit, n_walks, max_steps, seed):
    out = np.full((len(targets), 9), np.nan)
    for i in range(len(targets)):
        v = targets[i]
        np.random.seed(seed + v * 7919)
        lengths = np.empty(n_walks)
        reached = np.empty(n_walks, dtype=np.int64)
        k = 0
        for _ in range(n_walks):
            cur = v
            for step in range(1, max_steps + 1):
                deg = indptr[cur + 1] - indptr[cur]
                if deg == 0:
                    break
                cur = indices[indptr[cur] + np.random.randint(deg)]
                if cur != v and illicit[cur]:
                    lengths[k] = step
                    reached[k] = cur
                    k += 1
                    break
        out[i, 7] = k / n_walks
        if k == 0:
            out[i, 8] = 0.0
            continue
        L = np.sort(lengths[:k])
        out[i, 0], out[i, 1], out[i, 2] = L[0], L[-1], L.mean()
        out[i, 3] = np.median(L)
        out[i, 4] = L.std()
        out[i, 5] = L[int(0.25 * (k - 1))]
        out[i, 6] = L[int(0.75 * (k - 1))]
        out[i, 8] = len(np.unique(reached[:k]))
    return out


def guilty_walker(snap: Snapshot, targets: np.ndarray, illicit: np.ndarray, n_walks: int, max_steps: int,
                  seed: int) -> np.ndarray:
    return _walk_kernel(targets, snap.indptr, snap.indices, illicit, n_walks, max_steps, seed)


def graph_features(txns: pd.DataFrame, alerts: pd.DataFrame, labels: pd.DataFrame, node_ids: dict[str, int],
                   cfg: FeaturesCfg, seed: int, hybrid_extra: pd.DataFrame | None = None,
                   base: bool = True) -> pd.DataFrame:
    """Degree + GW (+ GWd when `hybrid_extra` gives pseudo-positive alerts) for every alert.

    hybrid_extra columns: account, day (alert day), decision_day, pseudo_pos (bool)
    An entry joins the GWd illicit set on days where it is recent (day > d - window),
    already happened (day <= d) and not yet decided (decision_day > d).
    """
    gb = GraphBuilder(txns, node_ids, cfg)
    lidx = LabelIndex(labels, node_ids)
    gw = cfg.guilty_walker
    rows = []
    if hybrid_extra is not None:
        hx = hybrid_extra[hybrid_extra["pseudo_pos"]].copy()
        hx["node"] = hx["account"].map(node_ids)
    for d, grp in alerts.groupby("day", sort=True):
        d = int(d)
        targets = grp["account"].map(node_ids).to_numpy().astype(np.int64)
        illicit = lidx.illicit_mask_as_of(d, gw.illicit_memory_days)
        prior = lidx.prior_positive_count_as_of(d)[targets]
        snap = gb.snapshot(d, illicit)
        feats = {"alert_id": grp["alert_id"].to_numpy()}
        if base:
            deg = degree_features(snap, targets)
            for j, c in enumerate(DEGREE_COLS):
                feats[c] = deg[:, j]
            walks = guilty_walker(snap, targets, illicit, gw.n_walks, gw.max_steps, seed + d * 100_003)
            for j, s in enumerate(GW_STATS):
                feats[f"gw_{s}"] = walks[:, j]
            feats["gw_prior_confirmed"] = prior
        if hybrid_extra is not None:
            win = cfg.gwd.pseudo_label_window_days
            # pending = no decision recorded yet (null in production, or recorded after d in a backtest)
            pending = hx["decision_day"].isna() | (hx["decision_day"] > d)
            recent = hx[(hx["day"] <= d) & (hx["day"] > d - win) & pending]
            hybrid = illicit.copy()
            hybrid[recent["node"].to_numpy()] = True
            wd = guilty_walker(snap, targets, hybrid, gw.n_walks, gw.max_steps, seed + d * 100_003)
            for j, s in enumerate(GW_STATS):
                feats[f"gwd_{s}"] = wd[:, j]
            feats["gwd_prior_confirmed"] = prior
        rows.append(pd.DataFrame(feats))
    return pd.concat(rows, ignore_index=True)
