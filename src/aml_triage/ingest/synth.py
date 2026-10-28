"""Synthetic ledger written in the IBM AMLworld file format.

Why it exists: the real IBM file lives on Kaggle and can't be committed, and tests need a
small, deterministic world. Writing the *same* CSV/patterns format means the real reader
(`ibm.py`) is exercised on every run, and switching to the real data is a config change.

Design goals (each one exists to make the triage problem realistic, not easy):
  * legitimate behaviour that trips rules: payroll fan-outs, merchant fan-ins, large
    purchases, invoices just under 10k, broker pass-through  -> lots of false positives
  * laundering uses the 8 IBM typologies, routed through accounts that also have normal
    history (mules), often reused across attempts -> graph features have something to find
  * some laundering is low-and-slow and never alerts -> rules have imperfect recall too
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aml_triage.ingest.ibm import FX_TO_USD

TYPOLOGIES = ["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "STACK", "BIPARTITE", "RANDOM"]
FORMATS = np.array(["Credit Card", "ACH", "Cheque", "Cash", "Wire", "Bitcoin", "Reinvestment"])
RETAIL, BUSINESS, HNW, MERCHANT, PAYROLL = range(5)


class _World:
    def __init__(self, n_accounts: int, n_banks: int, rng: np.random.Generator):
        self.rng = rng
        self.n = n_accounts
        p = np.array([0.70, 0.19, 0.07, 0.025, 0.015])
        self.persona = rng.choice(5, size=n_accounts, p=p / p.sum())
        bank_w = 1.0 / np.arange(1, n_banks + 1) ** 0.9
        self.bank = rng.choice(np.arange(1, n_banks + 1), size=n_accounts, p=bank_w / bank_w.sum())
        hexes = set()
        while len(hexes) < n_accounts:
            hexes.update(f"{x:09X}" for x in rng.integers(0x100000000, 0xFFFFFFFFF, size=n_accounts))
        self.acct = np.array(sorted(hexes)[:n_accounts])
        rng.shuffle(self.acct)
        currencies = [c for c in FX_TO_USD if c != "Bitcoin"]
        home = np.where(rng.random(n_banks + 1) < 0.6, "US Dollar", rng.choice(currencies, size=n_banks + 1))
        self.currency = home[self.bank]

        self.idx = {k: np.flatnonzero(self.persona == k) for k in range(5)}
        merch = self.idx[MERCHANT]
        self.merchant_w = 1.0 / np.arange(1, len(merch) + 1) ** 1.1
        self.merchant_w /= self.merchant_w.sum()
        people = np.concatenate([self.idx[RETAIL], self.idx[HNW]])
        biz = self.idx[BUSINESS]
        self.contacts = rng.choice(np.concatenate([people, biz]), size=(n_accounts, 5))
        self.suppliers = rng.choice(np.concatenate([biz, merch]), size=(n_accounts, 6))
        self.landlord = rng.choice(biz[: max(10, len(biz) // 10)], size=n_accounts)
        self.pays_rent = rng.random(n_accounts) < 0.55
        self.rent = rng.uniform(700, 2600, size=n_accounts)
        self.biz_activity = rng.lognormal(0, 0.8, size=n_accounts)
        self.brokers = rng.choice(biz, size=max(5, len(biz) // 25), replace=False)
        self.employees = {int(p_): rng.choice(self.idx[RETAIL], size=rng.integers(15, 90), replace=False)
                          for p_ in self.idx[PAYROLL]}
        self.payday_offset = {int(p_): int(rng.integers(0, 14)) for p_ in self.idx[PAYROLL]}
        non_hub = np.concatenate([self.idx[RETAIL], self.idx[BUSINESS]])
        self.mule_pool = rng.choice(non_hub, size=max(50, n_accounts // 14), replace=False)
        self.non_hub = non_hub


def _legit_day(w: _World, day: int) -> list[tuple]:
    rng, out = w.rng, []

    def emit(src, dst, amt, fmt_p):
        k = len(src)
        if k == 0:
            return
        fmts = rng.choice(len(FORMATS), size=k, p=fmt_p)
        out.append((np.full(k, day), src, dst, amt, fmts))

    weekday = day % 7 < 5
    # retail spending
    r = w.idx[RETAIL]
    k = rng.poisson(0.33 * len(r) * (1.0 if weekday else 0.8))
    src = rng.choice(r, size=k)
    to_merchant = rng.random(k) < 0.6
    dst = np.where(to_merchant, rng.choice(w.idx[MERCHANT], size=k, p=w.merchant_w), w.contacts[src, rng.integers(0, 5, k)])
    amt = rng.lognormal(np.log(70), 1.0, size=k)
    emit(src, dst, amt, [0.45, 0.25, 0.05, 0.17, 0.05, 0.01, 0.02])
    # rent
    due = (w.pays_rent) & ((np.arange(w.n) % 28) == day % 28) & (w.persona == RETAIL)
    src = np.flatnonzero(due)
    emit(src, w.landlord[src], w.rent[src] * rng.uniform(0.98, 1.02, len(src)), [0, 0.8, 0.15, 0.05, 0, 0, 0])
    # business payments (incl. invoices just under 10k: a classic structuring false positive)
    b = w.idx[BUSINESS]
    k = rng.poisson(1.1 * len(b) * (1.0 if weekday else 0.3))
    src = rng.choice(b, size=k, p=w.biz_activity[b] / w.biz_activity[b].sum())
    dst = w.suppliers[src, rng.integers(0, 6, k)]
    amt = rng.lognormal(np.log(1400), 1.1, size=k)
    inv = rng.random(k) < 0.03
    amt[inv] = rng.uniform(8000, 9990, inv.sum())
    emit(src, dst, amt, [0.05, 0.5, 0.15, 0.1, 0.18, 0.01, 0.01])
    # high-net-worth: occasional very large purchases
    h = w.idx[HNW]
    k = rng.poisson(0.4 * len(h))
    src = rng.choice(h, size=k)
    dst = np.where(rng.random(k) < 0.5, w.suppliers[src, 0], w.contacts[src, rng.integers(0, 5, k)])
    amt = rng.lognormal(np.log(2500), 1.3, size=k)
    big = rng.random(k) < 0.04
    amt[big] = rng.uniform(15000, 250000, big.sum())
    emit(src, dst, amt, [0.25, 0.3, 0.1, 0.05, 0.25, 0.04, 0.01])
    # merchants pay suppliers
    m = w.idx[MERCHANT]
    k = rng.poisson(3.0 * len(m))
    src = rng.choice(m, size=k, p=w.merchant_w)
    emit(src, w.suppliers[src, rng.integers(0, 6, k)], rng.lognormal(np.log(2200), 1.0, k), [0, 0.6, 0.1, 0.05, 0.25, 0, 0])
    # payroll fan-outs every 14 days (fan-out false positives)
    for p_, emps in w.employees.items():
        if (day + w.payday_offset[p_]) % 14 == 0:
            emit(np.full(len(emps), p_), emps, rng.uniform(1500, 4200, len(emps)), [0, 0.95, 0.05, 0, 0, 0, 0])
    # broker pass-through (pass-through false positives)
    k = rng.poisson(max(3, len(w.brokers) // 2))
    brk = rng.choice(w.brokers, size=k)
    a_in = rng.lognormal(np.log(9000), 0.9, k)
    emit(rng.choice(w.idx[BUSINESS], size=k), brk, a_in, [0, 0.5, 0.1, 0, 0.4, 0, 0])
    emit(brk, rng.choice(w.idx[BUSINESS], size=k), a_in * rng.uniform(0.85, 0.99, k), [0, 0.5, 0.1, 0, 0.4, 0, 0])
    return out


def _pick(w: _World, k: int) -> np.ndarray:
    from_pool = w.rng.random(k) < 0.7
    chosen = np.where(from_pool, w.rng.choice(w.mule_pool, size=k), w.rng.choice(w.non_hub, size=k))
    return np.unique(chosen) if len(np.unique(chosen)) == k else _pick(w, k)


def _laundering_attempt(w: _World, typ: str, start: int) -> list[tuple[int, int, int, float]]:
    """Return edges (day, src, dst, amount) for one laundering attempt."""
    rng = w.rng
    total = rng.lognormal(np.log(25000), 0.9)
    stealth = rng.random() < 0.35            # low-and-slow: many smaller hops, spread out
    edges: list[tuple[int, int, int, float]] = []

    def day_at(hop: int) -> int:
        # fast schemes move money within a day or two per hop; stealthy ones wait days between hops
        per_hop = rng.integers(1, 4) if stealth else rng.integers(0, 2)
        return int(start + hop * per_hop + (rng.integers(0, 3) if stealth else 0))

    def split(amount: float, k: int) -> np.ndarray:
        return amount * rng.dirichlet(np.full(k, 2.0)) * (rng.uniform(0.6, 0.85) if stealth else 1)

    if typ in ("FAN-OUT", "FAN-IN"):
        k = int(rng.integers(4, 14))
        acc = _pick(w, k + 1)
        hub, leaves = acc[0], acc[1:]
        for i, (leaf, a) in enumerate(zip(leaves, split(total, k), strict=True)):
            d = day_at(i % 3)
            edges.append((d, hub, leaf, a) if typ == "FAN-OUT" else (d, leaf, hub, a))
    elif typ == "CYCLE":
        k = int(rng.integers(3, 9))
        acc = _pick(w, k)
        a = total / 3
        for i in range(k):
            a *= rng.uniform(0.93, 0.99)
            edges.append((day_at(i), acc[i], acc[(i + 1) % k], a))
    elif typ in ("SCATTER-GATHER", "GATHER-SCATTER"):
        k = int(rng.integers(3, 10))
        acc = _pick(w, k + 2)
        a_, b_, mid = acc[0], acc[1], acc[2:]
        parts = split(total, k)
        for _i, (m, a) in enumerate(zip(mid, parts, strict=True)):
            if typ == "SCATTER-GATHER":
                edges.append((day_at(0), a_, m, a))
                edges.append((day_at(2), m, b_, a * rng.uniform(0.9, 0.99)))
            else:
                edges.append((day_at(0), m, a_, a))
        if typ == "GATHER-SCATTER":
            outs = _pick(w, k)
            for o, a in zip(outs, split(total * 0.95, k), strict=True):
                edges.append((day_at(2), a_, o, a))
    elif typ in ("STACK", "BIPARTITE"):
        k1, k2 = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        acc = _pick(w, k1 + k2 + (k2 if typ == "STACK" else 0))
        l1, l2 = acc[:k1], acc[k1 : k1 + k2]
        for s in l1:
            for d_, a in zip(l2, split(total / k1, k2), strict=True):
                if rng.random() < 0.8:
                    edges.append((day_at(0), s, d_, a))
        if typ == "STACK":
            l3 = acc[k1 + k2 :]
            for s, d_ in zip(l2, l3, strict=True):
                edges.append((day_at(2), s, d_, total / k2 * rng.uniform(0.8, 0.95)))
    else:  # RANDOM chain
        k = int(rng.integers(3, 9))
        acc = _pick(w, k + 1)
        a = total / 2
        for i in range(k):
            a *= rng.uniform(0.9, 0.99)
            edges.append((day_at(i), acc[i], acc[i + 1], a))
    return edges


def generate(cfg_synth, seed: int) -> tuple[pd.DataFrame, list[tuple[str, list[int]]]]:
    rng = np.random.default_rng(seed)
    w = _World(cfg_synth.n_accounts, cfg_synth.n_banks, rng)
    chunks = []
    for day in range(cfg_synth.n_days):
        chunks.extend(_legit_day(w, day))
    day = np.concatenate([c[0] for c in chunks])
    src = np.concatenate([c[1] for c in chunks])
    dst = np.concatenate([c[2] for c in chunks])
    amt = np.concatenate([c[3] for c in chunks])
    fmt = FORMATS[np.concatenate([c[4] for c in chunks])]
    legit = pd.DataFrame({"day": day, "src": src, "dst": dst, "amount_usd": amt, "fmt": fmt, "is_laundering": 0})
    legit = legit[legit["src"] != legit["dst"]]

    rows, attempts = [], []
    launder_fmt_p = np.array([0.08, 0.35, 0.10, 0.17, 0.20, 0.10, 0.0])
    for _ in range(cfg_synth.laundering_patterns):
        typ = str(rng.choice(TYPOLOGIES))
        start = int(rng.integers(3, cfg_synth.n_days - 4))  # schemes may run past the end, like real ones
        edges = _laundering_attempt(w, typ, start)
        fmts = FORMATS[rng.choice(len(FORMATS), size=len(edges), p=launder_fmt_p)]
        first = len(rows)
        for (d, s, t, a), f in zip(edges, fmts, strict=True):
            if s != t and d < cfg_synth.n_days:
                rows.append((d, s, t, a, f, 1))
        attempts.append((typ, list(range(first, len(rows)))))
    bad = pd.DataFrame(rows, columns=["day", "src", "dst", "amount_usd", "fmt", "is_laundering"])
    bad["_row"] = np.arange(len(bad))
    df = pd.concat([legit.assign(_row=-1), bad], ignore_index=True)
    minutes = rng.integers(0, 24 * 60, size=len(df))
    start_ts = pd.Timestamp(cfg_synth.start_date)
    df["ts"] = start_ts + pd.to_timedelta(df["day"], unit="D") + pd.to_timedelta(minutes, unit="m")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return _to_ibm(df, w), [(t, df.index[df["_row"].isin(r)].tolist()) for t, r in attempts]


def _to_ibm(df: pd.DataFrame, w: _World) -> pd.DataFrame:
    src, dst = df["src"].to_numpy(), df["dst"].to_numpy()
    pay_cur = np.where(df["fmt"] == "Bitcoin", "Bitcoin", w.currency[src])
    rec_cur = np.where(df["fmt"] == "Bitcoin", "Bitcoin", w.currency[dst])
    fx_pay = pd.Series(pay_cur).map(FX_TO_USD).to_numpy()
    fx_rec = pd.Series(rec_cur).map(FX_TO_USD).to_numpy()
    usd = df["amount_usd"].to_numpy()
    return pd.DataFrame({
        "Timestamp": df["ts"].dt.strftime("%Y/%m/%d %H:%M"),
        "From Bank": w.bank[src],
        "Account": w.acct[src],
        "To Bank": w.bank[dst],
        "Account.1": w.acct[dst],
        "Amount Received": np.round(usd / fx_rec, 2),
        "Receiving Currency": rec_cur,
        "Amount Paid": np.round(usd / fx_pay, 2),
        "Payment Currency": pay_cur,
        "Payment Format": df["fmt"].to_numpy(),
        "Is Laundering": df["is_laundering"].astype(int).to_numpy(),
    })


def write_synthetic(cfg, out_dir: str | Path) -> tuple[Path, Path]:
    s = cfg.data.synthetic
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ibm, attempts = generate(s, cfg.seed)
    trans_path, pat_path = out_dir / s.out_trans_file, out_dir / s.out_patterns_file
    ibm.to_csv(trans_path, index=False)
    with open(pat_path, "w") as f:
        for typ, idx in attempts:
            if not idx:
                continue
            f.write(f"BEGIN LAUNDERING ATTEMPT - {typ}:  synthetic\n")
            ibm.iloc[idx].to_csv(f, header=False, index=False)
            f.write(f"END LAUNDERING ATTEMPT - {typ}\n\n")
    return trans_path, pat_path
