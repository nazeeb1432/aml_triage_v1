"""FastAPI service behind the analyst UI.

Reads the outputs of one pipeline run (reports/<run>/) and keeps analyst decisions in a
SQLite audit log. Ground-truth labels never leave this service through the queue or alert
endpoints; only aggregate evaluation numbers do (/api/model, /api/policy/simulate).

Env vars: AML_TRIAGE_REPORTS (default reports), AML_TRIAGE_RUN (default newest run),
AML_TRIAGE_ARTIFACTS (default artifacts), AML_TRIAGE_DB (default reports/<run>/audit.sqlite).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aml_triage.serving.reasons import RULE_TEXT

PUBLIC_ALERT_COLS = ["alert_id", "account", "day", "rules", "probability", "priority_rank",
                     "a_sum_in", "a_sum_out", "a_max_out", "a_max_in", "a_n_src", "a_n_dst"]


def _clean(o):
    """JSON has no NaN/inf; Starlette refuses them. Turn them into null, recursively."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _band(p: float, tau: float) -> str:
    # probabilities are honest and small (base rate ~1%): "high" = at least 1 in 10 alerts like this was suspicious
    if p >= max(0.10, tau):
        return "high"
    if p >= tau:
        return "medium"
    return "low"


class Disposition(BaseModel):
    action: Literal["escalate", "close", "sar"]
    analyst: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=2000)


class Approval(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    role: Literal["model_risk", "compliance", "engineering"]
    note: str = Field(default="", max_length=2000)


class Simulation(BaseModel):
    review_share: float | None = Field(default=None, ge=0, le=1)
    reviews_per_day: float | None = Field(default=None, ge=0)


class Store:
    def __init__(self):
        reports = Path(os.environ.get("AML_TRIAGE_REPORTS", "reports"))
        run = os.environ.get("AML_TRIAGE_RUN")
        if not run:
            runs = sorted((p for p in reports.glob("*/scored_alerts.parquet")), key=lambda p: p.stat().st_mtime)
            if not runs:
                raise RuntimeError(f"no scored run under {reports}/ — run `aml-triage run` first")
            run = runs[-1].parent.name
        self.dir = reports / run
        self.metrics = _clean(json.loads((self.dir / "metrics.json").read_text()))
        self.alerts = pd.read_parquet(self.dir / "scored_alerts.parquet")
        self.txns = pd.read_parquet(self.dir / "live_transactions.parquet")
        labels = pd.read_parquet(self.dir / "labels.parquet")
        self.confirmed = labels[labels["label"]][["account", "decision_day"]]
        self.start = pd.Timestamp(self.metrics["data"].get("start_date") or "2024-01-01")
        self.tau = float(self.metrics["champion"]["tau_calibrated"])
        cfg_path = self.dir / "config_resolved.yaml"
        self.tws = int(yaml.safe_load(cfg_path.read_text())["features"]["graph"]["tws_days"]) if cfg_path.exists() else 30
        self.manifest_path = Path(os.environ.get("AML_TRIAGE_ARTIFACTS", "artifacts")) / "models" / \
            self.metrics["run_id"] / "manifest.json"
        self.db = os.environ.get("AML_TRIAGE_DB", str(self.dir / "audit.sqlite"))

    def connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db)
        c.execute("""CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, alert_id INTEGER, action TEXT, analyst TEXT, note TEXT,
            model_run_id TEXT, policy_version TEXT, probability REAL, routed_decision TEXT)""")
        return c

    def date(self, day: int) -> str:
        return (self.start + pd.Timedelta(days=int(day))).date().isoformat()

    def known_suspicious(self, accounts: list[str], as_of: int) -> set[str]:
        k = self.confirmed[(self.confirmed["decision_day"] <= as_of) & self.confirmed["account"].isin(accounts)]
        return set(k["account"])

    def dispositions(self) -> pd.DataFrame:
        with closing(self.connect()) as c:
            return pd.read_sql("SELECT * FROM audit ORDER BY id DESC", c)


def _public(row: pd.Series, store: Store, last_action: dict) -> dict:
    out = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in PUBLIC_ALERT_COLS}
    out["rules"] = [r for r in row["rules"].split("|") if r]
    out["date"] = store.date(row["day"])
    out["band"] = _band(row["probability"], store.tau)
    # BTL samples look exactly like any other reviewed alert to the analyst
    out["decision"] = "SUPPRESS" if row["decision"] == "SUPPRESS" else "REVIEW"
    out["status"] = last_action.get(int(row["alert_id"]), "open")
    out["top_reason"] = json.loads(row["reasons_json"])[0]["text"] if row["reasons_json"] else ""
    return out


def create_app() -> FastAPI:
    store = Store()
    app = FastAPI(title="AML Alert Triage", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    A = store.alerts

    def last_actions() -> dict:
        d = store.dispositions()
        return {} if d.empty else d.drop_duplicates("alert_id").set_index("alert_id")["action"].to_dict()

    @app.get("/api/health")
    def health():
        return {"status": "ok", "run": store.dir.name, "model_run_id": store.metrics["run_id"],
                "policy_version": store.metrics.get("policy", {}).get("version")}

    @app.get("/api/days")
    def days():
        g = A.groupby("day").agg(alerts=("alert_id", "size"),
                                 reviewed=("decision", lambda s: int((s != "SUPPRESS").sum())))
        return {"items": [{"day": int(d), "date": store.date(d), **{k: int(v) for k, v in r.items()}} for d, r in g.iterrows()]}

    @app.get("/api/summary")
    def summary(day: int | None = None):
        day = int(A["day"].max()) if day is None else day
        a = A[A["day"] == day]
        acted = last_actions()
        reviewed = a[a["decision"] != "SUPPRESS"]
        pol = store.metrics.get("policy", {})
        return {
            "day": day, "date": store.date(day), "alerts": len(a), "in_queue": len(reviewed),
            "suppressed": int((a["decision"] == "SUPPRESS").sum()),
            "hard_overrides": int((a["decision_reason"] == "hard_override").sum()),
            "worked": int(sum(1 for i in reviewed["alert_id"] if int(i) in acted)),
            "fp_reduction_expected": store.metrics["champion"]["fp_reduction"],
            "recall_expected": store.metrics["champion"]["recall"],
            "recall_ci": store.metrics["champion"]["recall_ci"],
            "recall_btl_estimate": pol.get("recall_estimated_from_btl"),
            "mode": pol.get("mode"), "policy_version": pol.get("version"),
        }

    @app.get("/api/alerts")
    def alerts(view: Literal["queue", "suppressed", "all"] = "queue", day: int | None = None,
               q: str | None = None, limit: int = Query(100, le=500), offset: int = 0):
        a = A if day is None else A[A["day"] == day]
        if view == "queue":
            a = a[a["decision"] != "SUPPRESS"].sort_values(["day", "probability"], ascending=[False, False])
        elif view == "suppressed":
            a = a[a["decision"] == "SUPPRESS"].sort_values(["day", "probability"], ascending=[False, False])
        else:
            a = a.sort_values(["day", "probability"], ascending=[False, False])
        if q:
            a = a[a["account"].str.contains(q, case=False, regex=False) | (a["alert_id"].astype(str) == q)]
        acted = last_actions()
        page = a.iloc[offset: offset + limit]
        return _clean({"total": len(a), "items": [_public(r, store, acted) for _, r in page.iterrows()]})

    @app.get("/api/alerts/{alert_id}")
    def alert(alert_id: int):
        rows = A[A["alert_id"] == alert_id]
        if rows.empty:
            raise HTTPException(404, "alert not found")
        r = rows.iloc[0]
        acct, day = r["account"], int(r["day"])
        out = _public(r, store, last_actions())
        out["reasons"] = json.loads(r["reasons_json"])
        out["rule_text"] = {x: RULE_TEXT.get(x, x) for x in out["rules"]}
        out["policy_version"], out["model_run_id"] = r["policy_version"], r["model_run_id"]
        t = store.txns
        today = t[(t["day"] == day) & ((t["src"] == acct) | (t["dst"] == acct))].sort_values("ts")
        out["transactions"] = [{"time": x.ts.strftime("%H:%M"), "direction": "out" if x.src == acct else "in",
                                "counterparty": x.dst if x.src == acct else x.src, "amount_usd": round(float(x.amount_usd), 2),
                                "format": x.pay_format, "currency": x.currency} for x in today.itertuples()][:200]
        hist = t[(t["day"] > day - 14) & (t["day"] <= day)]
        hs = hist[hist["src"] == acct].groupby("day")["amount_usd"].sum()
        hr = hist[hist["dst"] == acct].groupby("day")["amount_usd"].sum()
        out["history"] = [{"day": int(d), "date": store.date(d), "sent": float(hs.get(d, 0)), "received": float(hr.get(d, 0))}
                          for d in range(day - 13, day + 1)]
        out["graph"] = _ego_graph(store, acct, day)
        d = store.dispositions()
        out["dispositions"] = d[d["alert_id"] == alert_id].to_dict(orient="records") if not d.empty else []
        return _clean(out)

    @app.post("/api/alerts/{alert_id}/disposition")
    def dispose(alert_id: int, body: Disposition):
        rows = A[A["alert_id"] == alert_id]
        if rows.empty:
            raise HTTPException(404, "alert not found")
        r = rows.iloc[0]
        rec = (datetime.now(timezone.utc).isoformat(), alert_id, body.action, body.analyst, body.note,
               r["model_run_id"], r["policy_version"], float(r["probability"]), r["decision"])
        with closing(store.connect()) as c:
            c.execute("INSERT INTO audit (ts, alert_id, action, analyst, note, model_run_id, policy_version, probability, "
                      "routed_decision) VALUES (?,?,?,?,?,?,?,?,?)", rec)
            c.commit()
        return {"ok": True, "alert_id": alert_id, "action": body.action}

    @app.get("/api/audit")
    def audit(limit: int = Query(200, le=2000)):
        d = store.dispositions().head(limit)
        return {"items": d.to_dict(orient="records")}

    @app.get("/api/model")
    def model():
        keys = ["run_id", "created_at", "data", "alert_stats", "split", "target_fpr", "rules_only_baseline", "ladder",
                "augmentation", "champion", "curves", "per_rule", "per_typology", "feature_importance",
                "drift_psi_train_vs_test", "policy", "robustness"]
        out = {k: store.metrics.get(k) for k in keys}
        if store.manifest_path.exists():
            man = json.loads(store.manifest_path.read_text())
            out["manifest"] = {k: man[k] for k in ("status", "approvals", "git_sha", "data_fingerprint", "features")}
        return out

    @app.post("/api/model/approve")
    def approve(body: Approval):
        if not store.manifest_path.exists():
            raise HTTPException(404, "model manifest not found")
        man = json.loads(store.manifest_path.read_text())
        man["approvals"].append({**body.model_dump(), "ts": datetime.now(timezone.utc).isoformat()})
        roles = {a["role"] for a in man["approvals"]}
        man["status"] = "approved" if {"model_risk", "compliance"} <= roles else "in_review"
        store.manifest_path.write_text(json.dumps(man, indent=2))
        return {"status": man["status"], "approvals": man["approvals"]}

    @app.post("/api/policy/simulate")
    def simulate(body: Simulation):
        """What would a different review budget do on the test period? (hindsight labels, aggregate only)"""
        days_n = A["day"].nunique()
        share = body.review_share if body.review_share is not None else \
            min(1.0, (body.reviews_per_day or 0) * days_n / len(A))
        y = A["label"].astype(bool).to_numpy()
        order = np.argsort(-A["score_raw"].to_numpy(), kind="stable")
        k = int(round(share * len(A)))
        reviewed = np.zeros(len(A), bool)
        reviewed[order[:k]] = True
        return {"review_share": share, "reviews_per_day": round(k / days_n, 1),
                "recall": float((reviewed & y).sum() / max(y.sum(), 1)),
                "fp_reduction": float(1 - (reviewed & ~y).sum() / max((~y).sum(), 1)),
                "missed_suspicious": int((~reviewed & y).sum()), "suspicious_total": int(y.sum())}

    @app.get("/api/monitoring")
    def monitoring():
        g = A.groupby("day")
        series = [{"day": int(d), "date": store.date(d), "alerts": len(x),
                   "reviewed": int((x["decision"] == "REVIEW").sum()), "btl": int((x["decision"] == "BTL_REVIEW").sum()),
                   "suppressed": int((x["decision"] == "SUPPRESS").sum()),
                   "mean_probability": float(x["probability"].mean())} for d, x in g]
        return {"series": series, "psi": store.metrics.get("drift_psi_train_vs_test", []),
                "policy": store.metrics.get("policy", {})}

    ui = Path(__file__).resolve().parents[3] / "ui" / "dist"
    if ui.exists():
        app.mount("/assets", StaticFiles(directory=ui / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = ui / path
            return FileResponse(f if path and f.is_file() else ui / "index.html")

    return app


def _ego_graph(store: Store, acct: str, day: int, max_nb: int = 12, max_nb2: int = 3) -> dict:
    t = store.txns
    w = t[(t["day"] > day - store.tws) & (t["day"] <= day)]
    e1 = w[(w["src"] == acct) | (w["dst"] == acct)]
    nb = pd.concat([e1.loc[e1["src"] == acct, ["dst", "amount_usd"]].rename(columns={"dst": "n"}),
                    e1.loc[e1["dst"] == acct, ["src", "amount_usd"]].rename(columns={"src": "n"})])
    top = nb.groupby("n")["amount_usd"].sum().sort_values(ascending=False).head(max_nb).index.tolist()
    edges = e1[(e1["src"].isin(top)) | (e1["dst"].isin(top))]
    second = []
    for n in top:
        e2 = w[((w["src"] == n) | (w["dst"] == n)) & (w["src"] != acct) & (w["dst"] != acct)]
        other = np.where(e2["src"] == n, e2["dst"], e2["src"])
        keep = pd.Series(e2["amount_usd"].to_numpy(), index=other).groupby(level=0).sum().sort_values(ascending=False)
        chosen = keep.head(max_nb2).index.tolist()
        second += [(n, o) for o in chosen]
        edges = pd.concat([edges, e2[(e2["src"].isin(chosen)) | (e2["dst"].isin(chosen))]])
    nodes = {acct: 0, **{n: 1 for n in top}}
    for _, o in second:
        nodes.setdefault(o, 2)
    edges = edges[edges["src"].isin(nodes) & edges["dst"].isin(nodes)]
    agg = edges.groupby(["src", "dst"]).agg(amount=("amount_usd", "sum"), count=("amount_usd", "size"),
                                            last_day=("day", "max")).reset_index()
    sus = store.known_suspicious(list(nodes), day)
    return {
        "nodes": [{"id": n, "ring": r, "known_suspicious": n in sus, "center": n == acct} for n, r in nodes.items()],
        "edges": [{"src": x.src, "dst": x.dst, "amount": round(float(x.amount), 2), "count": int(x.count),
                   "days_ago": int(day - x.last_day)} for x in agg.itertuples()],
        "window_days": store.tws,
    }
