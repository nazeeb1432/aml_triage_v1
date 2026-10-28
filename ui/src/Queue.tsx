import { useEffect, useState } from "react";
import type { AlertDetail, AlertRow } from "./api";
import { api, money, prob, RULE_LABEL, short } from "./api";
import { Meter } from "./charts";
import { Network } from "./Network";

const ACTION_DONE: Record<string, string> = { escalate: "Escalated", close: "Closed as not suspicious", sar: "Marked for SAR" };

export function Queue({ view, day, tau, analyst, onToast }: {
  view: "queue" | "suppressed"; day?: number; tau: number; analyst: string; onToast: (m: string) => void;
}) {
  const [rows, setRows] = useState<AlertRow[]>([]);
  const [total, setTotal] = useState(0);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    api.alerts(view, day, q || undefined)
      .then((r) => { setRows(r.items); setTotal(r.total); setErr(null); if (r.items.length && sel === null) setSel(r.items[0].alert_id); })
      .catch((e) => setErr(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, day, q, tick]);

  return (
    <div className="split">
      <div className="list" role="listbox" aria-label={view === "queue" ? "Alerts to review" : "Suppressed alerts"}>
        <div className="list-head">
          <input placeholder="Find by account or alert ID" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search alerts" />
          <span className="small muted num">{total.toLocaleString()}</span>
        </div>
        {err && <p className="empty">Could not load alerts: {err}. Is the API running on port 8000?</p>}
        {!err && rows.length === 0 && <p className="empty">{view === "queue" ? "Nothing to review for this day." : "No alerts were suppressed on this day."}</p>}
        {rows.map((r) => (
          <button key={r.alert_id} className="row" role="option" aria-selected={sel === r.alert_id} onClick={() => setSel(r.alert_id)}>
            <span className="acct">{short(r.account)} <span className="muted small">#{r.alert_id}</span></span>
            <span><span className={`band ${r.band}`}>{prob(r.probability)}</span></span>
            <Meter p={r.probability} tau={tau} band={r.band} />
            <span className="why">{r.rules.map((x) => RULE_LABEL[x] ?? x).join(", ")} — {r.top_reason}</span>
            {r.status !== "open" && <span className={`status ${r.status}`}>{ACTION_DONE[r.status] ?? r.status}</span>}
          </button>
        ))}
      </div>
      <div className="detail">
        {sel !== null ? <Detail id={sel} tau={tau} analyst={analyst} view={view}
          onDone={(m) => { onToast(m); setTick((t) => t + 1); }} /> : <p className="empty">Select an alert.</p>}
      </div>
    </div>
  );
}

function Detail({ id, tau, analyst, view, onDone }: { id: number; tau: number; analyst: string; view: string; onDone: (m: string) => void }) {
  const [a, setA] = useState<AlertDetail | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);
  useEffect(() => { setA(null); api.alert(id).then(setA); }, [id, tick]);
  if (!a) return <p className="muted">Loading alert #{id}…</p>;
  const maxImpact = Math.max(...a.reasons.map((r) => Math.abs(r.impact)), 1e-9);
  const act = async (action: string) => {
    setBusy(true);
    try {
      await api.dispose(a.alert_id, action, analyst, note);
      setNote("");
      onDone(`${ACTION_DONE[action]}: alert #${a.alert_id}`);
      setTick((t) => t + 1);
    } finally { setBusy(false); }
  };
  const histMax = Math.max(...a.history.map((h) => Math.max(h.sent, h.received)), 1);
  return (
    <>
      <section className="section">
        <div className="actions" style={{ justifyContent: "space-between" }}>
          <h1>{a.account}</h1>
          <span className={`band ${a.band}`} style={{ fontSize: "var(--fs-md)" }}>Score {prob(a.probability)}</span>
        </div>
        <p className="muted">Alert #{a.alert_id} raised {a.date} by {a.rules.map((r) => RULE_LABEL[r] ?? r).join(" and ")}.
          {view === "suppressed" && " The model scored this alert below the review line, so it was suppressed. It stays here for supervisors to spot-check."}</p>
        <div style={{ maxWidth: 520 }}><Meter p={a.probability} tau={tau} band={a.band} /></div>
        <p className="small muted">The red mark is the review line ({prob(tau)}). It was fixed on past data so that about 80% of false alerts fall below it.</p>
      </section>

      <section className="section">
        <h2>Why this score</h2>
        <div className="evidence">
          {a.reasons.map((r) => {
            const w = (Math.abs(r.impact) / maxImpact) * 50;
            return (
              <div className="ev" key={r.feature}>
                <div className="bar" aria-hidden="true"><span className={r.impact > 0 ? "up" : "down"} style={{ width: `${w}%` }} /></div>
                <div><div>{r.text}</div><div className="small muted">{r.impact > 0 ? "Raises" : "Lowers"} the score</div></div>
              </div>
            );
          })}
        </div>
        <div>{a.rules.map((r) => <span className="chip" key={r}>{RULE_LABEL[r] ?? r}: {a.rule_text[r]}</span>)}</div>
      </section>

      <section className="section">
        <h2>Who the money moved with</h2>
        <Network graph={a.graph} />
      </section>

      <section className="section">
        <h2>Last 14 days</h2>
        <svg className="fig" viewBox="0 0 560 120" role="img" aria-label="Daily sent and received amounts, last 14 days">
          {a.history.map((h, i) => {
            const x = 8 + i * 39, s = (h.sent / histMax) * 90, r = (h.received / histMax) * 90;
            return (
              <g key={h.day}>
                <rect x={x} y={100 - r} width={16} height={r} fill="var(--review)"><title>{`${h.date} received ${money(h.received)}`}</title></rect>
                <rect x={x + 17} y={100 - s} width={16} height={s} fill="var(--line)"><title>{`${h.date} sent ${money(h.sent)}`}</title></rect>
                {i % 3 === 0 && <text x={x + 16} y={116} textAnchor="middle">{h.date.slice(5)}</text>}
              </g>
            );
          })}
        </svg>
        <div className="legend"><span><i style={{ background: "var(--review)" }} />Received</span><span><i style={{ background: "var(--line)" }} />Sent</span></div>
      </section>

      <section className="section">
        <h2>Payments on {a.date}</h2>
        <div className="tablewrap">
          <table>
            <thead><tr><th>Time</th><th>Direction</th><th>Counterparty</th><th className="r">Amount (USD)</th><th>Channel</th></tr></thead>
            <tbody>
              {a.transactions.map((t, i) => (
                <tr key={i}><td>{t.time}</td><td>{t.direction === "in" ? "Received" : "Sent"}</td><td>{short(t.counterparty)}</td>
                  <td className="r">{t.amount_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td><td>{t.format}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section">
        <h2>Decision</h2>
        {a.dispositions.length > 0 && (
          <ul className="small">{a.dispositions.map((d, i) => <li key={i}>{ACTION_DONE[d.action] ?? d.action} by {d.analyst} on {d.ts.slice(0, 16).replace("T", " ")}{d.note ? ` — ${d.note}` : ""}</li>)}</ul>
        )}
        <textarea rows={3} placeholder="What did you check? This note goes into the audit log." value={note} onChange={(e) => setNote(e.target.value)} aria-label="Decision note" />
        <div className="actions">
          <button className="btn primary" disabled={busy} onClick={() => act("escalate")}>Escalate to L2</button>
          <button className="btn" disabled={busy} onClick={() => act("sar")}>Mark for SAR</button>
          <button className="btn go" disabled={busy} onClick={() => act("close")}>Close as not suspicious</button>
        </div>
        <p className="small muted">Recorded with model {a.model_run_id} and policy {a.policy_version}.</p>
      </section>
    </>
  );
}
