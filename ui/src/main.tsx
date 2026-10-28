import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./tokens.css";
import "./app.css";
import type { Summary } from "./api";
import { api, pct } from "./api";
import { AuditPage, ModelPage, MonitorPage, PolicyPage } from "./Evidence";
import { Queue } from "./Queue";

type Page = "queue" | "suppressed" | "model" | "policy" | "monitoring" | "audit";
const PAGES: { id: Page; label: string }[] = [
  { id: "queue", label: "Review queue" },
  { id: "suppressed", label: "Below the line" },
  { id: "model", label: "Model evidence" },
  { id: "policy", label: "Review policy" },
  { id: "monitoring", label: "Monitoring" },
  { id: "audit", label: "Audit log" },
];

function App() {
  const initial = (location.hash.slice(1) as Page) || "queue";
  const [page, setPage] = useState<Page>(PAGES.some((p) => p.id === initial) ? initial : "queue");
  const [days, setDays] = useState<{ day: number; date: string }[]>([]);
  const [day, setDay] = useState<number | undefined>();
  const [s, setS] = useState<Summary | null>(null);
  const [tau, setTau] = useState(0.05);
  const [toast, setToast] = useState<string | null>(null);
  const analyst = "analyst.demo";

  useEffect(() => { api.days().then((d) => { setDays(d.items); setDay(d.items[d.items.length - 1]?.day); }); api.model().then((m) => setTau(m.champion.tau_calibrated)); }, []);
  useEffect(() => { if (day !== undefined) api.summary(day).then(setS); }, [day, toast]);
  useEffect(() => { location.hash = page; }, [page]);
  useEffect(() => {
    const onHash = () => { const h = location.hash.slice(1) as Page; if (PAGES.some((p) => p.id === h)) setPage(h); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => { if (!toast) return; const t = setTimeout(() => setToast(null), 3500); return () => clearTimeout(t); }, [toast]);

  return (
    <div className="shell">
      <nav className="rail" aria-label="Sections">
        <div className="brand">Alert Triage<small>Rule alerts, ranked by risk</small></div>
        <div className="nav">
          {PAGES.map((p) => <button key={p.id} aria-current={page === p.id ? "page" : undefined} onClick={() => setPage(p.id)}>{p.label}</button>)}
        </div>
        <div className="rail-foot">Signed in as {analyst}<br />{s?.policy_version}</div>
      </nav>
      <main className="main">
        {(page === "queue" || page === "suppressed") && (
          <div className="topbar">
            <select className="field" value={day} onChange={(e) => setDay(Number(e.target.value))} aria-label="Alert day">
              {days.map((d) => <option key={d.day} value={d.day}>{d.date}</option>)}
            </select>
            {s && (
              <div className="facts">
                <span><b>{s.alerts}</b> alerts raised</span>
                <span><b>{s.in_queue}</b> to review</span>
                <span><b>{s.suppressed}</b> below the line</span>
                <span><b>{s.worked}</b> decided</span>
                <span>Expected to keep <b>{pct(s.recall_expected)}</b> of suspicious alerts</span>
              </div>
            )}
          </div>
        )}
        {page === "queue" && <Queue key={`q${day}`} view="queue" day={day} tau={tau} analyst={analyst} onToast={setToast} />}
        {page === "suppressed" && <Queue key={`s${day}`} view="suppressed" day={day} tau={tau} analyst={analyst} onToast={setToast} />}
        {page === "model" && <ModelPage onToast={setToast} />}
        {page === "policy" && <PolicyPage />}
        {page === "monitoring" && <MonitorPage />}
        {page === "audit" && <AuditPage />}
      </main>
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
