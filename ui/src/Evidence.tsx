import { useEffect, useState } from "react";
import type { Json } from "./api";
import { api, pct } from "./api";
import { CiStrip, LineChart, StackedBars } from "./charts";

const SET_LABEL: Record<string, string> = {
  alert: "Rule details only",
  "alert+profiles": "+ account history",
  "alert+profiles+degree": "+ network degrees",
  "alert+profiles+degree+gw": "+ distance to confirmed (GW)",
  "alert+profiles+degree+gwd": "+ distance incl. pending (GWd)",
};
const ARM_LABEL: Record<string, string> = {
  none: "No rebalancing", class_weight: "Class weights", smote: "SMOTE oversampling", ctgan: "CTGAN synthetic rows",
};

export function ModelPage({ onToast }: { onToast: (m: string) => void }) {
  const [m, setM] = useState<Json | null>(null);
  const [who, setWho] = useState("");
  const [role, setRole] = useState("model_risk");
  useEffect(() => { api.model().then(setM); }, []);
  if (!m) return <p className="empty">Loading model evidence…</p>;
  const c = m.champion;
  const curve = m.curves.recall_vs_review_share.map((p: Json) => [p.review_share, p.recall]);
  const reviewShare = m.policy?.reviewed_per_day && m.policy?.alerts_per_day ? m.policy.reviewed_per_day / m.policy.alerts_per_day : c.review_share;
  const rulesFp = m.alert_stats.false_positive_rate_of_rules;
  return (
    <div className="page">
      <section className="section">
        <h1>Is the model safe to use?</h1>
        <p className="lead">On the {m.split.test.alerts.toLocaleString()} alerts it never saw, the model would let analysts skip {pct(c.fp_reduction)} of
          false alerts while still sending {pct(c.recall)} of suspicious ones to review (95% range {pct(c.recall_ci[0])}–{pct(c.recall_ci[1])}).</p>
        <div className="facts-row">
          <div><div className="bignum">{pct(rulesFp, 1)}</div><div className="small muted">of rule alerts are false today</div></div>
          <div><div className="bignum">{m.split.test.positives}</div><div className="small muted">suspicious alerts in the test period</div></div>
          <div><div className="bignum">{pct(c.pr_auc, 0)}</div><div className="small muted">PR-AUC (base rate {pct(m.split.test.positives / m.split.test.alerts, 1)})</div></div>
          <div><div className="bignum">{c.n_features}</div><div className="small muted">features in the chosen model</div></div>
        </div>
      </section>

      <section className="section">
        <h2>Review budget against suspicious alerts caught</h2>
        <p className="muted">Alerts sorted by score. Reading left to right: if analysts review this share of alerts, this share of suspicious alerts is caught.</p>
        <LineChart xLabel="Share of alerts reviewed" yLabel="Suspicious alerts caught"
          series={[{ name: "Model", color: "var(--ink)", points: curve }, { name: "Random order", color: "var(--rule-strong)", dashed: true, points: [[0, 0], [1, 1]] }]}
          marker={{ x: reviewShare, label: "current review line" }} />
      </section>

      <div className="grid2">
        <section className="section">
          <h2>What each feature family adds</h2>
          <p className="muted small">Recall on the test period at the frozen review line, with 95% day-resampled ranges. Overlapping ranges mean the difference may be noise.</p>
          <CiStrip rows={m.ladder.map((r: Json) => ({ label: SET_LABEL[r.feature_set] ?? r.feature_set, value: r.recall, ci: r.recall_ci, pick: r.feature_set === c.feature_set && c.key[0] === "ladder" }))} />
        </section>
        <section className="section">
          <h2>Does synthetic data help?</h2>
          <p className="muted small">Same features and settings; only the training data changes. Validation and test data are never augmented.</p>
          <CiStrip rows={m.augmentation.map((r: Json) => ({ label: ARM_LABEL[r.augmentation] ?? r.augmentation, value: r.recall, ci: r.recall_ci, pick: r.augmentation === c.augmentation && c.key[0] === "arm" }))} />
          <div className="tablewrap"><table>
            <thead><tr><th>Training data</th><th className="r">Synthetic rows</th><th className="r">Real vs synthetic separable</th><th className="r">Val pAUC</th></tr></thead>
            <tbody>{m.augmentation.map((r: Json) => (
              <tr key={r.augmentation}><td>{ARM_LABEL[r.augmentation]}</td><td className="r">{(r.n_synthetic ?? 0).toLocaleString()}</td>
                <td className="r">{r.fidelity_auc_real_vs_synth ? pct(r.fidelity_auc_real_vs_synth) : "—"}</td><td className="r">{r.val_pauc.toFixed(3)}</td></tr>))}
            </tbody></table></div>
          <p className="small muted">"Separable" is how well a classifier tells real suspicious rows from synthetic ones: 50% means indistinguishable, 100% means obviously fake.</p>
        </section>
      </div>

      {m.robustness && (
        <section className="section">
          <h2>Does it hold up?</h2>
          <p className="muted">One test period can flatter a model. Here the whole procedure is repeated at {m.robustness.folds.length} earlier
            cut dates, each using only the labels that existed then.{m.robustness.seed_spread?.sd !== undefined &&
            ` Refitting with ${m.robustness.seeds.length} random seeds moves recall between ${pct(m.robustness.seed_spread.min, 1)} and ${pct(m.robustness.seed_spread.max, 1)}, so smaller differences than that are noise.`}</p>
          <div className="grid2">
            <div>
              <CiStrip rows={m.robustness.folds.map((f: Json) => ({ label: `Test days ${f.test_days[0]}–${f.test_days[1]} (${f.n_positives} suspicious)`, value: f.recall, ci: f.recall_ci }))} />
              <div className="tablewrap"><table>
                <thead><tr><th>Features</th><th className="r">Mean recall</th><th className="r">Spread</th><th className="r">Periods where it beat the row above</th></tr></thead>
                <tbody>{m.robustness.families.map((r: Json) => (
                  <tr key={r.feature_set}><td>{SET_LABEL[r.feature_set] ?? r.feature_set}</td><td className="r">{pct(r.mean_recall, 1)}</td>
                    <td className="r">±{(r.sd_recall * 100).toFixed(1)}</td><td className="r">{r.folds_better_than_previous ?? "—"}</td></tr>))}</tbody>
              </table></div>
            </div>
            <div>
              <h3>A pattern it has never seen</h3>
              <p className="small muted">Every training example of one laundering pattern is removed, then the model is tested on that pattern.
                A big drop means the model memorised the pattern instead of learning what makes money movement suspicious.</p>
              <div className="tablewrap"><table>
                <thead><tr><th>Pattern</th><th className="r">Test cases</th><th className="r">Seen in training</th><th className="r">Never seen</th></tr></thead>
                <tbody>{m.robustness.held_out_typology.map((r: Json) => (
                  <tr key={r.typology}><td>{r.typology.toLowerCase()}</td><td className="r">{r.test_positives}</td>
                    <td className="r">{pct(r.recall_when_seen)}</td><td className="r">{pct(r.recall_when_unseen)}</td></tr>))}</tbody>
              </table></div>
            </div>
          </div>
        </section>
      )}

      <div className="grid2">
        <section className="section">
          <h2>Caught by laundering pattern</h2>
          <div className="tablewrap"><table>
            <thead><tr><th>Pattern</th><th className="r">Suspicious alerts</th><th className="r">Sent to review</th></tr></thead>
            <tbody>{[...m.per_typology].sort((a: Json, b: Json) => a.recall - b.recall).map((t: Json) => (
              <tr key={t.typology}><td>{t.typology.toLowerCase()}</td><td className="r">{t.n_positive_alerts}</td><td className="r">{pct(t.recall)}</td></tr>))}</tbody>
          </table></div>
        </section>
        <section className="section">
          <h2>By rule</h2>
          <div className="tablewrap"><table>
            <thead><tr><th>Rule</th><th className="r">Alerts</th><th className="r">Suspicious</th><th className="r">Kept</th><th className="r">False alerts removed</th></tr></thead>
            <tbody>{m.per_rule.map((r: Json) => (
              <tr key={r.rule}><td>{r.rule.replace(/_/g, " ")}</td><td className="r">{r.n_alerts.toLocaleString()}</td><td className="r">{r.n_positives}</td>
                <td className="r">{pct(r.recall)}</td><td className="r">{pct(r.fp_reduction)}</td></tr>))}</tbody>
          </table></div>
        </section>
      </div>

      <div className="grid2">
        <section className="section">
          <h2>Are the probabilities honest?</h2>
          <LineChart xLabel="Predicted probability" yLabel="Observed share suspicious"
            max={Math.min(1, Math.ceil(Math.max(...m.curves.calibration.map((b: Json) => Math.max(b.mean_pred, b.frac_pos))) * 1.2 * 20) / 20 || 1)}
            series={[{ name: "Model", color: "var(--ink)", points: m.curves.calibration.map((b: Json) => [b.mean_pred, b.frac_pos]) },
              { name: "Perfect", color: "var(--rule-strong)", dashed: true, points: [[0, 0], [1, 1]] }]} />
          <p className="small muted">Each point is a tenth of test alerts, by predicted probability. Points on the dashed line mean "a 5% score" really is suspicious about 5% of the time. Brier score {c.brier.toFixed(4)}; {c.calibration === "platt" ? "sigmoid (Platt)" : "isotonic"} calibration fitted on validation data.</p>
        </section>
        <section className="section">
          <h2>Strongest signals</h2>
          {(() => {
            const top = m.feature_importance.slice(0, 12);
            const mx = Math.max(...top.map((f: Json) => f.gain));
            return (
              <div className="evidence">{top.map((f: Json) => (
                <div className="ev" key={f.feature} style={{ gridTemplateColumns: "1fr 140px" }}>
                  <code className="small">{f.feature}</code>
                  <div style={{ height: 6, background: "var(--paper-sunk)" }}><div style={{ width: `${(f.gain / mx) * 100}%`, height: 6, background: "var(--ink)" }} /></div>
                </div>))}
              </div>
            );
          })()}
        </section>
      </div>

      <section className="section">
        <h2>Sign-off</h2>
        <p className="muted">Status: <b>{m.manifest?.status ?? "unknown"}</b>. The model can move to production only after model risk and compliance both approve. Built from data {m.manifest?.data_fingerprint}, code {m.manifest?.git_sha}.</p>
        {m.manifest?.approvals?.length > 0 && <ul className="small">{m.manifest.approvals.map((a: Json, i: number) => <li key={i}>{a.reviewer} ({a.role.replace("_", " ")}) on {a.ts.slice(0, 10)}{a.note ? ` — ${a.note}` : ""}</li>)}</ul>}
        <div className="actions">
          <input className="field" style={{ maxWidth: 220 }} placeholder="Your name" value={who} onChange={(e) => setWho(e.target.value)} aria-label="Reviewer name" />
          <select className="field" value={role} onChange={(e) => setRole(e.target.value)} aria-label="Reviewer role">
            <option value="model_risk">Model risk</option><option value="compliance">Compliance</option><option value="engineering">Engineering</option>
          </select>
          <button className="btn go" disabled={!who} onClick={async () => { const r = await api.approve(who, role, ""); onToast(`Approval recorded. Status: ${r.status}`); setM({ ...m, manifest: { ...m.manifest, status: r.status, approvals: r.approvals } }); }}>Record approval</button>
        </div>
      </section>
    </div>
  );
}

export function PolicyPage() {
  const [m, setM] = useState<Json | null>(null);
  const [share, setShare] = useState(0.25);
  const [sim, setSim] = useState<Json | null>(null);
  useEffect(() => { api.model().then((x) => { setM(x); const s = x.policy?.reviewed_per_day / x.policy?.alerts_per_day; if (s) setShare(Math.round(s * 100) / 100); }); }, []);
  useEffect(() => { const t = setTimeout(() => api.simulate(share).then(setSim), 120); return () => clearTimeout(t); }, [share]);
  if (!m) return <p className="empty">Loading policy…</p>;
  const p = m.policy;
  const curve = m.curves.recall_vs_review_share.map((x: Json) => [x.review_share, x.recall]);
  return (
    <div className="page">
      <section className="section">
        <h1>Where should the line go?</h1>
        <p className="lead">Move the review budget and see what it costs. Numbers come from the test period, where we know the outcome of every alert.</p>
        <label className="small muted" htmlFor="share">Analysts review {pct(share)} of alerts{sim ? ` (${sim.reviews_per_day} a day)` : ""}</label>
        <input id="share" type="range" min={0.02} max={1} step={0.01} value={share} onChange={(e) => setShare(Number(e.target.value))} />
        {sim && (
          <div className="facts-row">
            <div><div className="bignum">{pct(sim.recall, 1)}</div><div className="small muted">of suspicious alerts reviewed</div></div>
            <div><div className="bignum">{sim.missed_suspicious}</div><div className="small muted">suspicious alerts below the line (of {sim.suspicious_total})</div></div>
            <div><div className="bignum">{pct(sim.fp_reduction)}</div><div className="small muted">of false alerts no longer reviewed</div></div>
          </div>
        )}
        <LineChart xLabel="Share of alerts reviewed" yLabel="Suspicious alerts caught" series={[{ name: "Model", color: "var(--ink)", points: curve }]}
          marker={{ x: share, label: "review line" }} />
      </section>
      <section className="section">
        <h2>Active policy {p.version}</h2>
        <div className="tablewrap"><table><tbody>
          <tr><th>Mode</th><td>{p.mode === "suppress" ? "Suppress alerts below the line" : "Review everything, ordered by score"}</td></tr>
          <tr><th>Never suppressed</th><td>Rules: {p.config.never_suppress_rules.join(", ") || "none"}; accounts with an earlier confirmed-suspicious alert: {p.config.override_prior_positive ? "yes" : "no"}</td></tr>
          <tr><th>Below-the-line sample</th><td>{p.config.btl.strata.map((s: Json) => `${pct(s.rate)} of alerts scoring ${Math.round(s.lo_frac_of_tau * 100)}–${Math.round(s.hi_frac_of_tau * 100)}% of the line`).join("; ")}. Sampled alerts look like any other in the queue.</td></tr>
          <tr><th>Volume</th><td>{p.alerts_per_day} alerts a day; {p.reviewed_per_day} reviewed a day</td></tr>
          <tr><th>Recall, as production would measure it</th><td>{pct(p.recall_estimated_from_btl, 1)} estimated from the below-the-line sample
            {p.recall_estimated_from_btl_ci ? ` (95% range ${pct(p.recall_estimated_from_btl_ci[0])}–${pct(p.recall_estimated_from_btl_ci[1])}, from ${p.btl_positives_found} suspicious alerts found in ${p.btl_sampled} sampled)` : ""};
            {" "}{pct(p.recall_true_hindsight, 1)} in hindsight. The estimate is all a live system can know, so the sample has to stay on, and a wide range means it needs to be larger.</td></tr>
        </tbody></table></div>
      </section>
    </div>
  );
}

export function MonitorPage() {
  const [m, setM] = useState<Json | null>(null);
  useEffect(() => { api.monitoring().then(setM); }, []);
  if (!m) return <p className="empty">Loading monitoring…</p>;
  return (
    <div className="page">
      <section className="section">
        <h1>Daily volume</h1>
        <StackedBars keys={[{ key: "reviewed", color: "var(--ink)", name: "Reviewed (score or override)" }, { key: "btl", color: "var(--line)", name: "Below-the-line sample" }, { key: "suppressed", color: "var(--quiet-soft)", name: "Suppressed" }]}
          data={m.series.map((d: Json) => ({ label: d.date.slice(5), values: d }))} />
      </section>
      <section className="section">
        <h2>Has the data shifted since training?</h2>
        <p className="muted small">Population stability index of the strongest features, training period against the live period. Above 0.2 needs a look; above 0.25 usually means retrain.</p>
        <div className="tablewrap"><table>
          <thead><tr><th>Feature</th><th className="r">PSI</th><th>Reading</th></tr></thead>
          <tbody>{m.psi.map((r: Json) => (
            <tr key={r.feature}><td><code>{r.feature}</code></td><td className="r">{r.psi.toFixed(3)}</td>
              <td>{r.psi > 0.25 ? "Shifted: investigate" : r.psi > 0.1 ? "Watch" : "Stable"}</td></tr>))}</tbody>
        </table></div>
      </section>
    </div>
  );
}

export function AuditPage() {
  const [rows, setRows] = useState<Json[] | null>(null);
  useEffect(() => { api.audit().then((r) => setRows(r.items)); }, []);
  if (!rows) return <p className="empty">Loading audit log…</p>;
  return (
    <div className="page">
      <section className="section">
        <h1>Audit log</h1>
        <p className="muted">Every decision, with the model and policy version that routed the alert. Kept so the bank can answer "why was this alert handled this way?" years later.</p>
        {rows.length === 0 ? <p className="empty">No decisions yet. Open an alert in the queue and record one.</p> : (
          <div className="tablewrap"><table>
            <thead><tr><th>When (UTC)</th><th>Alert</th><th>Decision</th><th>Analyst</th><th>Note</th><th className="r">Score</th><th>Routed as</th><th>Model / policy</th></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.id}><td>{r.ts.slice(0, 19).replace("T", " ")}</td><td>#{r.alert_id}</td><td>{r.action}</td><td>{r.analyst}</td><td>{r.note}</td>
                <td className="r">{pct(r.probability)}</td><td>{r.routed_decision === "BTL_REVIEW" ? "Below-the-line sample" : r.routed_decision.toLowerCase()}</td>
                <td className="small muted">{r.model_run_id}<br />{r.policy_version}</td></tr>))}</tbody>
          </table></div>
        )}
      </section>
    </div>
  );
}
