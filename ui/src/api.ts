export type Band = "high" | "medium" | "low";

export interface AlertRow {
  alert_id: number;
  account: string;
  day: number;
  date: string;
  rules: string[];
  probability: number;
  priority_rank: number;
  band: Band;
  decision: "REVIEW" | "SUPPRESS";
  status: string;
  top_reason: string;
  a_sum_in: number;
  a_sum_out: number;
}

export interface Reason { feature: string; value: number | null; impact: number; direction: string; text: string }
export interface GraphNode { id: string; ring: number; known_suspicious: boolean; center: boolean }
export interface GraphEdge { src: string; dst: string; amount: number; count: number; days_ago: number }

export interface AlertDetail extends AlertRow {
  reasons: Reason[];
  rule_text: Record<string, string>;
  transactions: { time: string; direction: "in" | "out"; counterparty: string; amount_usd: number; format: string; currency: string }[];
  history: { day: number; date: string; sent: number; received: number }[];
  graph: { nodes: GraphNode[]; edges: GraphEdge[]; window_days: number };
  dispositions: { ts: string; action: string; analyst: string; note: string }[];
  policy_version: string;
  model_run_id: string;
}

export interface Summary {
  day: number; date: string; alerts: number; in_queue: number; suppressed: number; hard_overrides: number;
  worked: number; fp_reduction_expected: number; recall_expected: number; recall_ci: [number, number];
  recall_btl_estimate: number | null; mode: string; policy_version: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} returned ${r.status}`);
  return r.json();
}
async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path} returned ${r.status}: ${await r.text()}`);
  return r.json();
}

export const api = {
  health: () => get<{ run: string; model_run_id: string; policy_version: string }>("/api/health"),
  days: () => get<{ items: { day: number; date: string; alerts: number; reviewed: number }[] }>("/api/days"),
  summary: (day?: number) => get<Summary>(`/api/summary${day !== undefined ? `?day=${day}` : ""}`),
  alerts: (view: string, day?: number, q?: string) =>
    get<{ total: number; items: AlertRow[] }>(
      `/api/alerts?view=${view}&limit=300${day !== undefined ? `&day=${day}` : ""}${q ? `&q=${encodeURIComponent(q)}` : ""}`),
  alert: (id: number) => get<AlertDetail>(`/api/alerts/${id}`),
  dispose: (id: number, action: string, analyst: string, note: string) =>
    post(`/api/alerts/${id}/disposition`, { action, analyst, note }),
  audit: () => get<{ items: Json[] }>("/api/audit"),
  model: () => get<Json>("/api/model"),
  approve: (reviewer: string, role: string, note: string) => post<Json>("/api/model/approve", { reviewer, role, note }),
  simulate: (review_share: number) => post<Json>("/api/policy/simulate", { review_share }),
  monitoring: () => get<Json>("/api/monitoring"),
};

export const money = (v: number) =>
  v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e4 ? `$${(v / 1e3).toFixed(0)}k` : `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
export const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
/* probabilities: keep two significant digits for small values so a 0.019% review line doesn't print as 0.0% */
export const prob = (v: number) => (v >= 0.1 ? `${(v * 100).toFixed(0)}%` : v >= 0.01 ? `${(v * 100).toFixed(1)}%` : `${+(v * 100).toPrecision(2)}%`);
export const short = (acct: string) => (acct.length > 14 ? `${acct.slice(0, 6)}…${acct.slice(-5)}` : acct);
export const RULE_LABEL: Record<string, string> = {
  large_txn: "Large payment", structuring: "Structuring", fan_out: "Fan-out", fan_in: "Fan-in",
  spike: "Volume spike", high_risk_channel: "Cash / crypto", pass_through: "Pass-through", low_value: "Low-value transfers",
};
