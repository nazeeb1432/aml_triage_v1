import type { AlertDetail } from "./api";
import { money, short } from "./api";

/* Counterparty network around the alerted account: rings by hop distance, edges sized by amount.
   Red rings mark accounts already confirmed suspicious as of the alert day (never future labels). */
export function Network({ graph }: { graph: AlertDetail["graph"] }) {
  const W = 560, H = 420, cx = W / 2, cy = H / 2;
  const rings = [0, 115, 190];
  const pos: Record<string, [number, number]> = {};
  for (const ring of [0, 1, 2]) {
    const ns = graph.nodes.filter((n) => n.ring === ring);
    ns.forEach((n, i) => {
      const a = (i / Math.max(ns.length, 1)) * Math.PI * 2 - Math.PI / 2 + ring * 0.35;
      pos[n.id] = ring === 0 ? [cx, cy] : [cx + Math.cos(a) * rings[ring], cy + Math.sin(a) * rings[ring] * 0.92];
    });
  }
  const maxAmt = Math.max(...graph.edges.map((e) => e.amount), 1);
  if (graph.nodes.length <= 1) return <p className="muted">No counterparties in the last {graph.window_days} days.</p>;
  return (
    <figure style={{ margin: 0 }}>
      <svg className="fig" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Counterparty network">
        <defs>
          <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0 0L10 5L0 10z" fill="var(--rule-strong)" />
          </marker>
        </defs>
        {[1, 2].map((r) => <ellipse key={r} cx={cx} cy={cy} rx={rings[r]} ry={rings[r] * 0.92} fill="none" stroke="var(--rule)" strokeDasharray="2 4" />)}
        {graph.edges.map((e, i) => {
          const a = pos[e.src], b = pos[e.dst];
          if (!a || !b) return null;
          const dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) || 1, k = 12 / d;
          return (
            <line key={i} x1={a[0] + dx * k} y1={a[1] + dy * k} x2={b[0] - dx * k} y2={b[1] - dy * k}
              stroke="var(--rule-strong)" strokeWidth={1 + 4 * Math.sqrt(e.amount / maxAmt)} markerEnd="url(#arrowhead)">
              <title>{`${short(e.src)} → ${short(e.dst)}: ${money(e.amount)} in ${e.count} payment(s), last ${e.days_ago} day(s) before the alert`}</title>
            </line>
          );
        })}
        {graph.nodes.map((n) => {
          const [x, y] = pos[n.id];
          return (
            <g key={n.id}>
              <circle cx={x} cy={y} r={n.center ? 11 : 7} fill={n.center ? "var(--ink)" : "var(--paper-raised)"}
                stroke={n.known_suspicious ? "var(--line)" : "var(--ink-soft)"} strokeWidth={n.known_suspicious ? 3 : 1.2}>
                <title>{`${n.id}${n.known_suspicious ? " — confirmed suspicious before this alert" : ""}`}</title>
              </circle>
              {(n.center || n.known_suspicious) && (
                <text x={x} y={y + (n.center ? 28 : 21)} textAnchor="middle" style={{ fill: n.known_suspicious ? "var(--line)" : "var(--ink)", fontWeight: 600 }}>
                  {n.center ? "this account" : short(n.id)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="legend">
        <span><i style={{ background: "var(--ink)" }} />This account</span>
        <span><i style={{ background: "var(--paper-raised)", border: "2px solid var(--line)" }} />Confirmed suspicious before this alert</span>
        <span>Inner ring: direct counterparties. Outer ring: their counterparties. Line width = amount, last {graph.window_days} days. Hover for IDs.</span>
      </figcaption>
    </figure>
  );
}
