import type { ReactNode } from "react";

/* Small hand-drawn SVG charts. Colors come from tokens via CSS variables. */

interface Series { name: string; color: string; points: [number, number][]; dashed?: boolean }

export function LineChart({ series, xLabel, yLabel, marker, height = 240, max = 1,
  xFmt = (v) => `${+(v * 100).toFixed(max < 0.2 ? 1 : 0)}%`, yFmt = (v) => `${+(v * 100).toFixed(max < 0.2 ? 1 : 0)}%` }: {
  series: Series[]; xLabel: string; yLabel: string; marker?: { x: number; label: string }; height?: number;
  max?: number; xFmt?: (v: number) => string; yFmt?: (v: number) => string;
}) {
  const W = 560, H = height, L = 44, R = 12, T = 12, B = 36;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const x0 = 0, x1 = max ?? Math.max(...xs, 1);
  const sx = (v: number) => L + ((Math.min(v, x1) - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (v: number) => T + (1 - Math.min(v, max) / max) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
  return (
    <svg className="fig" style={{ maxWidth: 760 }} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} against ${xLabel}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line className="grid" x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} />
          <text x={L - 6} y={sy(t) + 4} textAnchor="end">{yFmt(t)}</text>
          <text x={sx(t)} y={H - B + 16} textAnchor="middle">{xFmt(t)}</text>
        </g>
      ))}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
      {series.map((s) => (
        <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined}
          points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
      ))}
      {marker && (
        <g>
          <line x1={sx(marker.x)} x2={sx(marker.x)} y1={T} y2={H - B} stroke="var(--line)" strokeWidth={2} />
          <text x={sx(marker.x) + 6} y={T + 12} style={{ fill: "var(--line)", fontWeight: 600 }}>{marker.label}</text>
        </g>
      )}
    </svg>
  );
}

/* Point estimate with 95% CI on a shared 0..1 scale: used to compare model variants honestly. */
export function CiStrip({ rows, lo = 0.5 }: { rows: { label: ReactNode; value: number; ci: [number, number]; pick?: boolean }[]; lo?: number }) {
  const W = 560, rowH = 30, L = 210, R = 16, H = rows.length * rowH + 28;
  const sx = (v: number) => L + ((v - lo) / (1 - lo)) * (W - L - R);
  return (
    <svg className="fig" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Recall with 95% confidence intervals">
      {[lo, (lo + 1) / 2, 1].map((t) => (
        <g key={t}><line className="grid" x1={sx(t)} x2={sx(t)} y1={0} y2={H - 22} />
          <text x={sx(t)} y={H - 6} textAnchor="middle">{Math.round(t * 100)}%</text></g>
      ))}
      {rows.map((r, i) => {
        const y = i * rowH + 16;
        return (
          <g key={i}>
            <text x={0} y={y + 4} style={{ fill: "var(--ink)", fontWeight: r.pick ? 700 : 400, fontSize: 12 }}>{r.label}</text>
            <line x1={sx(r.ci[0])} x2={sx(r.ci[1])} y1={y} y2={y} stroke="var(--ink-soft)" strokeWidth={2} />
            <circle cx={sx(r.value)} cy={y} r={5} fill={r.pick ? "var(--line)" : "var(--ink)"} />
          </g>
        );
      })}
    </svg>
  );
}

export function StackedBars({ data, keys, height = 200 }: {
  data: { label: string; values: Record<string, number> }[]; keys: { key: string; color: string; name: string }[]; height?: number;
}) {
  const W = 560, H = height, L = 40, B = 24, T = 8;
  const max = Math.max(...data.map((d) => keys.reduce((a, k) => a + (d.values[k.key] || 0), 0)), 1);
  const bw = (W - L) / data.length;
  return (
    <>
      <svg className="fig" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Alerts per day by outcome">
        {[0, 0.5, 1].map((t) => (
          <g key={t}><line className="grid" x1={L} x2={W} y1={T + (1 - t) * (H - T - B)} y2={T + (1 - t) * (H - T - B)} />
            <text x={L - 6} y={T + (1 - t) * (H - T - B) + 4} textAnchor="end">{Math.round(max * t)}</text></g>
        ))}
        {data.map((d, i) => {
          let acc = 0;
          return (
            <g key={i}>
              {keys.map((k) => {
                const v = d.values[k.key] || 0;
                const h = (v / max) * (H - T - B);
                const y = H - B - acc - h;
                acc += h;
                return <rect key={k.key} x={L + i * bw + 1} y={y} width={Math.max(bw - 2, 1)} height={h} fill={k.color}><title>{`${d.label}: ${k.name} ${v}`}</title></rect>;
              })}
              {i % Math.ceil(data.length / 6) === 0 && <text x={L + i * bw + bw / 2} y={H - 6} textAnchor="middle">{d.label}</text>}
            </g>
          );
        })}
      </svg>
      <div className="legend">{keys.map((k) => <span key={k.key}><i style={{ background: k.color }} />{k.name}</span>)}</div>
    </>
  );
}

/* Log scale from 0.01% to 100%: most alerts have tiny probabilities, and a linear bar would put them all at zero. */
export const meterPos = (p: number) => Math.max(0, Math.min(1, (Math.log10(Math.max(p, 1e-4)) + 4) / 4)) * 100;

export function Meter({ p, tau, band }: { p: number; tau: number; band: string }) {
  return (
    <div className="meter" aria-label={`Probability ${(p * 100).toFixed(1)} percent; review line at ${(tau * 100).toFixed(1)} percent`}>
      <div className={`fill ${band}`} style={{ width: `${Math.max(1, meterPos(p))}%` }} />
      <div className="tau" style={{ left: `${Math.min(99.5, meterPos(tau))}%` }} />
    </div>
  );
}
