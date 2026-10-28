# UI rules

Who uses it: L1/L2 AML analysts working a daily queue, their team lead, and model-risk reviewers.
Its job: help an analyst decide one alert well and quickly, and let anyone check that suppression is safe.

## The signature: the review line
The product has one recurring visual idea: a red vertical **review line** (τ). It appears on every score meter, on the
capacity curve, and on the policy simulator. Above the line is reviewed; below the line is suppressed. Do not invent
other decorative motifs; spend visual emphasis only on the line and on high-risk evidence.

## Layout
- Desktop: left rail (sections) · list (440px) · detail. Mobile (<960px): rail becomes a top strip; list above detail.
- Detail reads top to bottom in the order an analyst works: who and why → evidence → network → history → payments → decision.
- Left-aligned text everywhere. Numbers right-aligned in tables, tabular figures always.

## Copy
- Sentence case. Plain verbs. Name things the way analysts do: "Below the line", "Escalate to L2", "Close as not suspicious".
- Buttons say what happens; the toast repeats the same verb ("Escalated: alert #123").
- Every model number shown to non-ML users gets a one-sentence reading next to it.
- Empty and error states say what to do next ("Is the API running on port 8000?"). No apologies.
- No all-caps labels, no eyebrow text above headings, no "→" on buttons.

## Data honesty
- Never show ground truth, typology, or the BTL flag in analyst views. BTL-sampled alerts are indistinguishable.
- Show uncertainty: CI ranges for comparisons, not just point estimates.
- Reason codes always show the alert's own value so the analyst can verify them against the payments table.
- Graph marks only accounts confirmed suspicious *before* the alert day.

## Accessibility and quality floor
- Keyboard: every action reachable; visible focus (`--focus`).
- Contrast AA for text; colour is never the only signal (bands also show the number; graph legend explains rings).
- SVG figures have `role="img"` and an `aria-label` describing what they show.
- Respect `prefers-reduced-motion` and `prefers-color-scheme`. No entrance animations.
- Must work at 390px wide without horizontal page scroll (tables scroll inside `.tablewrap`).

## Components
Use the existing ones before adding new: `Meter`, `LineChart`, `CiStrip`, `StackedBars`, `Network`, `.row`, `.ev`, `.band`, `.btn`.
