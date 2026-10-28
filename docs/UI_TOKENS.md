# UI tokens

Source of truth in code: `ui/src/tokens.css`. Components use CSS variables only. Change a value here and there together.

## Palette: security paper and ink
Chosen from the material world of the subject (bank statement paper, ledger ink, a red review line), not a SaaS default.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `#f2f5f3` | `#0f1c1b` | page ground (cool, slightly green) |
| `--paper-raised` | `#fbfcfb` | `#152624` | selected row, inputs |
| `--paper-sunk` | `#e6ece9` | `#0b1514` | hover, meter track |
| `--ink` | `#17302e` | `#e3ebe8` | text, primary marks |
| `--ink-soft` | `#5b6b68` | `#9db0ab` | secondary text, axes |
| `--rule` / `--rule-strong` | `#c9d3cf` / `#9fb0ab` | `#2a3d3a` / `#3f5551` | dividers, borders |
| `--line` | `#c2362b` | `#ef6a5c` | the review line, high band, "raises score", escalate |
| `--review` | `#2f6f5e` | `#5fbf9f` | close / approve, "lowers score", focus |
| `--caution` | `#a8701a` | `#e0a948` | medium band |
| `--quiet` | `#8b9a96` | `#6f8480` | suppressed / low |
| `*-soft` | | | band backgrounds |

Bands (relative to the review line τ, because calibrated probabilities are honest and small at ~1% prevalence): high = p ≥ 10% (at least 1 in 10 alerts like this was suspicious, ~10× the base rate); medium = τ ≤ p < 10%; low = p < τ. Meters use a log scale from 0.01% to 100%.

## Type
One family: **Schibsted Grotesk** (newsprint grotesk; distinct from the usual UI defaults), tabular figures everywhere.

| Token | Size | Use |
|---|---|---|
| `--fs-xs` | 12 | band chips, status |
| `--fs-sm` | 13 | tables, captions |
| `--fs-md` | 15 | body |
| `--fs-lg` | 18 | section headings, lead text |
| `--fs-xl` | 24 | page / account heading |
| `--fs-xxl` | 34 | single key numbers |

Weights: 400 body, 600 headings and emphasis, 700 key numbers and brand.

## Space and shape
4px base: `--s1` 4 · `--s2` 8 · `--s3` 12 · `--s4` 16 · `--s5` 24 · `--s6` 32 · `--s7` 48.
Radii: `--radius-sm` 3px (controls, chips), `--radius-md` 6px (toast). No shadows; hierarchy comes from rules and ground tone.
Layout: `--rail-w` 208px, `--list-w` 440px.
