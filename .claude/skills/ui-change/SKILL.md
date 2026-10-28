---
name: ui-change
description: Make or review a change to the analyst console in ui/ following the project's UI rules and tokens. Use for any work on screens, components, copy, charts or the API shapes the UI reads.
---

# UI change

1. Read `docs/UI_RULES.md` and `docs/UI_TOKENS.md` first.
2. Reuse components (`Meter`, `LineChart`, `CiStrip`, `StackedBars`, `Network`) and classes in `app.css` before adding new ones.
   Colours and sizes only through CSS variables in `tokens.css`.
3. If the change needs new data, add it to `api/app.py` and the type in `ui/src/api.ts`. Analyst endpoints must not return
   `label`, `typology`, `decision_reason` or anything that reveals a BTL sample.
4. Copy: sentence case, plain verbs, the button label and its toast use the same verb. Empty/error states say what to do.
5. Check: `cd ui && npm run build`; run `make serve`, open http://127.0.0.1:8000, look at the change at 1280px and 390px,
   keyboard-tab through it, and in dark mode. Save a screenshot to `docs/img/` if the screen changed visibly.
