# CLAUDE.md

Project memory for Claude Code. Read this first, then `docs/PROGRESS.md` to see where work stands.
Shared rules for every coding agent (Claude, Codex, Cursor) live in `AGENTS.md`; this file adds Claude-specific notes.

## What this is

An alert triage model for AML transaction monitoring. A rule engine raises alerts (95–99% false); a
LightGBM model scores each alert using rule, account-history and transaction-graph features; a policy
suppresses or deprioritises low-score alerts, with a blind below-the-line (BTL) sample kept for safety.
Based on Eddin et al., AAAI 2022 (graph features, GuiltyWalker) and Bakry et al., J. Supercomputing 2023
(alert-level suppression). Data: IBM AMLworld (real file) or a generator writing the same format.

## Commands

```bash
make setup          # venv + python deps (+ GAN extras) + UI deps
make run            # full pipeline on synthetic data (~25 min on 2 cores; CTGAN is most of it)
make run-fast       # same with tiny search budgets (~4 min) for iteration
make run-ibm        # real IBM HI-Small (put the CSV in data/raw/ first; docs/DATA.md)
make get-amlsim     # IBM AMLSim example from Kaggle into data/raw/amlsim/
make run-amlsim     # pipeline on AMLSim (configs/amlsim.yaml: low-value rule + retuned thresholds), ~40 min
make run-amlsim-fast  # same with a small tuning budget, ~15 min
make doctor         # environment check with fixes
make test           # pytest, incl. the point-in-time test (~2 min)
make lint           # ruff
make serve          # API on :8000, serves ui/dist if built
make ui-dev         # Vite dev server on :5173, proxies /api to :8000
make ui-build
```

Single stage: `aml-triage run --stages train score --set training.n_trials=5`.
Robustness only (after a run): `aml-triage run --stages backtest` → `reports/<run>/backtest.md`.

## Where things live

| Path | What |
|---|---|
| `src/aml_triage/ingest/` | IBM reader (`ibm.py`), IBM-format generator (`synth.py`) |
| `src/aml_triage/rules/engine.py` | rule engine → alerts |
| `src/aml_triage/labels/store.py` | bitemporal labels, `LabelIndex` as-of lookups |
| `src/aml_triage/features/` | `profiles.py` (a_, p_), `graph.py` (g_, gw_, gwd_), `build.py` (assembly, stage-1) |
| `src/aml_triage/augment/` | SMOTE / CTGAN, training split only |
| `src/aml_triage/training/` | `train.py` (split, tune, calibrate, τ), `experiment.py` (ladder, arms, reports, scoring) |
| `src/aml_triage/evaluation/` | metrics, bootstrap CIs, PSI |
| `src/aml_triage/serving/` | `policy.py` (routing, BTL), `reasons.py` (reason codes) |
| `src/aml_triage/api/app.py` | FastAPI for the UI |
| `ui/` | React + Vite analyst console |
| `configs/` | every number the pipeline uses |
| `docs/` | architecture, plan, progress, evaluation, reviewer checklist, UI rules/tokens, data |

## Non-negotiables (a change that breaks one of these is wrong even if metrics go up)

1. **Point-in-time.** Anything computed for alert day *d* uses only transactions with day ≤ *d* and labels with
   `decision_day ≤ d`. `tests/test_point_in_time.py` must pass. If you add a feature, it must be covered by that test.
2. **Test data is never touched for choices.** Champion, features, τ, calibration are chosen on validation.
3. **Augmentation is training-only.** Synthetic rows never reach validation, test, metrics or the graph.
4. **Model and policy stay separate.** The model outputs a score; `serving/policy.py` decides. Policy changes are versioned.
5. **No ground truth in the analyst UI.** `/api/alerts*` never returns `label`; BTL alerts look like normal alerts.
6. **Config, not constants.** New numbers go in `configs/default.yaml` and `config.py`.

## How to work here

- Before coding: read the relevant section of `docs/ARCHITECTURE.md` and the current milestone in `docs/BUILD_PLAN.md`.
- Use the skills in `.claude/skills/` (run-pipeline, leakage-audit, add-feature, evaluate-change, ui-change).
- Any change to features, labels, splits or training: run `make test`, then `make run-fast`, and compare
  `reports/<run>/ablation.md` to the previous run. Report deltas *with* their CIs.
- UI work: follow `docs/UI_RULES.md`; colors and sizes only from `ui/src/tokens.css` (documented in `docs/UI_TOKENS.md`).
- When done: tick the item in `docs/PROGRESS.md`, add a dated line to its log, and ask the `reviewer` subagent
  (`.claude/agents/reviewer.md`) to check the diff against `docs/REVIEWER.md`.

## Gotchas

- `polars` rolling sums drift (1e-12 residue): guard divisions by the count, see `features/profiles.py`.
- Python `~True == -2` (truthy). Use `not flag` with plain bools in pandas masks (this bit `split()` once).
- `gw_` is a prefix of `gwd_`; use `features.build.columns_for`, never `startswith("gw")`.
- Isotonic calibration is a step function: decisions use `tau_raw` on the raw score; humans see calibrated p.
- CTGAN fit time is dominated by per-column mode-specific normalisation; it scales with feature count.
