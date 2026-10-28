# Build plan

Each milestone has acceptance criteria someone other than the author can check. Status lives in `PROGRESS.md`.

## M0 — Foundations ✅
Repo layout, typed config, data contracts, CI, harness docs.
- `make lint test` green in CI. Every pipeline number is read from `configs/`.

## M1 — Data ✅
IBM AMLworld reader; generator writing the identical format with the 8 IBM typologies and rule-tripping legitimate behaviour.
- Reader unit test on a hand-written IBM snippet (incl. zero-padded bank IDs, patterns file).
- Synthetic rules-only false-positive rate between 95% and 99%.

## M2 — Rules and labels ✅
Seven-rule engine; alert labels from `Is Laundering`; simulated decision delay (positives slower than negatives).
- `test_rules_fire_on_their_patterns`, `test_own_label_is_never_known_on_alert_day`.

## M3 — Point-in-time features ✅
Profiles, alert features, degree, GW and GWd with as-of labels; numba walks; hub removal.
- `test_features_do_not_see_the_future` passes for two cut days, all families.
- Feature build for ~100k alerts under 2 minutes on a laptop.

## M4 — Training and evaluation ✅
Label-delay-aware split, Optuna tuning, gain-based selection, isotonic calibration, frozen τ, day-block bootstrap CIs,
ablation ladder, per-rule and per-typology recall, PSI.
- `reports/<run>/ablation.md` produced; champion picked on validation only (code path never reads test for selection).

## M5 — Augmentation experiment ✅
SMOTE and CTGAN arms on training rows only, fidelity AUC, same evaluation as the ladder.
- `test_augment_refuses_non_training_rows`, `test_smote_rows_are_flagged`.
- Report states plainly whether augmentation beat class weights, with CIs.

## M6 — Serving and UI ✅
Policy router with overrides and BTL, reason codes, FastAPI, analyst console (queue, below the line, model evidence,
policy simulator, monitoring, audit log, sign-off), Docker.
- `test_pipeline_and_api`: queue never returns labels or SUPPRESS rows; dispositions land in the audit log.
- UI passes `docs/UI_RULES.md` checklist at 1280px and 390px wide.

## M7 — Run on real IBM data ⏳ (next)
- ✅ AMLSim example: reader, calibrated rule profile, full run and robustness (`reports/amlsim/`).
- Download HI-Small; `make run-ibm`; adjust `configs/ibm_hi_small.yaml` windows to the printed day span.
- Acceptance: pipeline completes; ablation table and data card in `docs/DATA.md` updated with real numbers.
- Repeat on LI-Small (lower illicit ratio) and report both.

## M8 — Robustness of the evidence ✅
- Rolling-origin backtest (≥4 folds) instead of one split; report mean ± spread per feature family.
- Held-out typology test: train without one typology, test on it (does the model generalise or memorise?).
- Seeds: 3 seeds per config; a feature family "helps" only if its CI lower bound beats the baseline mean.

## M9 — Production platform
- Orchestration (Dagster assets per stage, daily partitions, backfills).
- Postgres for audit + label store; dispositions → labels with real `decision_day`.
- MLflow registry replacing `artifacts/registry.json`; promotion gated on two approvals (exists in UI as a stub).
- Monitoring job: PSI, alert volume, BTL-estimated recall, label-delay distribution; alerting on thresholds.
- Shadow mode: score live alerts, show nothing, compare to dispositions for 2–3 months.

## M10 — Go-live sequence (organisational, not code)
Prioritisation only → suppression on one low-risk rule family with BTL → widen. Each step signed off by compliance and MRM.
