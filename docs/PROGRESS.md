# Progress tracker

Agents: update this file at the end of every task (tick, then one dated log line). Humans: read the log bottom-up.

## Milestones

| | Milestone | Status | Evidence |
|---|---|---|---|
| ✅ | M0 Foundations | done | CI config, `configs/`, contracts |
| ✅ | M1 Data: IBM reader + IBM-format generator | done | `test_ibm_reader`, `data/work/*/data_meta.json` |
| ✅ | M2 Rules + bitemporal labels | done | `alert_stats.json`, rule tests |
| ✅ | M3 Point-in-time features (a, p, g, gw, gwd) | done | `test_features_do_not_see_the_future` |
| ✅ | M4 Training + evaluation with CIs | done | `reports/synthetic-default/ablation.md` |
| ✅ | M5 Augmentation arms (SMOTE, CTGAN) | done | augmentation table in ablation.md |
| ✅ | M6 Policy, reasons, API, analyst UI, Docker | done | `test_pipeline_and_api`, screenshots in `docs/img/` |
| ✅ | M6.1 GW hit-rate drift | done | top feature PSI now 0.011 (DECISIONS D11) |
| ✅ | M7a Real IBM data: AMLSim example | done | `reports/amlsim/`, README results |
| ⏳ | M7b IBM AMLworld HI-Small / LI-Small | next | needs the Kaggle files |
| ✅ | M8 Rolling-origin backtest, held-out typology, seeds, fold-based selection | done | `reports/synthetic-default/backtest.md`, D12–D13 |
| ⬜ | M9 Orchestration, Postgres, MLflow, monitoring jobs, shadow mode | | |
| ⬜ | M10 Go-live sequence with compliance and MRM | | |

## Open questions for the team
- Which outcome is the training label at our bank: L2 escalation, SAR filed, or both (train on one, evaluate on the other)?
- What is the real review capacity per day, and is it fixed or does it flex?
- Which rule families must never be suppressed?
- Unit of review: alert, account-day, customer, or case?
- What below-the-line sample size can the team afford? At 10% just below the line, the recall estimate's range is ±9 points.

## Log
- 2026-10-25 — M0–M6 built. Replication leak fixes (as-of labels, label-delay split, as-of window retention), synthetic
  world in IBM format, full pipeline, API and UI. 13 tests green incl. point-in-time.
- 2026-10-25 — Monitoring PSI flagged `gwd_hit_rate` (1.96) and `p_account_age_days` (8.1) as time trends. Added 60-day
  label memory for walks and capped account age (DECISIONS D10); gwd_hit_rate PSI now 0.33 (still above 0.25: watch).
  Run `synthetic-default-20261025T090437Z`: recall 94.1% (91–97), FP reduction 79.8%. SMOTE/CTGAN: no gain.
- 2026-10-25 — M6.1: GW hit-rate ramp came from the label store warming up; memory 60 → 21 days, generator lets schemes
  run to the end. Top-feature PSI 0.33 → 0.011.
- 2026-10-25 — M8: backtest stage (4 rolling folds, families across folds, held-out pattern, 5 seeds). It showed the
  validation-chosen model losing to `+gw` in every fold, so selection moved to pre-test folds with a 0.01 parsimony
  margin (D12–D13). BTL recall estimate now has a bootstrap range. Run `synthetic-default-20261025T104314Z`: recall
  93.3% (91–96), FP reduction 78.7%; backtest 86.6–94.7%; no memorisation; augmentation still no gain. 14 tests green.
- 2026-10-25 — AMLSim (Kaggle anshankul/ibm-amlsim-example-dataset): downloaded through the Composio sandbox
  (Kaggle is not reachable from the build workspace; the sandbox has 1 GB RAM, too small for the pipeline). Calibrated
  rules on the full file there: new low-value rule (<$25) + fan-in ≥4, fan-out ≥10, large ≥$1M → ~1,870 alerts/day,
  0.9% precision, 100% coverage. Added `ingest/amlsim.py`, `configs/amlsim.yaml`, `scripts/get_amlsim.py`,
  `make get-amlsim run-amlsim`, 2 tests (16 green). The new rule is off in the default config; its always-zero column
  means synthetic results may differ slightly from the last recorded run until `make run` is repeated.
- 2026-10-26 — M7a: AMLSim files uploaded by the user. Full run on 373,744 alerts (features 33 s, training ~29 min).
  Recall 98.5% (98–99), FP reduction 78.9%, backtest 95.5–98.8% over 5 periods, seeds sd 0.004. GW adds nothing over
  degrees across folds (0/5); GWd +0.5 pts (4/5). SMOTE/CTGAN no gain. Held-out pattern: cycle 97→79%, fan-in
  100→86% (D15). Isotonic calibration collapsed to 0 with 75 validation positives → Platt when <200 (D14).
  Console: small probabilities now print with two significant digits; zero-counterparty reasons reworded.
- 2026-10-26 — Setup: `docs/SETUP.md` (macOS), `make doctor`, `make run-amlsim-fast`, `PYTHON=` override. Verified from a
  clean copy: setup ~6 min, doctor all OK, 16 tests pass, fast AMLSim run ~15 min (recall 98.2%), console serves it.
