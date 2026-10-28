---
name: run-pipeline
description: Run the AML triage pipeline (all stages or some) on synthetic or IBM data and summarise the results. Use when asked to run, rerun, retrain, rescore, or reproduce a result.
---

# Run the pipeline

1. Pick the data:
   - synthetic (default): nothing to prepare.
   - IBM: confirm `data/raw/HI-Small_Trans.csv` exists; if not, point the user to `docs/DATA.md` and stop.
2. Pick the budget:
   - iterating: `make run-fast` (tiny Optuna budget, 3 CTGAN epochs, 100 bootstrap reps). Numbers are for smoke-testing only.
   - reporting: `make run` (or `make run-ibm`).
3. Partial reruns: `aml-triage run --stages <stages> [--set key=value ...]`. Stages read the previous stage's files from
   `data/work/<run_name>/`, so features must be rebuilt after any change in `rules/`, `labels/` or `features/`.
4. After it finishes, read `reports/<run_name>/ablation.md` and `metrics.json` and report:
   - rules-only false-positive rate (`alert_stats.false_positive_rate_of_rules`)
   - champion feature set + augmentation, test recall with CI, FP reduction
   - the ladder deltas and whether CIs overlap
   - BTL-estimated recall vs hindsight (`policy` block)
5. Never quote numbers from a `run-fast` run as results.
