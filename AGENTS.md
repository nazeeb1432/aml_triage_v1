# AGENTS.md

Rules for any coding agent working in this repo (Claude Code, Codex, Cursor, Copilot agents) and for the
humans who direct them. Claude-specific notes are in `CLAUDE.md`.

## Roles

Work is split into roles so that the agent that writes a change is never the only one that checks it.
Each role has a prompt file in `.claude/agents/`; other tools can paste the same file as a system prompt.

| Role | File | Does | Never does |
|---|---|---|---|
| ML engineer | `ml-engineer.md` | features, labels, training, evaluation code | edits UI; changes metrics definitions to make a number look better |
| UI engineer | `ui-engineer.md` | `ui/`, API response shapes for the UI | exposes labels or BTL flags to the analyst views |
| Leakage auditor | `leakage-auditor.md` | reads diffs touching data/features/labels/splits, hunts look-ahead | writes features |
| Reviewer | `reviewer.md` | runs `docs/REVIEWER.md` against a diff, blocks or approves | fixes the code it reviews |

## Hand-off protocol

1. Pick the next unchecked item in `docs/PROGRESS.md` (or the one you were given). Note it in the log as started.
2. Read its acceptance criteria in `docs/BUILD_PLAN.md`. If there are none, write them first.
3. Make the change on a branch. Keep diffs small: one milestone item per PR.
4. Run `make lint test`. For ML changes also `make run-fast` and paste the ablation table diff into the PR.
5. Leakage auditor reviews anything under `ingest/ labels/ features/ training/ augment/`.
6. Reviewer approves against `docs/REVIEWER.md`. Then tick the item and add a log line in `docs/PROGRESS.md`.

## Definition of done

- Tests pass, including `tests/test_point_in_time.py`.
- New behaviour has a test; new features are covered by the point-in-time test automatically (prefix families).
- Config values live in `configs/`, docs updated where the behaviour is described.
- For metric claims: numbers come from a run in `reports/`, quoted with 95% CIs and the run id.

## Things agents get wrong in this domain

- Using the row's own label or any future label in a feature (classic: a "known bad" set built from the full year).
- Shuffled splits, or tuning on test.
- Oversampling before splitting, so synthetic copies of test-period positives leak into training.
- Reporting AUC instead of recall at the review budget the team can actually afford.
- Treating "closed by L1" as a clean negative; treating a GAN as a source of new laundering knowledge.
- Showing analysts which alerts are BTL samples.
