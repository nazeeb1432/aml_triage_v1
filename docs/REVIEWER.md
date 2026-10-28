# Reviewer checklist

Run this against every diff. Block on any unchecked item in sections 1–3. The reviewer comments; the author fixes.

## 1. Time and leakage (block)
- [ ] No feature reads labels except through `LabelIndex` / `illicit_as_of` with the alert's own day.
- [ ] No rolling window is `closed="both"`/forward-looking; windows end on the alert day.
- [ ] No new code calls `table["label"]` or `is_laundering` outside `labels/`, `training/`, `evaluation/`.
- [ ] New feature families have a registered prefix in `features/build.FAMILIES` (so the point-in-time test covers them).
- [ ] `tests/test_point_in_time.py` passes.

## 2. Evaluation honesty (block)
- [ ] Nothing is selected on test (features, params, τ, champion, augmentation arm).
- [ ] Augmentation receives only training rows; synthetic rows are flagged and excluded from metrics.
- [ ] Claimed improvements quote CIs and the run id; overlapping CIs are described as no detectable difference.
- [ ] Metric definitions in `evaluation/metrics.py` unchanged, or the change is explained in `docs/EVALUATION.md`.

## 3. Safety and governance (block)
- [ ] Analyst endpoints (`/api/alerts*`) expose no `label`, `typology`, `decision_reason`, or BTL flag.
- [ ] Policy changes alter `policy_version`; overrides still win over score.
- [ ] Every disposition records model run id and policy version.
- [ ] Model artifacts carry data fingerprint, git sha, features and τ in `manifest.json`.

## 4. Engineering
- [ ] New numbers live in `configs/` and `config.py` with types.
- [ ] Tests for new behaviour; `make lint test` green.
- [ ] Docs updated: ARCHITECTURE (if structure changed), PROGRESS log line.

## 5. UI (for `ui/` diffs)
- [ ] Only tokens from `tokens.css`; no raw hex in components.
- [ ] Follows `docs/UI_RULES.md` (copy, states, accessibility). Checked at 1280px and 390px.
