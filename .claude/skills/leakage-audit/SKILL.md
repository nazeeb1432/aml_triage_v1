---
name: leakage-audit
description: Audit code or a diff for look-ahead and label leakage in AML features, labels, splits, or augmentation. Use before merging any change under ingest/, labels/, features/, training/ or augment/, or when results look too good.
---

# Leakage audit

Work through each check and report pass/fail with file:line.

1. **Label access.** `grep -n "label\|is_laundering" src/aml_triage/features src/aml_triage/rules`. Features may read labels only
   via `LabelIndex` with the alert's own day. Anything else fails.
2. **Windows.** Every rolling window must end on the alert day (`closed="right"` with the day as index, or `closed="left"` for
   strictly-past). Look for `shift(-`, forward fills across days, `max()` over the full table.
3. **Global statistics.** Quantiles, means, encoders, scalers or hub caps computed over all days and reused per day leak. They must
   be computed per snapshot or on training data only.
4. **Splits.** Shuffling, `train_test_split`, or tuning/selection that reads `split == "test"`.
5. **Augmentation.** `augment()` must receive only train rows; synthetic rows must never reach `_evaluate`.
6. **Pseudo-labels (GWd).** Stage-1 is trained on warm-up only; warm-up alerts are never pseudo-labelled.
7. **Run the test.** `pytest tests/test_point_in_time.py -q`. If a new feature family was added, confirm its prefix is in
   `features/build.FAMILIES` so the test covers it.
8. **Sniff test.** If a change improves test recall by more than its CI width with no clear mechanism, assume leakage and look again.

Report as a table: check, result, evidence, fix.
