---
name: evaluate-change
description: Compare two pipeline runs (before/after a change) and write an honest verdict with confidence intervals. Use after any modelling change, or when asked whether something "improved" the model.
---

# Evaluate a change

1. Make sure both runs used the same data fingerprint (`metrics.json → data.sha256_16`) and the same split config. If not, say the
   comparison is invalid and stop.
2. For each run collect: champion recall at frozen τ with CI, FP reduction with CI, PR-AUC, Brier, per-typology recall,
   BTL-estimated recall.
3. Verdict rules (from `docs/EVALUATION.md`):
   - "Better" only if the new recall CI lower bound > old point estimate *and* FP reduction did not drop by more than 2 points.
   - Overlapping CIs → "no detectable difference".
   - Any typology with ≥20 positives dropping >10 points → flag as regression even if the headline improved.
4. Write the comparison as a short table plus 2–3 sentences, and append a log line to `docs/PROGRESS.md` with both run ids.
