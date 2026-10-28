---
name: add-feature
description: Add a new feature or feature family to the alert triage model in a point-in-time-safe way and measure whether it helps. Use when asked to add, try, or engineer a feature.
---

# Add a feature

1. Decide the family. Existing prefixes: `a_` alert/rule, `p_` profile, `g_` degree, `gw_` GuiltyWalker, `gwd_` GW-delay.
   A new family needs a new prefix registered in `features/build.FAMILIES` and a new entry in
   `configs/default.yaml: evaluation.feature_sets`.
2. Implement it where its inputs live (`features/profiles.py` for per-account aggregates, `features/graph.py` for snapshot
   computations). Inputs: transactions with day ≤ d; labels only through `LabelIndex(...).*_as_of(d)`.
3. Put every parameter in `configs/default.yaml` and `config.py`.
4. Add a reason-code template in `serving/reasons.py:describe` so analysts get a sentence, not a column name.
5. Run `pytest tests/test_point_in_time.py` — it covers the new columns automatically through the prefix.
6. Run `make run-fast` for a smoke check, then `make run` and compare the ladder in `reports/<run>/ablation.md`.
   Keep the feature only if the new rung's recall CI lower bound is above the previous rung's point estimate, or if it
   helps a typology/rule that was weak.
7. Update `docs/ARCHITECTURE.md` (feature family table) and `docs/PROGRESS.md`.
