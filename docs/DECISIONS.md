# Decisions

Short records of choices that someone will later ask "why?" about. Newest last.

### D1 — New repo instead of patching the replication
The replication's structure (constants in modules, a single experiment script, pickled models) made the leakage fixes
touch everything. Kept: rolling profiles, sliding-window idea, deterministic walks, temporal split, recall@FPR.

### D2 — IBM file format as the only ingest format
Synthetic data is written in IBM's format so the real reader is exercised on every run and switching data is a config change.

### D3 — Simulated, label-dependent decision delay
Positives median 14 days, negatives 3 (synthetic). Training drops rows decided after the split boundary. This makes
GW/GWd and training realistic and reproduces the bias that recent positives are missing.

### D4 — Threshold frozen on validation, decisions on the raw score
Isotonic calibration is a step function; thresholding calibrated p creates ties and jumpy workload. Analysts see calibrated p.

### D5 — Hub removal for walks (top 1% degree)
Without it, ~94% of walks from any account reached a confirmed account and GuiltyWalker carried almost no signal
(test AUC of hit rate 0.55 → 0.69 with the cap). See ARCHITECTURE, graph features.

### D6 — Partial AUC (FPR ≤ 0.3) as tuning objective
Recall at exactly 20% FPR on a validation set with ~80 positives is a step function; partial AUC is smooth and focused on the operating region.

### D7 — Champion chosen on validation, even when test disagrees
Picking by test would make the reported test number optimistic. If test and validation disagree, that is reported, not acted on.

### D8 — Augmentation judged as an oversampler
CTGAN cannot add knowledge absent from training data. It stays in the codebase as an experiment arm and is promoted only if
it beats class weights outside CI overlap on real held-out alerts.

### D9 — SQLite audit log and file registry for now
Enough for a demo and single analyst; M9 swaps in Postgres and MLflow behind the same functions.

### D10 — Bounded label memory for GuiltyWalker; capped account age
Found through the console's own drift table (first full run): `gwd_hit_rate` PSI 1.96 and `p_account_age_days` PSI 8.1
between training and the live period. The known-bad set only ever grew, so walk hit rates rose month by month; and in the
synthetic world account age is just the calendar day. Both are time trends the model would learn and then mis-apply.
Fix: walk targets are accounts confirmed in the last `illicit_memory_days` (60); account age is capped at the longest
profile window (30), which keeps the "new account" signal. Lesson for real data: run the PSI table before trusting a model.

### D11 — Walk memory 21 days; laundering schemes allowed to run to the end of the period
With a 60-day memory the GW hit rate kept rising until ~day 70 (the label store starts empty, so the known-bad set
"warms up"), and training covered the ramp while test sat at steady state: PSI 0.28–0.33. Measured on the test period:
60 days → PSI 0.28, hit-rate AUC 0.71; 30 → 0.10, 0.65; 21 → 0.07, 0.68. 21 days keeps most of the signal and is stable.
On real data, where history exists before the first training day, re-check this with the PSI table: a longer memory may be fine.
The generator also stopped starting schemes 14 days before the end, which had left the last test days almost free of
laundering (an artificial prior shift); schemes now start up to 4 days before the end and are cut off at the boundary.

### D12 — Choose the champion on pre-test rolling folds, not on the validation split alone (refines D6/D7)
The first robustness run showed the validation-chosen model (`+degree`, chosen on 73 validation positives) losing about
3 recall points to `+gw` in every one of 4 backtest folds. One small validation split is too noisy to pick between
variants that differ by a few points. Selection now uses the mean recall at the frozen threshold over the rolling folds
whose test windows end before the real test period (days 75–94 and 95–114: ~400 positives); validation pAUC only breaks
ties. Test still never influences a choice. CTGAN is not refit per fold by default (minutes per fold), so it can't become
champion unless `augmentation.select_ctgan_in_folds: true`.

### D13 — Parsimony margin for selection; a confidence range on the below-the-line recall estimate
After D12 the fold scores of the augmentation arms were 0.893 (class weights), 0.895 (SMOTE) and 0.896 (no weighting):
the champion was being decided by differences a third the size of the seed-to-seed spread (sd ≈ 0.011). Now a richer
feature set, or an augmentation arm, must beat the simpler choice by `training.selection_min_gain` (0.01) on the
pre-test folds. The same run showed the below-the-line estimate at 86% against a hindsight 94%: the high stratum
sampled 895 alerts and found 5 suspicious ones, so each find moves the estimate a lot. The estimate now carries a
bootstrap 95% range. Operational reading: if the range is wide, raise the sample rate in the stratum just below the line.

### D14 — Platt calibration when validation has few positives
On AMLSim, label delay leaves only 75 decided positives in validation (days 140–159; positives take ~14 days to
decide). Isotonic regression then mapped the whole lower score range, including the review line, to a probability of
exactly 0, so the console showed "0%" scores above the line. Decisions were unaffected (they use the raw-score
threshold). `training.calibration: auto` now uses sigmoid (Platt) calibration below `isotonic_min_positives` (200) and
isotonic above it. The run records which one it used (`champion.calibration`).

### D15 — AMLSim results are an upper bound; the held-out-pattern test is the number to quote
In AMLSim every laundering transfer is under $25 and legitimate ones rarely are, and there are only two typologies.
The strongest features are minimum transfer sizes, so the model partly learns the simulator's amount signature.
Removing one typology from training drops recall on it from 97% to 78% (cycle) and from 100% to 86% (fan-in).
That gap, not the 98.5% headline, is the realistic expectation for a scheme the model has not seen before.
