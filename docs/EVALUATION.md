# Evaluation protocol

This is the contract for every number this project reports. If a result was produced some other way, it is not comparable.

## Splits (by alert day, never shuffled)

| Split | Days (default synthetic) | Used for | Labels |
|---|---|---|---|
| warm-up | 0–29 | stage-1 pseudo-labeller for GWd only | decided before day 30 |
| train | 30–94 | fitting | only rows decided before day 95 |
| validation | 95–114 | tuning, feature selection, calibration, τ, tie-break for champion | only rows decided before day 115 |
| test | 115–149 | reporting only | hindsight (all) |

Dropping train/val rows whose decision came later is deliberate: a model trained on day 95 cannot know them.
Because positives take longer to decide, this also reproduces the real bias that recent positives are under-represented.

## Headline metric

**Recall at the frozen threshold**: τ is set on validation so that 20% of validation false alerts would be reviewed
(the paper's operating point), then applied unchanged to test. We report recall and false-positive reduction at that τ.
This is what the bank would actually experience, unlike "recall at 20% FPR on test", which re-fits the threshold to
test (reported too, as `oracle`).

Secondary: PR-AUC (informative at ~1% prevalence), Brier score and a reliability curve, F2 (Bakry et al. use F-beta),
per-rule and per-typology recall, PSI of top features.

## Uncertainty

95% CIs by **day-block bootstrap** (1,000 resamples of whole test days). Alerts on the same day share accounts and
laundering schemes, so resampling single alerts would overstate confidence. A difference between two variants is a
finding only if it holds outside the overlap of their intervals; otherwise write "no detectable difference".

## Champion selection

Mean recall at the frozen threshold over the rolling folds that end before the test period (days 75–94 and 95–114,
each trained only on labels decided before its own cut). Validation pAUC breaks ties. A richer feature set or an
augmentation arm is chosen only if it beats the simpler option by more than `selection_min_gain` (0.01, about the
seed-to-seed spread). See DECISIONS D12–D13.

## The ladder (does the paper replicate?)

`alert` → `+profiles` → `+degree` → `+gw` → `+gwd`, each tuned independently with the same budget.

## Augmentation arms (does a GAN help?)

On the best ladder feature set (by validation): `none` (no weighting), `class_weight` (tuned), `smote`, `ctgan`.
Rules:
1. Fit only on training rows; NaN replaced by a sentinel for all arms so they are comparable.
2. Synthetic rows bring positives to 15% of the training set; they are flagged and never evaluated.
3. Report `fidelity_auc_real_vs_synth`: a 5-fold classifier separating real from synthetic positives (0.5 = indistinguishable).
4. If an arm's test CI overlaps class weights, the conclusion is "no evidence it helps", whatever the point estimate.

What the GAN cannot do: it learns the distribution of feature vectors it is shown. It cannot create laundering behaviour
absent from training data, and its rows have no transactions or graph behind them. It is an oversampler, judged as one.

## Robustness (stage `backtest`, `reports/<run>/backtest.md`)

- **Rolling origin**: the full train → calibrate → freeze τ → test procedure at 4 cut dates (days 75, 95, 115, 130), each
  with only the labels decided by then. Hyperparameters are the champion's (stated in the report).
- **Families across folds**: a feature family "helps" if it beats the rung below in most folds, not just the headline one.
- **Held-out pattern**: remove every training positive of one typology, test on that typology. Large drops mean memorisation.
- **Seeds**: 5 refits with different seeds; differences smaller than this spread are noise.

## Production estimate of recall (BTL)

After launch, suppressed alerts produce no labels, so recall cannot be measured directly. `policy.btl_recall_estimate`
estimates it from the stratified below-the-line sample; the run reports this estimate next to the hindsight truth.

## Acceptance thresholds for promotion (proposed; agree with compliance)

- Test recall lower CI bound ≥ 0.90 at the frozen τ.
- FP reduction ≥ 60%.
- No typology with ≥ 20 positive alerts below 0.80 recall.
- No top-15 feature with PSI > 0.25 between train and the most recent month.
- `test_point_in_time.py` green on the exact commit.
