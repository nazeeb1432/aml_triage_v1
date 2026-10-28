# Ablation — amlsim-20261026T121506Z

Test period alerts: 82,759 (734 suspicious). Threshold frozen on validation at 20% FPR.

## Feature-set ladder (class-weighted LightGBM)

Selection recall = mean recall at the frozen threshold over rolling folds that end before the test period;
the champion is picked on it (validation pAUC breaks ties). Test columns are reported only.

| Feature set | Features (selected/total) | Selection recall | Val pAUC | Test recall @ frozen τ (95% CI) | FP reduction (95% CI) | PR-AUC | Oracle recall@20%FPR |
|---|---|---|---|---|---|---|---|
| alert | 7/21 | 0.772 | 0.857 | 0.813 (0.79–0.83) | 80.3% (80.0%–80.7%) | 0.622 | 0.813 |
| alert+profiles | 18/122 | 0.926 | 0.942 | 0.941 (0.93–0.96) | 80.1% (79.6%–80.5%) | 0.846 | 0.941 |
| alert+profiles+degree | 43/135 | 0.951 | 0.980 | 0.970 (0.96–0.98) | 82.5% (82.1%–82.9%) | 0.888 | 0.973 |
| alert+profiles+degree+gw | 54/145 | 0.960 | 0.985 | 0.971 (0.96–0.98) | 81.5% (81.1%–81.8%) | 0.893 | 0.971 |
| alert+profiles+degree+gwd | 61/145 | 0.974 | 0.986 | 0.986 (0.98–0.99) | 79.3% (78.8%–79.8%) | 0.908 | 0.986 |

## Augmentation arms (training set only)

| Arm | Synthetic rows | Real-vs-synthetic AUC | Selection recall | Val pAUC | Test recall (95% CI) | FP reduction | PR-AUC |
|---|---|---|---|---|---|---|---|
| none | 0 | — | 0.976 | 0.988 | 0.989 (0.98–1.00) | 78.2% | 0.900 |
| class_weight | 0 | — | 0.974 | 0.986 | 0.986 (0.98–0.99) | 79.3% | 0.908 |
| smote | 31,773 | 0.997 | 0.966 | 0.987 | 0.984 (0.98–0.99) | 77.8% | 0.876 |

**Champion (chosen on pre-test folds):** alert+profiles+degree+gwd / class_weight — test recall 0.986 (0.98–0.99), FP reduction 79.3%.
