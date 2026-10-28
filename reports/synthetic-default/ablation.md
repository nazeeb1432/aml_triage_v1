# Ablation — synthetic-default-20261025T104314Z

Test period alerts: 22,275 (329 suspicious). Threshold frozen on validation at 20% FPR.

## Feature-set ladder (class-weighted LightGBM)

Selection recall = mean recall at the frozen threshold over rolling folds that end before the test period;
the champion is picked on it (validation pAUC breaks ties). Test columns are reported only.

| Feature set | Features (selected/total) | Selection recall | Val pAUC | Test recall @ frozen τ (95% CI) | FP reduction (95% CI) | PR-AUC | Oracle recall@20%FPR |
|---|---|---|---|---|---|---|---|
| alert | 11/19 | 0.814 | 0.829 | 0.818 (0.78–0.85) | 80.4% (79.9%–80.9%) | 0.498 | 0.818 |
| alert+profiles | 75/120 | 0.847 | 0.862 | 0.860 (0.82–0.90) | 80.1% (79.3%–80.9%) | 0.628 | 0.860 |
| alert+profiles+degree | 77/133 | 0.876 | 0.915 | 0.891 (0.86–0.92) | 78.8% (78.0%–79.8%) | 0.693 | 0.891 |
| alert+profiles+degree+gw | 90/143 | 0.893 | 0.907 | 0.933 (0.91–0.96) | 78.7% (78.0%–79.5%) | 0.717 | 0.930 |
| alert+profiles+degree+gwd | 68/143 | 0.891 | 0.910 | 0.939 (0.92–0.96) | 79.2% (78.7%–79.8%) | 0.721 | 0.933 |

## Augmentation arms (training set only)

| Arm | Synthetic rows | Real-vs-synthetic AUC | Selection recall | Val pAUC | Test recall (95% CI) | FP reduction | PR-AUC |
|---|---|---|---|---|---|---|---|
| none | 0 | — | 0.896 | 0.908 | 0.915 (0.89–0.94) | 78.8% | 0.709 |
| class_weight | 0 | — | 0.893 | 0.907 | 0.933 (0.91–0.96) | 78.7% | 0.717 |
| smote | 6,372 | 0.992 | 0.895 | 0.905 | 0.921 (0.90–0.95) | 76.9% | 0.653 |
| ctgan | 6,372 | 1.0 | not run | 0.901 | 0.903 (0.88–0.93) | 79.1% | 0.640 |

**Champion (chosen on pre-test folds):** alert+profiles+degree+gw / class_weight — test recall 0.933 (0.91–0.96), FP reduction 78.7%.
