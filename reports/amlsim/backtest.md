# Robustness — amlsim-20261026T121506Z

Model: alert+profiles+degree+gwd. Champion hyperparameters reused in every fold (not re-tuned).

## Rolling-origin backtest

| Test days | Train positives | Test positives | Recall (95% CI) | FP reduction |
|---|---|---|---|---|
| 100–119 | 653 | 242 | 0.967 (0.95–0.99) | 81.6% |
| 120–139 | 995 | 310 | 0.971 (0.96–0.99) | 82.6% |
| 140–159 | 1303 | 337 | 0.985 (0.97–1.00) | 80.6% |
| 160–179 | 1567 | 394 | 0.985 (0.98–0.99) | 80.2% |
| 180–199 | 1863 | 340 | 0.994 (0.98–1.00) | 78.4% |

## Feature families across folds

| Feature set | Mean recall | SD | Folds better than previous rung | Mean gain |
|---|---|---|---|---|
| alert | 0.825 | 0.035 | — | — |
| alert+profiles | 0.927 | 0.023 | 5/5 | +0.102 |
| alert+profiles+degree | 0.968 | 0.018 | 5/5 | +0.041 |
| alert+profiles+degree+gw | 0.969 | 0.016 | 2/5 | +0.001 |
| alert+profiles+degree+gwd | 0.979 | 0.009 | 4/5 | +0.010 |

## Held-out laundering pattern

Training positives of the pattern removed; recall on that pattern in test.

| Pattern | Test positives | Removed from training | Recall when seen | Recall when never seen |
|---|---|---|---|---|
| CYCLE | 369 | 868 | 0.97 | 0.72 |
| FAN-IN | 365 | 699 | 1.00 | 0.81 |

## Seeds

Recall across 5 seeds: 0.985–0.986 (sd 0.001). Differences between variants smaller than this are noise.
