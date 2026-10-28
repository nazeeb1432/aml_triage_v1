# Robustness — synthetic-default-20261025T104314Z

Model: alert+profiles+degree+gw. Champion hyperparameters reused in every fold (not re-tuned).

## Rolling-origin backtest

| Test days | Train positives | Test positives | Recall (95% CI) | FP reduction |
|---|---|---|---|---|
| 75–94 | 69 | 164 | 0.866 (0.82–0.90) | 81.5% |
| 95–114 | 215 | 249 | 0.920 (0.89–0.95) | 79.0% |
| 115–134 | 389 | 234 | 0.927 (0.90–0.95) | 79.1% |
| 130–149 | 524 | 132 | 0.947 (0.90–0.98) | 79.1% |

## Feature families across folds

| Feature set | Mean recall | SD | Folds better than previous rung | Mean gain |
|---|---|---|---|---|
| alert | 0.818 | 0.012 | — | — |
| alert+profiles | 0.871 | 0.011 | 4/4 | +0.053 |
| alert+profiles+degree | 0.886 | 0.013 | 4/4 | +0.015 |
| alert+profiles+degree+gw | 0.910 | 0.029 | 3/4 | +0.024 |
| alert+profiles+degree+gwd | 0.913 | 0.029 | 2/4 | +0.003 |

## Held-out laundering pattern

Training positives of the pattern removed; recall on that pattern in test.

| Pattern | Test positives | Removed from training | Recall when seen | Recall when never seen |
|---|---|---|---|---|
| RANDOM | 96 | 55 | 0.95 | 0.93 |
| CYCLE | 56 | 44 | 0.96 | 0.95 |
| GATHER-SCATTER | 48 | 83 | 0.92 | 0.90 |
| BIPARTITE | 40 | 41 | 0.85 | 0.85 |
| STACK | 37 | 65 | 0.86 | 0.86 |
| FAN-IN | 27 | 52 | 1.00 | 1.00 |
| SCATTER-GATHER | 21 | 29 | 1.00 | 1.00 |

## Seeds

Recall across 5 seeds: 0.921–0.936 (sd 0.005). Differences between variants smaller than this are noise.
