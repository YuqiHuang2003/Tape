# Main results

Scores use acc_norm on the complete official test splits. Avg is the equal-weight mean of the two tasks. Granite 50% uses the latest rerun.

## ERNIE

| Expert retention | Method | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---|---:|---:|---:|---:|
| 100% | Original | 78.24 | 46.20 | 62.22 | 100.00 |
| 75% | REAP | 75.80 | 41.20 | 58.50 | 94.02 |
| 75% | Frequency | 71.38 | 37.20 | 54.29 | 87.25 |
| 75% | EAN-Mean | 74.07 | 42.20 | 58.14 | 93.44 |
| 75% | REAP+Frequency | 73.23 | 38.40 | 55.82 | 89.71 |
| 75% | TAPE | 80.93 | 44.40 | 62.67 | 100.72 |
| 50% | REAP | 63.34 | 34.20 | 48.77 | 78.38 |
| 50% | Frequency | 59.30 | 29.20 | 44.25 | 71.12 |
| 50% | EAN-Mean | 67.42 | 32.60 | 50.01 | 80.38 |
| 50% | REAP+Frequency | 61.95 | 34.00 | 47.98 | 77.11 |
| 50% | TAPE | 68.52 | 33.00 | 50.76 | 81.58 |

## GRANITE

| Expert retention | Method | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---|---:|---:|---:|---:|
| 100% | Original | 67.59 | 38.00 | 52.80 | 100.00 |
| 75% | REAP | 46.89 | 30.60 | 38.74 | 73.38 |
| 75% | Frequency | 52.69 | 30.40 | 41.55 | 78.69 |
| 75% | REAP+Frequency | 53.45 | 32.80 | 43.13 | 81.68 |
| 75% | TAPE | 54.12 | 34.40 | 44.26 | 83.84 |
| 50% | REAP | 33.46 | 26.80 | 30.13 | 57.07 |
| 50% | Frequency | 28.49 | 29.00 | 28.75 | 54.45 |
| 50% | REAP+Frequency | 35.90 | 28.00 | 31.95 | 60.52 |
| 50% | TAPE | 37.67 | 27.00 | 32.33 | 61.24 |

## ERNIE 75%: ablations and search-process comparison

| Method | ARC-Easy (%) | OpenBookQA (%) | Avg (%) |
|---|---:|---:|---:|
| TAPE | 80.93 | 44.40 | 62.67 |
| w/o Layer Allocation | 77.99 | 41.60 | 59.79 |
| w/o Identity Search | 76.43 | 40.40 | 58.42 |
| w/o Evolved Identities | 76.14 | 40.00 | 58.07 |
| TAPE (0 generations) | 80.60 | 41.60 | 61.10 |
