# Main results

Scores use `acc_norm` on the complete official test splits: 2,376 ARC-Easy questions and 500 OpenBookQA questions. Avg is the equal-weight mean of the two task accuracies in each run. TAPE entries report the mean ± sample standard deviation across three independent search seeds; SD is in percentage points, not a confidence interval. Baselines are fixed expert sets evaluated once, not three-seed averages.

ERNIE 75%, ERNIE 50%, and Granite 75% use search seeds 6707-6709. Granite 50% uses the latest requested summary, seeds 6708-6710. Seed 6710 was added after viewing the original results, so this is not presented as a preplanned replacement cohort. All four Granite 50% runs, the original cohort, and the all-four summary remain below and in `artifacts/multiseed_summary.json`.

## ERNIE

| Expert retention | Method | n | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---|---:|---:|---:|---:|---:|
| 100% | Original | 1 | 78.24 | 46.20 | 62.22 | 100.00 |
| 75% | REAP | 1 | 75.80 | 41.20 | 58.50 | 94.02 |
| 75% | Frequency | 1 | 71.38 | 37.20 | 54.29 | 87.25 |
| 75% | EAN-Mean | 1 | 74.07 | 42.20 | 58.14 | 93.44 |
| 75% | REAP+Frequency | 1 | 73.23 | 38.40 | 55.82 | 89.71 |
| 75% | TAPE | 3 | 80.86 ± 0.28 | 43.67 ± 0.70 | 62.27 ± 0.35 | 100.07 ± 0.56 |
| 50% | REAP | 1 | 63.34 | 34.20 | 48.77 | 78.38 |
| 50% | Frequency | 1 | 59.30 | 29.20 | 44.25 | 71.12 |
| 50% | EAN-Mean | 1 | 67.42 | 32.60 | 50.01 | 80.38 |
| 50% | REAP+Frequency | 1 | 61.95 | 34.00 | 47.98 | 77.11 |
| 50% | TAPE | 3 | 68.76 ± 0.89 | 33.60 ± 0.72 | 51.18 ± 0.77 | 82.25 ± 1.24 |

## GRANITE

| Expert retention | Method | n | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---|---:|---:|---:|---:|---:|
| 100% | Original | 1 | 67.59 | 38.00 | 52.80 | 100.00 |
| 75% | REAP | 1 | 46.89 | 30.60 | 38.74 | 73.38 |
| 75% | Frequency | 1 | 52.69 | 30.40 | 41.55 | 78.69 |
| 75% | REAP+Frequency | 1 | 53.45 | 32.80 | 43.13 | 81.68 |
| 75% | TAPE | 3 | 54.83 ± 0.69 | 33.47 ± 0.90 | 44.15 ± 0.38 | 83.62 ± 0.72 |
| 50% | REAP | 1 | 33.46 | 26.80 | 30.13 | 57.07 |
| 50% | Frequency | 1 | 28.49 | 29.00 | 28.75 | 54.45 |
| 50% | REAP+Frequency | 1 | 35.90 | 28.00 | 31.95 | 60.52 |
| 50% | TAPE | 3 | 37.74 ± 0.06 | 26.73 ± 0.64 | 32.24 ± 0.32 | 61.06 ± 0.61 |

## Individual main-experiment search seeds

Each row is a separately recorded search and its training-selected expert set's full-test evaluation. Unchanged data samples, baseline anchors, candidate budget, and training-only selection are shared across seeds. Initialization uses REAP, Frequency, REAP+Frequency, a historical feasible TAPE expert set, and local perturbations; it is not REAP-only initialization.

| Model | Expert retention | Search seed | Portable case ID | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) | In latest summary |
|---|---:|---:|---|---:|---:|---:|---:|---|
| ERNIE | 75% | 6707 | `ernie_48` | 80.93 | 44.40 | 62.67 | 100.72 | Yes |
| ERNIE | 75% | 6708 | `ernie_48_seed_6708` | 81.10 | 43.00 | 62.05 | 99.73 | Yes |
| ERNIE | 75% | 6709 | `ernie_48_seed_6709` | 80.56 | 43.60 | 62.08 | 99.77 | Yes |
| ERNIE | 50% | 6707 | `ernie_32` | 68.52 | 33.00 | 50.76 | 81.58 | Yes |
| ERNIE | 50% | 6708 | `ernie_32_seed_6708` | 68.01 | 33.40 | 50.71 | 81.50 | Yes |
| ERNIE | 50% | 6709 | `ernie_32_seed_6709` | 69.74 | 34.40 | 52.07 | 83.69 | Yes |
| GRANITE | 75% | 6707 | `granite_24` | 54.12 | 34.40 | 44.26 | 83.84 | Yes |
| GRANITE | 75% | 6708 | `granite_24_seed_6708` | 55.51 | 33.40 | 44.46 | 84.20 | Yes |
| GRANITE | 75% | 6709 | `granite_24_seed_6709` | 54.84 | 32.60 | 43.72 | 82.81 | Yes |
| GRANITE | 50% | 6707 | `granite_16_seed_6707` | 36.41 | 27.00 | 31.70 | 60.05 | No; preserved history |
| GRANITE | 50% | 6708 | `granite_16` | 37.67 | 27.00 | 32.33 | 61.24 | Yes |
| GRANITE | 50% | 6709 | `granite_16_seed_6709` | 37.79 | 27.20 | 32.50 | 61.55 | Yes |
| GRANITE | 50% | 6710 | `granite_16_seed_6710` | 37.75 | 26.00 | 31.88 | 60.38 | Yes |

## Granite 50%: complete reporting history

Granite 50% seeds 6708-6710 are the latest requested reporting cohort, not a preplanned replacement cohort. Seed 6710 was requested after observing the original results. Seed 6707 and the original 6707-6709 cohort remain released, along with the all-four-seed statistics. Each run's expert set was selected on training data before its full-test evaluation.

| Cohort | Seeds | n | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---|---:|---:|---:|---:|---:|
| Latest requested summary | 6708, 6709, 6710 | 3 | 37.74 ± 0.06 | 26.73 ± 0.64 | 32.24 ± 0.32 | 61.06 ± 0.61 |
| Original three-seed cohort | 6707, 6708, 6709 | 3 | 37.29 ± 0.77 | 27.07 ± 0.12 | 32.18 ± 0.42 | 60.95 ± 0.79 |
| All recorded searches | 6707, 6708, 6709, 6710 | 4 | 37.41 ± 0.67 | 26.80 ± 0.54 | 32.10 ± 0.37 | 60.80 ± 0.71 |

## ERNIE 75%: single-run ablations and search-process comparison

All rows in this diagnostic table use one search seed (6707), not three-seed means. The full TAPE row is its seed-6707 reference run, not the aggregate main result. The two restricted searches have the same candidate budget and training-only selection rule; the capacity-matched and initialization controls reuse fixed expert sets from that reference run.

| Method | ARC-Easy (%) | OpenBookQA (%) | Avg (%) |
|---|---:|---:|---:|
| TAPE (seed 6707) | 80.93 | 44.40 | 62.67 |
| w/o Layer Allocation | 77.99 | 41.60 | 59.79 |
| w/o Identity Search | 76.43 | 40.40 | 58.42 |
| w/o Evolved Identities | 76.14 | 40.00 | 58.07 |
| TAPE (0 generations) | 80.60 | 41.60 | 61.10 |
