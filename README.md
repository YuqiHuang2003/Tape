# TAPE: Task-Aware Expert Pruning under a Fixed Global Budget

TAPE jointly searches how many experts to retain in each layer and which experts to retain. It evaluates complete expert sets using task-specific cross-entropy and teacher-distribution KL losses, while preserving an exact global expert budget. Model weights remain frozen.

This repository provides the TAPE implementation, fixed inputs, search traces, final expert masks, and full-test results for ERNIE-4.5-21B-A3B-PT and Granite-3.1-1B-A400M-Base at 75% and 50% expert retention. The latest [English report](report/report.pdf) is included unchanged. Completed EvoESAP experiments are retained separately as supplementary results.

![Expert pruning under a fixed global budget](docs/figures/fig01_tape_framework.png)

The diagram is illustrative: uniform pruning retains the same count in each layer; TAPE can change both layer capacities and expert identities while keeping the total fixed.

## Main results

The main tables follow Tables 2–3 of the current report: TAPE and direct-scoring baselines.

### ERNIE

| Expert retention | Method | ARC-Easy (%) | OpenBookQA (%) | Avg (%) |
|---|---|---:|---:|---:|
| 100% | Original | 78.24 | 46.20 | 62.22 |
| 75% | REAP | 75.80 | 41.20 | 58.50 |
| 75% | Frequency | 71.38 | 37.20 | 54.29 |
| 75% | EAN-Mean | 74.07 | 42.20 | 58.14 |
| 75% | REAP+Frequency | 73.23 | 38.40 | 55.82 |
| 75% | TAPE | 80.86 ± 0.28 | 43.67 ± 0.70 | 62.27 ± 0.35 |
| 50% | REAP | 63.34 | 34.20 | 48.77 |
| 50% | Frequency | 59.30 | 29.20 | 44.25 |
| 50% | EAN-Mean | 67.42 | 32.60 | 50.01 |
| 50% | REAP+Frequency | 61.95 | 34.00 | 47.98 |
| 50% | TAPE | 68.76 ± 0.89 | 33.60 ± 0.72 | 51.18 ± 0.77 |

### Granite

| Expert retention | Method | ARC-Easy (%) | OpenBookQA (%) | Avg (%) |
|---|---|---:|---:|---:|
| 100% | Original | 67.59 | 38.00 | 52.80 |
| 75% | REAP | 46.89 | 30.60 | 38.74 |
| 75% | Frequency | 52.69 | 30.40 | 41.55 |
| 75% | REAP+Frequency | 53.45 | 32.80 | 43.13 |
| 75% | TAPE | 54.83 ± 0.69 | 33.47 ± 0.90 | 44.15 ± 0.38 |
| 50% | REAP | 33.46 | 26.80 | 30.13 |
| 50% | Frequency | 28.49 | 29.00 | 28.75 |
| 50% | REAP+Frequency | 35.90 | 28.00 | 31.95 |
| 50% | TAPE | 37.29 ± 0.77 | 27.07 ± 0.12 | 32.18 ± 0.42 |

TAPE values are mean ± sample standard deviation over seeds 6707–6709 in all four settings. Deterministic baselines are evaluated once. Scores use `acc_norm` on the complete official test splits, and Avg weights the two tasks equally. Expert retention refers to routed experts, not total model parameters.

[All results](docs/RESULTS.md) includes the deterministic baselines, individual seeds, and single-run analyses. The additional Granite 50% TAPE seed 6710 is retained separately: the requested 6708–6710 subset has Avg 32.24 ± 0.32 and all four seeds have Avg 32.10 ± 0.37. [Report provenance](docs/REPORT_PROVENANCE.md) records the PDF checksum and the remaining differences between its wording and the experiment records.

## Supplementary EvoESAP experiments

The current report cites EvoESAP as related work but does not include these completed experiments in its main result tables. All 12 runs, their implementation, and their evidence remain available under [`artifacts/evoesap/`](artifacts/evoesap/), with [reproduction commands](docs/REPRODUCIBILITY.md#5-supplementary-evoesap-experiments) and [summary statistics](artifacts/evoesap/summary.json).

| Model | Expert retention | EvoESAP ARC-Easy (%) | EvoESAP OpenBookQA (%) | EvoESAP Avg (%) |
|---|---:|---:|---:|---:|
| ERNIE | 75% | 76.92 ± 0.79 | 42.13 ± 0.50 | 59.53 ± 0.33 |
| ERNIE | 50% | 61.83 ± 0.70 | 34.47 ± 0.90 | 48.15 ± 0.58 |
| Granite | 75% | 52.54 ± 0.38 | 31.80 ± 0.53 | 42.17 ± 0.27 |
| Granite | 50% | 36.15 ± 0.68 | 27.20 ± 0.53 | 31.68 ± 0.58 |

These values are mean ± sample standard deviation over seeds 6707–6709. TAPE has a higher mean Avg than this budget-matched EvoESAP adaptation in all four settings. EvoESAP keeps a fixed within-layer REAP ranking; TAPE can change both capacities and expert identities. Their objectives and initialization also differ, so this comparison does not isolate the effect of the search space alone.

## Quick verification: no GPU required

Clone the repository and run:

```bash
git clone https://github.com/YuqiHuang2003/Tape.git
cd Tape
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/reproduce_tape.py check
python scripts/reproduce_evoesap.py check
python -m pytest
python scripts/reproduce_tape.py replay --case all
python scripts/reproduce_evoesap.py replay --case all
python scripts/reproduce_tape.py statistics
python scripts/reproduce_evoesap.py statistics
python scripts/reproduce_tape.py tables
```

`check` verifies the release files and expert budgets. The replay commands regenerate the 15 TAPE cases and 12 EvoESAP cases using their recorded training fitness, checking all 256 candidates and the training-based final selection in each run. CPU replay does not load a model or remeasure accuracy. The statistics commands aggregate the released per-seed full-test scores.

## GPU reproduction

All experiments used one NVIDIA A100-SXM4-80GB. TAPE uses two Granite replicas on the same GPU and one ERNIE replica; EvoESAP uses one frozen model with teacher probabilities cached on the CPU. The model families require separate environments with different pinned Transformers versions.

See the [reproduction guide](docs/REPRODUCIBILITY.md) for installation, model preparation, full search, and individual baseline evaluation. Hardware, runtime, memory, and timing definitions are reported separately in [compute costs](docs/COMPUTE_COSTS.md).

Integrity checks, unit tests, CPU replays, and statistical reconstruction are described in [release validation](docs/VALIDATION.md). This release update does not run new GPU experiments.

## Paper figures

The two approved figures are the pruning illustration above and the [matched-capacity expert-selection comparison](docs/figures/fig04_expert_selection.png). Rebuild their PDF, SVG, and PNG files without model execution:

```bash
python3.10 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r requirements-figures.txt
.venv-figures/bin/python scripts/generate_paper_figures.py --package
```

See [figure files and captions](docs/figures/README.md). The expert-selection source is the portable table `artifacts/expert_analysis/expert_status.csv`.

## Paper evidence

| Paper table | Released evidence |
|---|---|
| Table 1: expert counts and bounds | Per-case `config.json` files under [`artifacts/cases/`](artifacts/cases/), with model revisions, shapes, global budgets, and layer bounds. |
| Tables 2–3: main results | [`docs/RESULTS.md`](docs/RESULTS.md), [`artifacts/multiseed_summary.json`](artifacts/multiseed_summary.json), and the TAPE/direct-baseline evaluation files under [`artifacts/cases/`](artifacts/cases/). |
| Tables 4–5: ablations and evolution | The two restricted-search cases under [`artifacts/cases/`](artifacts/cases/) and [`artifacts/ernie75_diagnostics.json`](artifacts/ernie75_diagnostics.json); repeat counts are clarified in [`REPORT_PROVENANCE.md`](docs/REPORT_PROVENANCE.md). |
| Table 6: expert associations | [`artifacts/expert_analysis.json`](artifacts/expert_analysis.json) and [`expert_status.csv`](artifacts/expert_analysis/expert_status.csv). |
| Table 7: compute cost | [`docs/COMPUTE_COSTS.md`](docs/COMPUTE_COSTS.md) and its [measurement records](docs/COMPUTE_COSTS.json). |

## Repository layout

- `scripts/reproduce_tape.py`: verification, CPU replay, statistics, GPU search, and full-test evaluation.
- `scripts/reproduce_evoesap.py`: the corresponding commands for the supplementary EvoESAP experiments.
- `scripts/generate_paper_figures.py`: model-free generation of the two paper figures.
- `src/dsai6707/`: budget-preserving operators, NSGA-II, task objectives, expert-mask controllers, and evaluation utilities.
- `artifacts/cases/`: fixed inputs, candidate masks, training-set selection records, and compact test summaries for 15 cases: 13 main-search runs and two single-run ablations.
- `artifacts/evoesap/`: the supplementary EvoESAP protocol and all 12 runs: four settings, each with seeds 6707–6709.
- `artifacts/expert_analysis/`: source expert-selection table for the matched-capacity comparison.
- `vendor/ernie/`: the ERNIE compatibility sources used in the experiments, with upstream copyright and license notices.
- `report/report.pdf`: the latest author-supplied English report, unchanged.
- `docs/`: results, reproducibility instructions, compute costs, report provenance, and figure files.
- `RELEASE_MANIFEST.json`: file sizes and SHA-256 checksums.

Model weights, download caches, dataset text, raw test prompts, credentials, obsolete manuscripts, and historical experiment reports are excluded. Shared implementation utilities are retained where needed. Historical TAPE starting points are provided as fixed masks, so reproducing these searches does not require rerunning the earlier experiments.

## Repository and license

Repository: [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape).

The project retains the Apache-2.0 license declared in its original package metadata. Third-party models, datasets, and source files retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
