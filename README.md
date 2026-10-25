# TAPE: Task-Aware Expert Pruning under a Fixed Global Budget

TAPE jointly searches how many experts to retain in each layer and which experts to retain. It evaluates complete expert sets using task-specific cross-entropy and teacher-distribution KL losses, while preserving an exact global expert budget. Model weights remain frozen.

This repository provides the implementation, fixed inputs, search traces, final expert masks, and full-test results for ERNIE-4.5-21B-A3B-PT and Granite-3.1-1B-A400M-Base at 75% and 50% expert retention.

## Main results

| Model | Expert retention | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---:|---:|---:|---:|---:|
| ERNIE | 75% | 80.93 | 44.40 | 62.67 | 100.72 |
| ERNIE | 50% | 68.52 | 33.00 | 50.76 | 81.58 |
| Granite | 75% | 54.12 | 34.40 | 44.26 | 83.84 |
| Granite | 50% | 37.67 | 27.00 | 32.33 | 61.24 |

Scores are `acc_norm` on the complete official test splits. Expert retention refers to routed experts, not total model parameters. See [all results](docs/RESULTS.md) for baselines, ablations, and the search-process comparison.

## Quick verification: no GPU required

Clone the repository and run:

```bash
git clone https://github.com/YuqiHuang2003/Tape.git
cd Tape
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/reproduce_tape.py check
python -m pytest
python scripts/reproduce_tape.py replay --case all
python scripts/reproduce_tape.py tables
```

`check` verifies the release files and expert budgets. `replay` re-executes NSGA-II using the recorded training objectives and checks all 256 candidates and the final selection. CPU replay verifies search logic; it does not rerun a model or remeasure accuracy.

## GPU reproduction

All experiments used one NVIDIA A100-SXM4-80GB. Granite uses two model replicas on the same GPU; ERNIE uses one. The models require separate environments with different pinned Transformers versions.

See the [reproduction guide](docs/REPRODUCIBILITY.md) for installation, model preparation, full search, and individual baseline evaluation. Hardware, runtime, memory, and timing definitions are reported separately in [compute costs](docs/COMPUTE_COSTS.md).

Installation checks, unit tests, CPU replays, and the single-GPU inference check are documented in [release validation](docs/VALIDATION.md).

## Repository layout

- `scripts/reproduce_tape.py`: verification, CPU replay, GPU search, and full-test evaluation.
- `src/dsai6707/`: budget-preserving operators, NSGA-II, task objectives, expert-mask controllers, and evaluation utilities.
- `artifacts/cases/`: fixed inputs, candidate masks, training-set selection records, and compact test summaries for six experiments.
- `vendor/ernie/`: the ERNIE compatibility sources used in the experiments, with upstream copyright and license notices.
- `report/main.tex`: the English report manuscript, including equations, results, ablations, and expert-selection analysis.
- `RELEASE_MANIFEST.json`: file sizes and SHA-256 checksums.

Model weights, download caches, dataset text, raw test prompts, credentials, and historical experiment reports are excluded. Shared implementation utilities are retained where needed. Historical TAPE starting points are provided as fixed masks, so reproducing these searches does not require rerunning the earlier experiments.

## Repository and license

Repository: [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape).

The project retains the Apache-2.0 license declared in its original package metadata. Third-party models, datasets, and source files retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
