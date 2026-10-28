# TAPE: Task-Aware Expert Pruning under a Fixed Global Budget

TAPE jointly searches how many experts to retain in each layer and which experts to retain. It evaluates complete expert sets using task-specific cross-entropy and teacher-distribution KL losses, while preserving an exact global expert budget. Model weights remain frozen.

This repository provides the implementation, fixed inputs, search traces, final expert masks, and full-test results for ERNIE-4.5-21B-A3B-PT and Granite-3.1-1B-A400M-Base at 75% and 50% expert retention.

![Expert pruning under a fixed global budget](docs/figures/fig01_tape_framework.png)

The diagram is illustrative: uniform pruning retains the same count in each layer; TAPE can change both layer capacities and expert identities while keeping the total fixed.

## Main results

| Model | Expert retention | ARC-Easy (%) | OpenBookQA (%) | Avg (%) | Performance retained (%) |
|---|---:|---:|---:|---:|---:|
| ERNIE | 75% | 80.86 ± 0.28 | 43.67 ± 0.70 | 62.27 ± 0.35 | 100.07 ± 0.56 |
| ERNIE | 50% | 68.76 ± 0.89 | 33.60 ± 0.72 | 51.18 ± 0.77 | 82.25 ± 1.24 |
| Granite | 75% | 54.83 ± 0.69 | 33.47 ± 0.90 | 44.15 ± 0.38 | 83.61 ± 0.72 |
| Granite | 50% | 37.74 ± 0.06 | 26.73 ± 0.64 | 32.24 ± 0.32 | 61.06 ± 0.61 |

Values are mean ± sample standard deviation over three search seeds, using `acc_norm` on the complete official test splits. Avg weights the two tasks equally. Expert retention refers to routed experts, not total model parameters.

ERNIE and Granite 75% use seeds 6707–6709. Granite 50% shows the requested 6708–6710 subset; seed 6710 was added after the original results were inspected. The original 6707–6709 cohort and all-four-seed statistics remain in [all results](docs/RESULTS.md), together with per-seed scores, baselines, and the two single-run search ablations. The original Granite 50% cohort has Avg 32.18 ± 0.42; all four seeds have Avg 32.10 ± 0.37.

The latest [English report](report/report.pdf) is published unchanged. [Report provenance](docs/REPORT_PROVENANCE.md) records its version and differences from the experiment records, including its Granite 50% cohort and ablation repeat counts.

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
python scripts/reproduce_tape.py statistics
python scripts/reproduce_tape.py tables
```

`check` verifies the release files and expert budgets. `replay` re-executes NSGA-II using the recorded training objectives and checks all 256 candidates and the final selection in each of the 15 cases. CPU replay verifies search logic; it does not rerun a model or remeasure accuracy. `statistics` aggregates the released per-seed full-test scores.

## GPU reproduction

All experiments used one NVIDIA A100-SXM4-80GB. Granite uses two model replicas on the same GPU; ERNIE uses one. The models require separate environments with different pinned Transformers versions.

See the [reproduction guide](docs/REPRODUCIBILITY.md) for installation, model preparation, full search, and individual baseline evaluation. Hardware, runtime, memory, and timing definitions are reported separately in [compute costs](docs/COMPUTE_COSTS.md).

Installation checks, unit tests, CPU replays, and the single-GPU inference check are documented in [release validation](docs/VALIDATION.md).

## Paper figures

The two approved figures are the pruning illustration above and the [matched-capacity expert-selection comparison](docs/figures/fig04_expert_selection.png). Rebuild their PDF, SVG, and PNG files without model execution:

```bash
python3.10 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r requirements-figures.txt
.venv-figures/bin/python scripts/generate_paper_figures.py --package
```

See [figure files and captions](docs/figures/README.md) and the [figure design specification](docs/FIGURE_DESIGN_PLAN.md). The expert-selection source is the portable table `artifacts/expert_analysis/expert_status.csv`.

## Repository layout

- `scripts/reproduce_tape.py`: verification, CPU replay, statistics, GPU search, and full-test evaluation.
- `scripts/generate_paper_figures.py`: model-free generation of the two paper figures.
- `src/dsai6707/`: budget-preserving operators, NSGA-II, task objectives, expert-mask controllers, and evaluation utilities.
- `artifacts/cases/`: fixed inputs, candidate masks, training-set selection records, and compact test summaries for 15 cases: 13 main-search runs and two single-run ablations.
- `artifacts/expert_analysis/`: source expert-selection table for the matched-capacity comparison.
- `vendor/ernie/`: the ERNIE compatibility sources used in the experiments, with upstream copyright and license notices.
- `report/report.pdf`: the latest author-supplied English report, unchanged.
- `report/legacy/main.tex`: the earlier English manuscript, not the source of the current PDF.
- `docs/`: results, reproducibility instructions, compute costs, report provenance, and figure files.
- `RELEASE_MANIFEST.json`: file sizes and SHA-256 checksums.

Model weights, download caches, dataset text, raw test prompts, credentials, and historical experiment reports are excluded. Shared implementation utilities are retained where needed. Historical TAPE starting points are provided as fixed masks, so reproducing these searches does not require rerunning the earlier experiments.

## Repository and license

Repository: [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape).

The project retains the Apache-2.0 license declared in its original package metadata. Third-party models, datasets, and source files retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
