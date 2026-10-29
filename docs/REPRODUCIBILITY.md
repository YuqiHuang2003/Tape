# Reproduction guide

Run every command from the repository root. Expose exactly one GPU. Replace `0` below with a single available device ID on your machine, not a list of IDs.

## 1. Environments

CPU verification requires Python 3.10, NumPy, and PyYAML. The experiments used Python 3.10.21, PyTorch 2.5.1, and CUDA 12.4, with BF16 inference throughout.

Granite:

```bash
python3.10 -m venv .venv-granite
source .venv-granite/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r requirements/granite.txt
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
```

Use a separate environment for ERNIE:

```bash
python3.10 -m venv .venv-ernie
source .venv-ernie/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r requirements/ernie.txt
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
```

Install a PyTorch wheel compatible with your CUDA runtime and driver. TAPE's Granite runner uses two 1B model replicas; EvoESAP uses one resident model and caches teacher probabilities on the CPU. The original runs used an A100 80GB. The full BF16 ERNIE checkpoint and its workspace run on one 80GB GPU.

## 2. Models

The first Granite run downloads this pinned model revision:

```text
ibm-granite/granite-3.1-1b-a400m-base
408b6e90baab8cf24f4aa9f8e19703ffa0a53b29
```

For ERNIE, download the pinned checkpoint and install the compatibility sources used in the experiments:

```bash
python scripts/prepare_released_ernie.py --destination models/ernie
```

This downloads `baidu/ERNIE-4.5-21B-A3B-PT` at revision `23c6cb79af127ec966c9c03506dc80a3ff9fa953`. Compatibility files are installed in the new checkpoint directory, with upstream copies preserved separately. Weights are not modified. Allow approximately 44 GB for the checkpoint and additional space for download caches.

## 3. Data and TAPE settings

The tasks are `arc_easy` and `openbookqa`. Each task contributes 64 official training questions for search and a disjoint set of 256 training questions for final selection. The complete test splits contain 2,376 and 500 questions, respectively.

Dataset revisions:

- ARC-Easy: `210d026faf9955653af8916fad021475a3f00453`.
- OpenBookQA: `388097ea7776314e93a529163e0fea805b8a6454`.

Each search has a population of 16 and generates 16 offspring per generation for 15 generations: 256 distinct candidates in total. Final selection considers all 16 initial candidates and eight generated candidates. CE/KL weights are 0.75/0.25, the temperature is 0.25, and choice scores are normalized by character length.

The four main settings start from REAP, Frequency, REAP+Frequency, and a historical feasible TAPE mask, followed by 12 local perturbations. Training samples and anchor masks are fixed across seeds; only the evolutionary-search randomness changes.

Each case records its settings, sampled row indices, source-data hashes, and tokenization hashes in `config.json` and `proxy_data.json`. Fixed initial populations, expert rankings, and final masks are included. Reproduction starts from these inputs and does not require historical run directories.

The TAPE component contains 15 independently replayable cases: 13 main-search runs and two single-run search ablations. The original case names remain available as reference runs; additional runs identify their search seed in the case name. The 12 EvoESAP cases are listed in Section 5.

| Setting | Reference case and seed | Additional case names |
|---|---|---|
| ERNIE 75% | `ernie_48`, seed 6707 | `ernie_48_seed_6708`, `ernie_48_seed_6709` |
| ERNIE 50% | `ernie_32`, seed 6707 | `ernie_32_seed_6708`, `ernie_32_seed_6709` |
| Granite 75% | `granite_24`, seed 6707 | `granite_24_seed_6708`, `granite_24_seed_6709` |
| Granite 50% | `granite_16`, seed 6708 | `granite_16_seed_6707`, `granite_16_seed_6709`, `granite_16_seed_6710` |

The ablation cases are `ernie_48_identity_only` and `ernie_48_allocation_only`, each with seed 6707. They are not three-seed averages. Select a run by its case name rather than changing the seed of an existing case: each case binds its configuration to the corresponding initial population, recorded search, and final mask.

## 4. TAPE search and final test

Granite:

```bash
python scripts/reproduce_tape.py run --case granite_24 --output runs/granite75 --test
python scripts/reproduce_tape.py run --case granite_16 --output runs/granite50 --test
python scripts/reproduce_tape.py run --case granite_16_seed_6710 --output runs/granite50_seed6710 --test
```

ERNIE:

```bash
python scripts/reproduce_tape.py run --case ernie_48 --model-path models/ernie --output runs/ernie75 --test
python scripts/reproduce_tape.py run --case ernie_48_seed_6708 --model-path models/ernie --output runs/ernie75_seed6708 --test
python scripts/reproduce_tape.py run --case ernie_32 --model-path models/ernie --output runs/ernie50 --test
python scripts/reproduce_tape.py run --case ernie_48_identity_only --model-path models/ernie --output runs/identity --test
python scripts/reproduce_tape.py run --case ernie_48_allocation_only --model-path models/ernie --output runs/allocation --test
```

Use a new output directory for each command. The runner recomputes candidate scores through model inference and does not read the released test scores. It selects the final expert set on the independent training subset and writes `frozen_selection.json` before opening test. Execution records separately report search, selection, total runtime, and process peak memory.

## 5. EvoESAP baseline

The release includes 12 completed EvoESAP runs: four model/budget settings, each with search seeds 6707, 6708, and 6709. They use the same pinned checkpoints, 64 search questions per task, 256 disjoint selection questions per task, exact expert budgets, and per-layer bounds as TAPE. The within-layer expert order is the unchanged REAP ranking used by the TAPE reference.

EvoESAP maximizes full-vocabulary Expected Speculative Acceptance Proxy (ESAP), `sum_v min(p_original(v), p_pruned(v))`, at teacher-forced gold-answer positions. Scores are averaged over answer positions and then questions, with temperature 1.0. This is not option-distribution KL or answer accuracy. It requires forward passes only; it does not decode text or train weights.

The allocation search has 32 initial candidates, four elites, and eight generations of 28 new candidates, for 256 distinct evaluations. Initialization includes uniform, increasing, decreasing, middle-heavy, and middle-light allocations, then uniformly sampled feasible count vectors. Transfer mutations preserve the total number of experts. Each allocation retains the corresponding prefix of the fixed REAP ranking in every layer.

Final selection evaluates the 24 highest search-ESAP candidates on the independent training subset; if uniform REAP is absent, it replaces the last shortlisted item. The highest selection-ESAP candidate wins, with mask SHA-256 as the tie-breaker. The expert set is frozen before full-test evaluation.

These are controlled-comparison adaptations of the original method: target-task prompt/gold-answer samples replace its calibration corpus; 256 evaluations and TAPE's narrower layer bounds replace its original budget and space; independent training-set selection replaces direct selection on search fitness; and random initial count vectors are sampled uniformly from the feasible set. The operators and full-vocabulary ESAP are implemented in `src/dsai6707/evoesap.py` and `src/dsai6707/esap.py`. Exact settings are recorded in `artifacts/evoesap/protocol.json`.

| Setting | EvoESAP case names |
|---|---|
| ERNIE 75% | `ernie_48_seed_6707`, `ernie_48_seed_6708`, `ernie_48_seed_6709` |
| ERNIE 50% | `ernie_32_seed_6707`, `ernie_32_seed_6708`, `ernie_32_seed_6709` |
| Granite 75% | `granite_24_seed_6707`, `granite_24_seed_6708`, `granite_24_seed_6709` |
| Granite 50% | `granite_16_seed_6707`, `granite_16_seed_6708`, `granite_16_seed_6709` |

Use the environment for the selected model family. The following examples recompute the search and selected set's full-test accuracy:

```bash
python scripts/reproduce_evoesap.py run --case granite_24_seed_6707 --output runs/evoesap_granite75_seed6707 --test
python scripts/reproduce_evoesap.py run --case granite_16_seed_6707 --output runs/evoesap_granite50_seed6707 --test
python scripts/reproduce_evoesap.py run --case ernie_48_seed_6707 --model-path models/ernie --output runs/evoesap_ernie75_seed6707 --test
python scripts/reproduce_evoesap.py run --case ernie_32_seed_6707 --model-path models/ernie --output runs/evoesap_ernie50_seed6707 --test
```

Choose the matching case name and a separate output directory for each seed. Re-evaluate a released final mask without searching:

```bash
python scripts/reproduce_evoesap.py evaluate --case granite_24_seed_6707 --output runs/evoesap_granite75_test.json
python scripts/reproduce_evoesap.py evaluate --case ernie_48_seed_6707 --model-path models/ernie --output runs/evoesap_ernie75_test.json
```

The baseline artifacts contain configuration and data evidence, fixed rankings, candidate masks and ESAP fitness, selection records, frozen winner masks, and compact full-test evaluations. They do not contain model weights, question text, or full-vocabulary probability caches.

## 6. Re-evaluate released TAPE expert sets

```bash
python scripts/reproduce_tape.py evaluate --case granite_16 --method tape --output runs/granite50_test.json
python scripts/reproduce_tape.py evaluate --case granite_16 --method reap --output runs/granite50_reap.json
python scripts/reproduce_tape.py evaluate --case ernie_48 --model-path models/ernie --method tape --output runs/ernie75_test.json
python scripts/reproduce_tape.py evaluate --case ernie_48 --model-path models/ernie --method same_capacity_reap --output runs/ernie75_projection.json
python scripts/reproduce_tape.py evaluate --case ernie_48 --model-path models/ernie --method initial_best --output runs/ernie75_initial.json
```

The four reference main cases support `original`, `reap`, `frequency`, `hybrid`, and `tape`. The ERNIE reference main cases additionally support `ean_mean`. The paired diagnostics `same_capacity_reap` and `initial_best` belong to `ernie_48` (seed 6707), not to the three-seed mean. Additional seed cases provide the corresponding searched TAPE expert sets. Full-test evaluation does not use a sample limit. Read `tasks.*.acc_norm`, not the legacy `mean_primary_acc` field.

## 7. Verification and reproduction scope

Run the following without loading a model or using a GPU:

```bash
python scripts/reproduce_tape.py check
python scripts/reproduce_evoesap.py check
python -m pytest
python scripts/reproduce_tape.py replay --case all
python scripts/reproduce_evoesap.py replay --case all
python scripts/reproduce_tape.py statistics
python scripts/reproduce_evoesap.py statistics
```

`check` verifies released inputs and expert budgets. CPU replay regenerates every recorded candidate and verifies training-based selection for all 15 TAPE cases and 12 EvoESAP cases. It uses recorded training fitness and does not remeasure inference accuracy. The statistics commands aggregate the released full-test scores without rerunning search or inference.

The paper's main-result cohort uses seeds 6707, 6708, and 6709 for both methods and all four settings. Means and sample standard deviations (`ddof=1`) describe independent search runs, not repeated inference of a fixed mask. Granite 50% TAPE also includes seed 6710, added after the original results were inspected. The original three seeds and the additional run are all retained; the requested 6708–6710 subset and the all-four summary are distinguished from the paper's cohort. The two TAPE search ablations, matched-capacity comparison, and evolution/expert-selection analyses are single-run results. See [results](RESULTS.md) and [report provenance](REPORT_PROVENANCE.md) for the reporting groups.

GPU commands recompute model scores. BF16 outputs can vary across hardware and numerical implementations; the release therefore records the software environment, masks, data hashes, and source code. This repository update packages completed experiments; it does not add new GPU experiments.

Full inference reproduction uses the frozen starting points supplied here. From-scratch calibration of REAP/Frequency/EAN rankings is upstream of these inputs. The release provides the resulting rankings and expert sets rather than requiring all earlier calibration and historical searches. Model and dataset revisions and third-party terms are listed in `THIRD_PARTY_NOTICES.md`.

## 8. Figures

The two active paper figures can be generated without model weights or a GPU:

```bash
python3.10 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r requirements-figures.txt
.venv-figures/bin/python scripts/generate_paper_figures.py --package
```

The pruning diagram is illustrative. The expert-selection comparison reads the portable source table `artifacts/expert_analysis/expert_status.csv`; it uses the single ERNIE 75% reference run and matched layer capacities. Generated PDF, SVG, and PNG files are written to `docs/figures/`. Historical figure revisions are not required.

## 9. Report

[`report/report.pdf`](../report/report.pdf) is the latest author-supplied English report, published unchanged. Its current LaTeX source was not supplied, so obsolete manuscript sources are not distributed. [Report provenance](REPORT_PROVENANCE.md) records its checksum and the remaining differences between the PDF wording and recorded experiments. The repository is [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape). The local Chinese working draft and its editorial annotations are not distributed.
