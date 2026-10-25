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

Install a PyTorch wheel compatible with your CUDA runtime and driver. Granite requires enough memory for two 1B model replicas; the original runs used an A100 80GB. The full BF16 ERNIE checkpoint and its workspace run on one 80GB GPU.

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

## 3. Data and fixed settings

The tasks are `arc_easy` and `openbookqa`. Each task contributes 64 official training questions for search and a disjoint set of 256 training questions for final selection. The complete test splits contain 2,376 and 500 questions, respectively.

Dataset revisions:

- ARC-Easy: `210d026faf9955653af8916fad021475a3f00453`.
- OpenBookQA: `388097ea7776314e93a529163e0fea805b8a6454`.

Each search has a population of 16 and generates 16 offspring per generation for 15 generations: 256 distinct candidates in total. Final selection considers all 16 initial candidates and eight generated candidates. CE/KL weights are 0.75/0.25, the temperature is 0.25, and choice scores are normalized by character length.

Each case records its settings, sampled row indices, source-data hashes, and tokenization hashes in `config.json` and `proxy_data.json`. Fixed initial populations, expert rankings, and final masks are included. Reproduction starts from these inputs and does not require historical run directories.

## 4. Full search and final test

Granite:

```bash
python scripts/reproduce_tape.py run --case granite_24 --output runs/granite75 --test
python scripts/reproduce_tape.py run --case granite_16 --output runs/granite50 --test
```

ERNIE:

```bash
python scripts/reproduce_tape.py run --case ernie_48 --model-path models/ernie --output runs/ernie75 --test
python scripts/reproduce_tape.py run --case ernie_32 --model-path models/ernie --output runs/ernie50 --test
python scripts/reproduce_tape.py run --case ernie_48_identity_only --model-path models/ernie --output runs/identity --test
python scripts/reproduce_tape.py run --case ernie_48_allocation_only --model-path models/ernie --output runs/allocation --test
```

Use a new output directory for each command. The runner recomputes candidate scores through model inference and does not read the released test scores. It selects the final expert set on the independent training subset and writes `frozen_selection.json` before opening test. Execution records separately report search, selection, total runtime, and process peak memory.

## 5. Re-evaluate released expert sets

```bash
python scripts/reproduce_tape.py evaluate --case granite_16 --method tape --output runs/granite50_test.json
python scripts/reproduce_tape.py evaluate --case granite_16 --method reap --output runs/granite50_reap.json
python scripts/reproduce_tape.py evaluate --case ernie_48 --model-path models/ernie --method tape --output runs/ernie75_test.json
python scripts/reproduce_tape.py evaluate --case ernie_48 --model-path models/ernie --method same_capacity_reap --output runs/ernie75_projection.json
python scripts/reproduce_tape.py evaluate --case ernie_48 --model-path models/ernie --method initial_best --output runs/ernie75_initial.json
```

All main cases support `original`, `reap`, `frequency`, `hybrid`, and `tape`. The ERNIE main cases additionally support `ean_mean`. `same_capacity_reap` and `initial_best` are available only for ERNIE 75%. Full-test evaluation does not use a sample limit. Read `tasks.*.acc_norm`, not the legacy `mean_primary_acc` field.

## 6. Verification and reproduction scope

CPU replay checks every candidate, the expert budgets, and the training-based selection. GPU commands recompute model scores. The released experiments are individual runs, not multi-run averages. BF16 outputs can vary across hardware and numerical implementations; the release therefore records the software environment, masks, data hashes, and source code.

Full inference reproduction uses the frozen starting points supplied here. From-scratch calibration of REAP/Frequency/EAN rankings is upstream of these inputs. The release provides the resulting rankings and expert sets rather than requiring all earlier calibration and historical searches. Model and dataset revisions and third-party terms are listed in `THIRD_PARTY_NOTICES.md`.

## 7. Report

`report/main.tex` is the English report manuscript. Compile it with pdfLaTeX and the official NeurIPS 2026 `neurips_2026.sty`; the template download location is recorded in the source header. The repository is [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape). The local Chinese working draft and its editorial annotations are not distributed.
