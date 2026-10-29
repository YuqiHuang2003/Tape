# Reproducibility information and compute costs

This file documents the TAPE costs in Table 7 of the current report. Search and test timings are reference-run measurements, except that ERNIE 75% search uses the mean of two valid timed searches. They are not the total cost of the three-seed experiments. The Granite 50% timing reference is seed 6708.

The current multi-seed scores are in [RESULTS.md](RESULTS.md). Each released case also retains its recorded search wall time in `artifacts/cases/<case>/search.json` and its evaluation cost in `evaluations.json`. The reference timing table below does not estimate the total cost of the later repeated searches.

## Hardware and execution

Every run used **one NVIDIA A100-SXM4-80GB**. Granite evaluated candidates with two model replicas on the same GPU; ERNIE used one replica. All model computations used BF16 forward inference with frozen weights.

## Search and test runtime

The search column measures the evolutionary search over 256 candidates. It excludes downloading and loading models, preparing teacher scores, selection on the additional training questions, and full-test evaluation.

| Experiment | Search (minutes) | Final-set full test (seconds) |
|---|---:|---:|
| Granite 75% | 4.93 | 100.54 |
| Granite 50% | 3.88 | 83.16 |
| ERNIE 75% | 102.17 | 282.90 |
| ERNIE 50% | 75.01 | 210.25 |
| ERNIE 75% w/o Layer Allocation | 105.93 | 288.82 |
| ERNIE 75% w/o Identity Search | 103.40 | 283.48 |

ERNIE 75% search is the mean of seed 6708 (103.32085 minutes) and seed 6709 (101.01681 minutes), rounded to 102.17 minutes. Its final-test time is from the seed-6707 reference. The original seed-6707 search timer was overwritten during cache resumption; the two later measurements supply the reported search cost rather than reconstructing that timer.

Each of these two ERNIE searches reused ten matching external score records, as recorded in the timing evidence; they were not cold-cache measurements.

The Granite 50% reference records ten matching external score-cache hits; these are individual score records, not necessarily ten complete search-candidate evaluations. The rerun reused tests for unchanged direct-scoring baselines and performed a new complete test for the selected TAPE set. Test runtimes above refer to the final set.

## Memory and resource accounting

| Experiment | Recorded peak device memory at final-set evaluation (GiB) |
|---|---:|
| Granite 75% | 6.04 |
| Granite 50%, seed-6708 reference | 6.01 |
| ERNIE 75% | 43.02 |
| ERNIE 50% | 42.49 |
| ERNIE 75% w/o Layer Allocation | 42.87 |
| ERNIE 75% w/o Identity Search | 43.02 |

These values are the device-peak fields recorded by the evaluation process. Earlier search work or resident replicas can affect them. They are not isolated measurements of a physically pruned model's deployment memory. Historical searches did not consistently record process-level peak memory.

Single-GPU allocation hours can be calculated from measured occupied seconds divided by 3,600. The available stage records do not cover every historical interruption and recovery, so no complete-project GPU-hour total is reported. Energy consumption and monetary cost were not measured.

## Software and reproduction

The experiments used Python 3.10.21, PyTorch 2.5.1, CUDA 12.4, and lm-eval 0.4.8. Granite used Transformers 4.46.3, datasets 3.1.0, and accelerate 1.1.1. ERNIE used Transformers 4.53.3, datasets 3.6.0, and accelerate 1.7.0. Install the two model environments separately.

Repository: [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape). Installation, CPU replay, GPU search, and evaluation commands are in [the reproduction guide](REPRODUCIBILITY.md).

## Evidence

[COMPUTE_COSTS.json](COMPUTE_COSTS.json) records the measurements, source artifact paths, and hashes, including the two valid ERNIE 75% timing sources. GPU models come from full-test environment records; run states expose exactly one device. EvoESAP's per-run search and evaluation records are distributed separately under `artifacts/evoesap/cases/`; the table above is the TAPE timing table, not a pooled baseline timing comparison. Package validation is documented in [VALIDATION.md](VALIDATION.md).
