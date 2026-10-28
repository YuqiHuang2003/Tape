# Reproducibility information and compute costs

This report separates hardware, search runtime, test runtime, and memory measurements for the original reference runs. These are individual-run timings, not averages across the three-seed result table. The Granite 50% reference is seed 6708; the earlier seed-6707 run is listed separately.

The current multi-seed scores are in [RESULTS.md](RESULTS.md). Each released case also retains its recorded search wall time in `artifacts/cases/<case>/search.json` and its evaluation cost in `evaluations.json`. The reference timing table below does not estimate the total cost of the later repeated searches.

## Hardware and execution

Every run used **one NVIDIA A100-SXM4-80GB**. Granite evaluated candidates with two model replicas on the same GPU; ERNIE used one replica. All model computations used BF16 forward inference with frozen weights.

## Search and test runtime

The search column measures the evolutionary search over 256 candidates. It excludes downloading and loading models, preparing teacher scores, selection on the additional training questions, and full-test evaluation. The last-invocation column is the interval between the recorded start and completion in the run state. For a resumed run, that interval is not the duration of the entire experiment.

| Experiment | Search (minutes) | Final-set full test (seconds) | Last invocation (minutes) | Timing details |
|---|---:|---:|---:|---|
| Granite 75% | 4.93 | 100.54 | 16.27 | Candidate scores recomputed |
| Granite 50%, seed-6708 reference | 3.88 | 83.16 | 6.65 | Ten identical-input score records reused; other scores recomputed |
| ERNIE 75% | Full-search timer unavailable | 282.90 | 18.73 | The recorded 2.11 seconds measure cache replay after resumption |
| ERNIE 50% | 75.01 | 210.25 | 111.45 | Candidate scores recomputed |
| ERNIE 75% w/o Layer Allocation | 105.93 | 288.82 | 151.00 | Candidate scores recomputed |
| ERNIE 75% w/o Identity Search | 103.40 | 283.48 | 157.32 | Candidate scores recomputed |
| Granite 50%, original seed 6707 | 4.04 | 83.25 | 13.57 | Candidate scores recomputed |

The initial ERNIE 75% invocation completed search and saved its scores, but resumption overwrote the search-stage timer. The remaining records do not recover the exact cold-search duration. The 2.11-second cache replay and file modification times are not substitutes for that duration.

The Granite 50% rerun reused tests for unchanged direct-scoring baselines and performed a new complete test for the selected TAPE set. Test runtimes above refer to the final set; reused results are not counted again as newly incurred work.

## Memory and resource accounting

| Experiment | Recorded peak device memory at final-set evaluation (GiB) |
|---|---:|
| Granite 75% | 6.04 |
| Granite 50%, seed-6708 reference | 6.01 |
| ERNIE 75% | 43.02 |
| ERNIE 50% | 42.49 |
| ERNIE 75% w/o Layer Allocation | 42.87 |
| ERNIE 75% w/o Identity Search | 43.02 |
| Granite 50%, original seed 6707 | 6.04 |

These values are the device-peak fields recorded by the evaluation process. Earlier search work or resident replicas can affect them. They are not isolated measurements of a physically pruned model's deployment memory. Historical searches did not consistently record process-level peak memory.

Single-GPU allocation hours can be calculated from measured occupied seconds divided by 3,600. The available stage records do not cover every historical interruption and recovery, so no complete-project GPU-hour total is reported. Energy consumption and monetary cost were not measured.

## Software and reproduction

The experiments used Python 3.10.21, PyTorch 2.5.1, CUDA 12.4, and lm-eval 0.4.8. Granite used Transformers 4.46.3, datasets 3.1.0, and accelerate 1.1.1. ERNIE used Transformers 4.53.3, datasets 3.6.0, and accelerate 1.7.0. Install the two model environments separately.

Repository: [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape). Installation, CPU replay, GPU search, and evaluation commands are in [the reproduction guide](REPRODUCIBILITY.md).

## Evidence

[COMPUTE_COSTS.json](COMPUTE_COSTS.json) records the measurements, source artifact paths, and hashes. GPU models come from full-test environment records; run states expose exactly one device. The additional package-validation inference check is documented separately in [VALIDATION.md](VALIDATION.md).
