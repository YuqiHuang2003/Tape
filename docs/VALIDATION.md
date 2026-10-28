# Release validation

## October 2026 multi-seed update

The release contains 15 replayable searches: 13 main runs and two single-run search ablations. Validation uses CPU checks and saved experiment records; no model search or inference is run as part of this update.

| Check | Scope |
|---|---|
| File integrity and feasibility | Verify every release checksum, exact global expert budgets, layer bounds, distinct candidates, and frozen output masks. |
| CPU search replay | Recreate all 256 candidates and the training-selected output in each of the 15 cases using recorded training objectives. |
| Statistical reconstruction | Recompute means and sample standard deviations from the 13 individual main-run test summaries; verify original, requested, and all-recorded seed groups. |
| Unit tests | Test search operators, task objectives, pruning controllers, release verification, statistics, and replay. |
| Portable figures | Generate both active figures from the released CSV without local experiment directories or model weights. |
| Independent package | Extract the distribution ZIP and repeat integrity, test, and replay checks outside the checkout. |
| Distribution hygiene | Check for credentials, private absolute paths, model weights, caches, oversized files, and non-English release text. |

Run these checks from the repository root:

```bash
python scripts/reproduce_tape.py check
python -m pytest
python scripts/reproduce_tape.py replay --case all
python scripts/reproduce_tape.py statistics
```

The CPU-only dependency set skips seven controller tests that require PyTorch. An environment with PyTorch runs those tests as well. CPU replay verifies the recorded search logic and selection; it does not remeasure the test accuracies.

Checks completed on 28 October 2026: all 89 tests passed with PyTorch; the CPU-only environment passed 82 tests and skipped the seven PyTorch-dependent tests. All 15 CPU replays matched the 256 saved candidates and final selections. Statistical reconstruction, checksum and feasibility checks, distribution scans, and both figure builds passed. An independently extracted ZIP installed successfully with no broken dependencies and repeated the CPU tests, all replays, integrity checks, and portable figure generation successfully. No GPU was used.

The latest author-supplied report PDF is included unchanged. Its checksum and the distinction from the older manuscript source are documented in [report provenance](REPORT_PROVENANCE.md). The PDF was not newly compiled during packaging.

## Earlier GPU validation: 25 October 2026

The previous release was independently extracted, installed, and tested. It passed 77 tests with PyTorch, or 70 tests with seven skips in the CPU-only environment, and replayed all six cases available at that time.

A separate full-test inference check of the Granite 50% seed-6708 mask covered all 2,376 ARC-Easy and 500 OpenBookQA questions. It reproduced 37.67% on ARC-Easy, 27.00% on OpenBookQA, and a 32.33% average. The expert-mask and test-data hashes matched. Evaluation used one A100 and took 80.60 seconds; this is not a search timing or a new search seed.

The original local validation artifact has SHA-256 `ca0559ad5bd0198034a235a311ff6dbc9fc58feefb39dd8a951cf083dc159a64`. Raw test text and large evaluation outputs are excluded from this repository. The earlier GPU check is not presented as a GPU rerun of the current multi-seed update.
