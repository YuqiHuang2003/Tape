# Release validation

These checks were performed on the release package. They do not replace the original experimental results.

| Check | Outcome |
|---|---|
| Installation from an independently extracted package | Passed |
| Installation of `.[dev]` in a fresh CPU environment | Passed; no broken dependencies |
| Release-file and expert-budget verification | Passed |
| Unit tests in a fresh CPU environment | 70 passed; 7 PyTorch-dependent controller tests skipped |
| Unit tests with PyTorch installed | All 77 passed |
| CPU replay of six NSGA-II experiments | All 256 candidates and the final selection matched in each case |
| Full test of Granite at 50% expert retention on one GPU | Matched the recorded main result |
| Checks for credentials, private machine paths, weights, and caches | Passed |
| English-only release-text check | Passed |

The repeated Granite 50% test covered all 2,376 ARC-Easy questions and 500 OpenBookQA questions. It reproduced 37.67% on ARC-Easy, 27.00% on OpenBookQA, and a 32.33% average. The expert mask and test-data hashes also matched. This check used one A100, took 80.60 seconds for evaluation, and released GPU memory on completion. Its runtime is separate from the original search cost.

The complete local validation artifact has SHA-256 `ca0559ad5bd0198034a235a311ff6dbc9fc58feefb39dd8a951cf083dc159a64`. Raw test text and large evaluation outputs are excluded from the public package.

The ERNIE GPU experiments were not rerun during packaging. All four ERNIE searches were replayed on CPU; their existing full-test metrics and model compatibility sources are included in the release manifest.

The public manuscript is in English. The document editor's compiler was unavailable during packaging, so no newly compiled report PDF is claimed. Source files, numerical tables, and experiment artifacts are provided in [YuqiHuang2003/Tape](https://github.com/YuqiHuang2003/Tape).
