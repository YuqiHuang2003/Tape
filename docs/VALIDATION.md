# Release validation

The current release packages 15 TAPE searches and 12 EvoESAP searches, together with their completed full-test evaluations. Release validation uses CPU checks and saved records; it does not launch new GPU inference or search.

Validation completed on 29 October 2026: all 161 tests passed with PyTorch; the CPU-only environment passed 150 tests and skipped 11 PyTorch-dependent tests. CPU replay matched all 256 candidates and the training-selected winner in each of the 15 TAPE and 12 EvoESAP cases. Both figures and their distribution package regenerated successfully. An independently extracted ZIP installed as version 0.3.0, passed dependency checks and the CPU-only test suite, verified all artifact checksums, recomputed both statistical summaries, and regenerated the figures and their package. These checks use saved records; they are not new model-inference experiments.

| Check | Scope |
|---|---|
| Integrity and feasibility | Verify release checksums, global expert budgets, per-layer bounds, distinct candidates, and frozen winner masks. |
| TAPE replay | Regenerate all 256 candidates and the training-selected winner for each of 15 cases from recorded task objectives. |
| EvoESAP replay | Regenerate all 256 candidates and independent training-set selection for each of 12 cases from recorded full-vocabulary ESAP fitness. |
| Statistics | Recompute means and sample standard deviations from per-run full-test scores, preserving the paper and additional Granite 50% seed cohorts. |
| Tests | Exercise search operators, task objectives, ESAP, pruning controllers, verification, statistics, and portable runners. |
| Figures and distribution | Rebuild the two figures from released inputs and scan the package for private paths, credentials, weights, caches, and non-English text. |

Run from the repository root after installing the development dependencies:

```bash
python scripts/reproduce_tape.py check
python scripts/reproduce_evoesap.py check
python -m pytest
python scripts/reproduce_tape.py replay --case all
python scripts/reproduce_evoesap.py replay --case all
python scripts/reproduce_tape.py statistics
python scripts/reproduce_evoesap.py statistics
```

Replay verifies search logic and training-only selection using saved fitness values. It does not load model weights or remeasure accuracy. Tests requiring PyTorch are skipped when PyTorch is absent; model-execution commands require the appropriate GPU environment from the [reproduction guide](REPRODUCIBILITY.md).

The latest supplied PDF is included unchanged, not recompiled during packaging. Its checksum and the remaining differences between its wording and the recorded experiments are documented in [report provenance](REPORT_PROVENANCE.md).
