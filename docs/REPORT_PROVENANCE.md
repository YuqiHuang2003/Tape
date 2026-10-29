# Report provenance

The latest author-supplied English report is published unchanged as [`report/report.pdf`](../report/report.pdf). Its SHA-256 is `95a973251785ad8aacf47ef9d142673cb838111b287a0674f2810cd5e03e668c` (14 pages, supplied on 29 October 2026). It includes the completed EvoESAP results and the 102.17-minute ERNIE 75% search cost.

The current PDF's LaTeX source was not supplied. Obsolete manuscript sources are excluded rather than presented as the source of this PDF.

## Results and compute-cost correspondence

Tables 2–3 use seeds 6707–6709 for TAPE and EvoESAP. The release includes each run's final expert mask, candidate trace, training-set selection record, and complete-test scores. EvoESAP is the budget-matched REAP-ranking adaptation described in the [reproduction guide](REPRODUCIBILITY.md), with its saved protocol under `artifacts/evoesap/protocol.json`.

The additional Granite 50% TAPE seed 6710 remains available but is not part of the paper's three-seed mean. Its requested 6708–6710 summary is 32.24 ± 0.32, whereas the paper's 6707–6709 summary is 32.18 ± 0.42. All four recorded seeds are retained.

Table 7's ERNIE 75% search cost, 102.17 minutes, is the mean of the valid seed-6708 and seed-6709 timers. The seed-6707 reference supplies its 282.90-second final-test time. [Compute costs](COMPUTE_COSTS.md) and its JSON evidence identify these sources.

## Report text and experimental records

The implementation and released records establish the following details. The supplied PDF has not been rewritten to resolve these differences.

| Topic | Supplied PDF | Released records |
|---|---|---|
| Main-search initialization | REAP and its perturbations (§3.2 and §4.4) | Four anchors: REAP, Frequency, REAP+Frequency, and a historical feasible TAPE set, plus 12 local perturbations. |
| Search ablation repeats | Three-seed means for both search ablations (§4.4, §5.3, Table 4) | Each search ablation is one seed-6707 run; repeated searches cover the four main settings only. The ablation scores 59.79 and 58.42 are individual-run averages. |
| Granite 50% ARC standard deviation | 0.76 for seeds 6707–6709 (Table 3) | The sample standard deviation is 0.768025572 percentage points, which rounds to 0.77. |

All standard deviations are sample standard deviations (`ddof=1`) over search seeds. [Results](RESULTS.md) keeps seed groups explicit. `python scripts/reproduce_tape.py statistics` and `python scripts/reproduce_evoesap.py statistics` compute summaries from the released scores. Deterministic baselines and the ERNIE 75% matched-capacity/evolution diagnostics are evaluated once.

EvoESAP performance-retained entries in Tables 2–3 use rounded displayed Avg values and SDs. The repository computes these ratios from unrounded scores, producing differences of 0.01 percentage points in some means or SDs. Task accuracies and Avg statistics are unchanged.

## Figures and reproduction

The two active figures are released with their generating code and source expert-selection table. The pruning diagram illustrates a fixed budget; its small network is not an experimental model configuration. The expert-selection figure analyzes the seed-6707 ERNIE 75% result, not an average expert mask across seeds.

See the [reproduction guide](REPRODUCIBILITY.md) for the 15 TAPE cases, 12 EvoESAP cases, CPU replay, GPU inference commands, and figure generation. This repository update packages completed experiments; no new model search or inference was run for packaging.
