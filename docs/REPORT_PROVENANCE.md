# Report provenance

The latest author-supplied English report is published unchanged as [`report/report.pdf`](../report/report.pdf). Its SHA-256 is `f17f1aef77511fd114d6952ac7cb6828f215c17b5d4fea871138034f20753444` (14 pages, supplied on 28 October 2026).

The earlier English manuscript is retained at `report/legacy/main.tex`. It contains the previous single-run tables and is not the source of the current PDF. The current PDF's LaTeX source was not supplied with this update.

## Report text and experimental records

The implementation and released records establish the following details. The supplied PDF has not been rewritten to resolve these differences.

| Topic | Supplied PDF | Released records |
|---|---|---|
| Main-search initialization | REAP and its perturbations (§3.2 and §4.4) | Four anchors: REAP, Frequency, REAP+Frequency, and a historical feasible TAPE set, plus 12 local perturbations. |
| Search ablation repeats | Three-seed means for both search ablations (§4.4, §5.3, Table 4) | Each search ablation is one seed-6707 run; repeated searches cover the four main settings only. The ablation scores 59.79 and 58.42 are individual-run averages. |
| Granite 50% cohort | Seeds 6707–6709: Avg 32.18 ± 0.42 (Table 3) | All four seeds 6707–6710 are retained. The subsequently requested 6708–6710 subset has Avg 32.24 ± 0.32; all four have Avg 32.10 ± 0.37. Seed 6710 is an additional, posthoc search run, not a replacement record. |
| Granite 50% ARC standard deviation | 0.76 for seeds 6707–6709 (Table 3) | The sample standard deviation is 0.768025572 percentage points, which rounds to 0.77. |

All standard deviations above are sample standard deviations (`ddof=1`) over search seeds. [Results](RESULTS.md) keeps the seed groups explicit; `python scripts/reproduce_tape.py statistics` computes the summaries from the released scores. The main ERNIE and Granite 75% results use seeds 6707–6709. Deterministic baselines and the ERNIE 75% matched-capacity/evolution diagnostics are evaluated once.

## Figures and reproduction

The two active figures are released with their generating code and source expert-selection table. The pruning diagram illustrates a fixed budget; its small network is not an experimental model configuration. The expert-selection figure analyzes the seed-6707 ERNIE 75% result, not an average expert mask across seeds.

See the [reproduction guide](REPRODUCIBILITY.md) for the 15 case names, CPU replay, GPU inference commands, and figure generation. No new model search or inference was run for this packaging update.
