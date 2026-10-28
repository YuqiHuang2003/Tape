# Paper figure design specification

The paper uses two figures: a minimal pruning illustration and an expert-selection comparison. Both use English labels and export as vector PDF, editable SVG, and PNG preview. Their stable filenames are `fig01_tape_framework` and `fig04_expert_selection`.

## Figure 1: pruning under a fixed budget

Show three parallel network views: Original, Uniform, and TAPE. Use circular expert nodes and thin neural-network-style connections, not a square-cell matrix. Each horizontal row represents a layer, with its retained-expert count beside it.

| Network | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Total retained |
|---|---:|---:|---:|---:|---:|
| Original | 8 | 8 | 8 | 8 | 32 |
| Uniform | 6 | 6 | 6 | 6 | 24 |
| TAPE | 7 | 5 | 6 | 6 | 24 |

The TAPE subtitle is `75%`. The allocation changes are deliberately moderate: no layer is more than one slot away from the uniform count. Pruned identities differ between rows in the uniform view because uniform pruning fixes counts, not expert identities.

Use blue/gold outlines, faint pruned nodes, and a compact kept/pruned legend. Text is limited to network headings, subtitles, layer labels, row counts, and legend labels. Do not include the search workflow, objective equations, data splits, or explanatory text boxes.

This figure illustrates the representation; its four-layer network and lines are not the dimensions or precise connectivity of either experimental model.

## Figure 2: expert choices at matched capacities

Use the actual ERNIE 75% seed-6707 keep sets. Compare TAPE with REAP reselected at the same per-layer capacities, ordering experts by REAP rank within each of 27 layers. The capacity boundary remains visible.

| Selection category | Fill |
|---|---|
| Kept by both | Pale gray `#F2F2F2` |
| Pruned by both | White `#FFFFFF` |
| Kept only by TAPE | Light green `#CCEBC5` |
| Kept only by matched-capacity REAP | Light orange `#FED9A6` |

Use categorical legend labels and `Matched layer capacities`. Hide all numeric annotations in the image: axis tick numbers, test scores, and legend counts. The image ends at the legend, without notes underneath. Keep the numerical context in the caption and source table.

The source is [`artifacts/expert_analysis/expert_status.csv`](../artifacts/expert_analysis/expert_status.csv). Counts are 1,168 kept by both, 304 pruned by both, and 128 in each disagreement category. The paired single-run test averages are 62.67% for TAPE and 58.07% for matched-capacity REAP. The figure illustrates differences between expert sets, not individual-expert causality or a systematic preference over depth.

## Placement and generation

Place Figure 1 with the method's joint representation and Figure 2 with the expert-selection analysis. Suggested captions and image previews are in the [figure README](figures/README.md).

Run from the repository root:

```bash
python3.10 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r requirements-figures.txt
.venv-figures/bin/python scripts/generate_paper_figures.py --package
```

Outputs are written to `docs/figures/`. Figure generation uses only the illustration specification and released expert table; it performs no model search, inference, or manuscript editing.
