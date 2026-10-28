"""Draw the matched-capacity expert-selection diagnostic from saved records."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figure_common import save_figure
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def _parse_bool(value: str) -> bool:
    """Reject malformed values instead of interpreting the string 'False' as true."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected an explicit CSV Boolean, got {value!r}")


def expert_csv_path(project_root: Path) -> Path:
    """Use the portable release artifact, with a development-tree fallback."""
    published = project_root / "artifacts/expert_analysis/expert_status.csv"
    if published.is_file():
        return published
    return project_root / "results/runs/uniform_tape_20261023/expert_analysis/expert_status.csv"


def draw_expert_matrix(project_root: Path, output_dir: Path) -> list[Path]:
    """Render Figure 4; no model loading, inference, or new experiments are used."""
    source = expert_csv_path(project_root)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 27 * 64:
        raise ValueError(f"Expected 1,728 layer-indexed experts, got {len(rows)}")

    # Class order: both kept, both pruned, TAPE only, matched-capacity REAP only.
    matrix = np.full((27, 64), -1, dtype=int)
    capacities: dict[int, int] = {}
    categories: Counter[int] = Counter()
    expert_keys: set[tuple[int, int]] = set()
    for row in rows:
        layer = int(row["layer"])
        expert = int(row["expert"])
        rank = int(row["reap_rank"])
        capacity = int(row["capacity"])
        if not (1 <= layer <= 27 and 0 <= expert < 64 and 1 <= rank <= 64):
            raise ValueError(f"Invalid expert coordinates: {layer=}, {expert=}, {rank=}")
        if (layer, expert) in expert_keys or matrix[layer - 1, rank - 1] != -1:
            raise ValueError("Duplicate expert or REAP rank within a layer")
        expert_keys.add((layer, expert))
        if not 0 <= capacity <= 64:
            raise ValueError(f"Invalid layer capacity: {capacity}")
        if layer in capacities and capacities[layer] != capacity:
            raise ValueError(f"Inconsistent capacity in MoE layer {layer}")
        capacities[layer] = capacity
        tape_keeps = _parse_bool(row["retained"])
        reap_keeps = _parse_bool(row["same_capacity_reap_retained"])
        if reap_keeps != (rank <= capacity):
            raise ValueError("Matched-capacity REAP mask disagrees with its rank cutoff")
        category = (
            0
            if tape_keeps and reap_keeps
            else 1
            if not tape_keeps and not reap_keeps
            else 2
            if tape_keeps
            else 3
        )
        matrix[layer - 1, rank - 1] = category
        categories[category] += 1

    expected = Counter({0: 1168, 1: 304, 2: 128, 3: 128})
    if categories != expected or np.any(matrix < 0):
        raise ValueError(f"Expert categories changed: expected {expected}, got {categories}")
    for layer in range(1, 28):
        retained = np.isin(matrix[layer - 1], [0, 2]).sum()
        if retained != capacities[layer]:
            raise ValueError(f"TAPE and REAP capacities differ in MoE layer {layer}")

    # ColorBrewer Pastel1 light green/orange, with neutral common-status fills.
    # Source: https://colorbrewer2.org/export/colorbrewer.json (Pastel1).
    palette = [
        "#F2F2F2",
        "#FFFFFF",
        "#CCEBC5",
        "#FED9A6",
    ]
    fig = plt.figure(figsize=(7.8, 5.1))
    ax = fig.add_axes([0.085, 0.265, 0.88, 0.565])
    ax.pcolormesh(
        np.arange(65) - 0.5,
        np.arange(28) - 0.5,
        matrix,
        cmap=ListedColormap(palette),
        norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
        shading="flat",
        antialiased=False,
        rasterized=False,
    )
    ax.set_xlim(-0.5, 63.5)
    ax.set_ylim(26.5, -0.5)
    ax.set_aspect("equal")
    # The cutoff is drawn as a staircase along cell edges, never through cells.
    boundary_x: list[float] = []
    boundary_y: list[float] = []
    for layer in range(1, 28):
        cutoff = capacities[layer] - 0.5
        boundary_x.extend([cutoff, cutoff])
        boundary_y.extend([layer - 1.5, layer - 0.5])
    ax.plot(boundary_x, boundary_y, color="#888888", linewidth=0.9)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Within-layer REAP rank", fontsize=9.5, labelpad=7)
    ax.set_ylabel("MoE layer", fontsize=9.5, labelpad=7)
    ax.tick_params(length=3, width=0.65, colors="#343B40")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#A9ADB0")

    fig.text(
        0.525,
        0.965,
        "Expert selection beyond REAP rankings",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
    )
    fig.text(
        0.525,
        0.913,
        "Matched layer capacities",
        ha="center",
        va="top",
        fontsize=9.5,
    )
    legend_handles = [
        Patch(facecolor=palette[0], edgecolor="#BBBBBB", linewidth=0.5, label="Kept by both"),
        Patch(
            facecolor=palette[2],
            edgecolor="#BBBBBB",
            linewidth=0.5,
            label="Kept only by TAPE",
        ),
        Patch(facecolor=palette[1], edgecolor="#BBBBBB", linewidth=0.5, label="Pruned by both"),
        Patch(
            facecolor=palette[3],
            edgecolor="#BBBBBB",
            linewidth=0.5,
            label="Kept only by matched-capacity REAP",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.166),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=2.0,
        labelspacing=0.6,
    )
    paths = save_figure(fig, output_dir, "fig04_expert_selection")
    plt.close(fig)
    return paths
