"""Show expert pruning as sparse neural networks with exact toy budgets.

Large circles are experts; small junctions denote shared representations between
MoE layers. The schematic is illustrative, not a model architecture or result.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from figure_common import save_figure
from matplotlib.patches import Circle


def example_masks() -> dict[str, list[list[int]]]:
    """Return the exact illustrative masks, all with scattered expert removal."""
    original = [[1] * 8 for _ in range(4)]
    uniform = [[1] * 8 for _ in range(4)]
    for row, removed in enumerate(((1, 6), (3, 7), (0, 5), (2, 4))):
        for expert in removed:
            uniform[row][expert] = 0
    tape = [
        [1, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 0, 1],
    ]
    assert tuple(map(sum, original)) == (8, 8, 8, 8)
    assert tuple(map(sum, uniform)) == (6, 6, 6, 6)
    assert tuple(map(sum, tape)) == (7, 5, 6, 6)
    assert sum(map(sum, original)) == 32
    assert sum(map(sum, uniform)) == sum(map(sum, tape)) == 24
    return {"Original": original, "Uniform": uniform, "TAPE": tape}


def draw_framework(output_dir: Path) -> list[Path]:
    """Save a neural-network-style pruning comparison as PDF, SVG, and PNG."""
    # Paul Tol's high-contrast palette: https://sronpersonalpages.nl/~pault/
    # Published blue/gold accents, with neutral structural elements.
    blue, gold = "#004488", "#DDAA33"
    ink, muted = "#253340", "#66727C"
    pale_blue, ghost, edge_blue = "#EDF3F8", "#CDD2D6", "#A3B9CF"

    fig = plt.figure(figsize=(8.4, 4.4))
    ax = fig.add_axes([0.018, 0.02, 0.964, 0.96])
    ax.set(xlim=(-0.06, 8.39), ylim=(0.0, 4.4), aspect="equal")
    ax.axis("off")

    row_ys = (3.42, 2.62, 1.82, 1.02)
    radius = 0.084

    def expert(x: float, y: float, retained: bool) -> None:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=pale_blue if retained else "#FBFBFB",
                edgecolor=blue if retained else ghost,
                linewidth=1.2 if retained else 0.9,
                linestyle="solid" if retained else (0, (2.3, 1.7)),
                zorder=4,
            )
        )
        if not retained:
            d = radius * 0.42
            for dy in (-d, d):
                ax.plot(
                    [x - d, x + d],
                    [y + dy, y - dy],
                    color=gold,
                    linewidth=1.15,
                    solid_capstyle="round",
                    zorder=5,
                )

    def connection(x1: float, y1: float, x2: float, y2: float, retained: bool) -> None:
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=edge_blue if retained else "#DFE2E5",
            linewidth=0.65 if retained else 0.48,
            linestyle="solid" if retained else (0, (2.1, 2.0)),
            zorder=1,
        )

    masks = example_masks()
    original_count = sum(map(sum, masks["Original"]))
    for left, (title, mask) in zip((0.12, 2.94, 5.76), masks.items(), strict=True):
        xs = [left + 0.37 + 0.235 * column for column in range(8)]
        center = (xs[0] + xs[-1]) / 2
        count_x = left + 2.34
        ax.text(
            center,
            4.15,
            title,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=blue if title == "TAPE" else ink,
        )
        ax.text(
            center,
            3.89,
            (
                f"{sum(map(sum, mask)) / original_count:.0%}"
                if title == "TAPE"
                else f"{sum(map(sum, mask))} experts"
            ),
            ha="center",
            va="center",
            fontsize=9.2,
            color=muted,
        )
        ax.text(count_x, 3.79, "Kept", ha="center", va="center", fontsize=8.1, color=muted)

        # Fan into a shared representation before feeding the following layer;
        # the edges do not imply dense expert-to-expert weight matrices.
        for row in range(3):
            y_top, y_bottom = row_ys[row], row_ys[row + 1]
            y_junction = (y_top + y_bottom) / 2
            for column, x in enumerate(xs):
                connection(x, y_top - radius, center, y_junction, bool(mask[row][column]))
                connection(center, y_junction, x, y_bottom + radius, bool(mask[row + 1][column]))
            ax.add_patch(
                Circle((center, y_junction), 0.022, facecolor="#8796A3", edgecolor="none", zorder=3)
            )

        for row, y in enumerate(row_ys):
            ax.text(
                left - 0.005, y, f"L{row + 1}", ha="left", va="center", fontsize=8.4, color=muted
            )
            for column, x in enumerate(xs):
                expert(x, y, bool(mask[row][column]))
            ax.text(
                count_x,
                y,
                str(sum(mask[row])),
                ha="center",
                va="center",
                fontsize=11.0,
                fontweight="bold" if title == "TAPE" else "normal",
                color=blue if title == "TAPE" else ink,
            )

    key_y = 0.34
    expert(0.57, key_y, True)
    ax.text(0.73, key_y, "Kept expert", va="center", fontsize=8.6, color=muted)
    expert(2.20, key_y, False)
    ax.text(2.36, key_y, "Pruned expert", va="center", fontsize=8.6, color=muted)
    ax.text(8.15, key_y, "Illustrative", ha="right", va="center", fontsize=8.1, color=muted)

    return save_figure(fig, Path(output_dir), "fig01_tape_framework")
