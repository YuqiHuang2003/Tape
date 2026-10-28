"""Shared publication styling and deterministic figure exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

COLORS = {
    "tape": "#007F73",
    "reap": "#4477AA",
    "frequency": "#777777",
    "ean_mean": "#AA4499",
    "hybrid": "#B8860B",
    "original": "#444444",
    "ink": "#20313D",
    "muted": "#61717B",
    "grid": "#E4E9EC",
    "pale_teal": "#EDF7F4",
    "pale_blue": "#F0F5FA",
    "pale_gray": "#F5F7F8",
    "orange": "#D55E00",
    "both_kept": "#D8D8D8",
    "both_pruned": "#F4F4F4",
}


def configure_style() -> None:
    """Use embedded fonts in PDF and editable text in SVG; never require TeX."""
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.edgecolor": "#A5B0B6",
            "axes.linewidth": 0.65,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "tape-paper-figures-v1",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        metadata = None
        if extension == "pdf":
            metadata = {"Creator": "TAPE figure generator", "CreationDate": None}
        elif extension == "svg":
            metadata = {"Creator": "TAPE figure generator", "Date": None}
        fig.savefig(path, bbox_inches="tight", pad_inches=0.10, metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.65)
