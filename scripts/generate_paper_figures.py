#!/usr/bin/env python3
"""Generate the two retained TAPE paper figures, without running any models."""

# Plotting helpers follow the script-path and font-cache setup.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tape-matplotlib-cache"))

from figure_common import configure_style
from tape_expert_matrix import draw_expert_matrix, expert_csv_path
from tape_framework import draw_framework

EXPERT_CSV = expert_csv_path(PROJECT_ROOT).relative_to(PROJECT_ROOT)


def make_overview(images: list[Path], output_dir: Path) -> Path:
    from PIL import Image

    cell_width, cell_height = 1200, 1050
    overview = Image.new("RGB", (2 * cell_width, cell_height), "#E9EEF0")
    for index, path in enumerate(images):
        with Image.open(path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((cell_width - 30, cell_height - 30), Image.Resampling.LANCZOS)
        x = index * cell_width + (cell_width - preview.width) // 2
        y = (cell_height - preview.height) // 2
        overview.paste(preview, (x, y))
    path = output_dir / "all_figures_preview.png"
    overview.save(path)
    return path


def create_package(output_dir: Path, artifacts: list[Path]) -> Path:
    package = output_dir / "tape_paper_figures.zip"
    files = [*artifacts, PROJECT_ROOT / "docs/figures/README.md"]
    files.extend(
        PROJECT_ROOT / "scripts/figures" / name
        for name in ("figure_common.py", "tape_framework.py", "tape_expert_matrix.py")
    )
    files.extend(
        (
            PROJECT_ROOT / "scripts/generate_paper_figures.py",
            PROJECT_ROOT / "requirements-figures.txt",
            PROJECT_ROOT / EXPERT_CSV,
        )
    )
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as archive:
        for path in dict.fromkeys(files):
            if not path.is_file():
                raise FileNotFoundError(f"Missing package input: {path}")
            relative = (
                Path("docs/figures") / path.relative_to(output_dir)
                if path.is_relative_to(output_dir)
                else path.relative_to(PROJECT_ROOT)
            )
            archive.write(path, arcname=str(relative))
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs/figures")
    parser.add_argument("--package", action="store_true", help="Also create a portable ZIP bundle")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    artifacts = draw_framework(output_dir)
    artifacts.extend(draw_expert_matrix(PROJECT_ROOT, output_dir))
    artifacts.append(make_overview([p for p in artifacts if p.suffix == ".png"], output_dir))
    manifest = {
        "revision": 6,
        "generated_without_model_execution": True,
        "figure_count": 2,
        "retained_figure_ids": [1, 4],
        "figure_1": {
            "style": "circular expert nodes and shared-representation connections",
            "illustrative_layer_counts": {
                "original": [8, 8, 8, 8],
                "uniform": [6, 6, 6, 6],
                "tape": [7, 5, 6, 6],
            },
            "experimental_model_dimensions": False,
            "tape_subtitle": "75%",
        },
        "palette_source": {
            "figure_1": {
                "name": "Paul Tol high-contrast; blue/gold outlines",
                "url": "https://sronpersonalpages.nl/~pault/",
            },
            "figure_4": {
                "name": "ColorBrewer Pastel1; light green/light orange fills",
                "url": "https://colorbrewer2.org/export/colorbrewer.json",
            },
        },
        "figure_4_palette": {
            "kept_by_both": "#F2F2F2",
            "pruned_by_both": "#FFFFFF",
            "tape_only": "#CCEBC5",
            "matched_capacity_reap_only": "#FED9A6",
        },
        "figure_4_below_legend_annotations": False,
        "figure_4_numeric_annotations": False,
        "source_expert_csv_sha256": hashlib.sha256(
            (PROJECT_ROOT / EXPERT_CSV).read_bytes()
        ).hexdigest(),
        "files": [
            {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in artifacts
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    artifacts.append(manifest_path)
    if args.package:
        artifacts.append(create_package(output_dir, artifacts))
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
