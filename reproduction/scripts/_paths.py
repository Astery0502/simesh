"""Resolve separately supplied inputs and the rendering destination."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("SIMESH_FIGURE_OUTPUT", ROOT / "output")).resolve()
inputs = Path(os.environ.get("SIMESH_FIGURE_DATA", ROOT / "data")).resolve()
if OUTPUT == inputs or inputs in OUTPUT.parents:
    raise ValueError("The output directory must be outside the data directory")
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "supporting").mkdir(exist_ok=True)


def data_directory():
    """Return the selected input root, defaulting to the adjacent data directory."""
    if not inputs.is_dir():
        raise ValueError(
            "Place plotting inputs in reproduction/data/ or use render.py --data DIR"
        )
    return inputs
