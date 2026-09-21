"""Drawing helpers for the saved native AMR slice in Figure 3."""

from __future__ import annotations

from _paths import data_directory, OUTPUT

DATA = data_directory()

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import Rectangle

from types import SimpleNamespace

OUT = OUTPUT / "supporting"
COORDINATE = 0.5052083333333333
HORIZONTAL_FRACTION = 0.50
MM = 10.0
LEVEL_COLORS = ("#e8e5df", "#b7d4d5", "#619faa", "#176579")
EDGE_COLOR = "#263640"


def polygons(rectangles):
    return np.stack(
        (
            rectangles[:, (0, 2)],
            rectangles[:, (1, 2)],
            rectangles[:, (1, 3)],
            rectangles[:, (0, 3)],
        ),
        axis=1,
    )


def block_rectangles(geometry):
    bounds = geometry.bounds * MM
    return np.column_stack(
        (bounds[:, 0, 0], bounds[:, 1, 0], bounds[:, 0, 1], bounds[:, 1, 1])
    )


def cell_data(result, scale):
    rectangles = []
    values = []
    for row in np.flatnonzero(result.valid):
        first_edges, second_edges = result.geometry.cell_edges(int(row))
        magnitude = np.linalg.norm(result.values[row], axis=-1) * scale
        for first in range(len(first_edges) - 1):
            for second in range(len(second_edges) - 1):
                rectangles.append(
                    (
                        first_edges[first] * MM,
                        first_edges[first + 1] * MM,
                        second_edges[second] * MM,
                        second_edges[second + 1] * MM,
                    )
                )
                values.append(magnitude[first, second])
    return np.asarray(rectangles), np.asarray(values)


def block_segments(rectangles):
    segments = []
    for x0, x1, y0, y1 in rectangles:
        segments.extend(
            (
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            )
        )
    return segments


def draw_level_blocks(ax, blocks, levels, *, alpha=1.0):
    for rectangle, level in zip(blocks, levels):
        x0, x1, y0, y1 = rectangle
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=LEVEL_COLORS[int(level) - 3],
                edgecolor=EDGE_COLOR,
                linewidth=0.24 if level >= 5 else 0.45,
                alpha=alpha,
                antialiased=True,
            )
        )


class SavedGeometry:
    """Read-only geometry used only for drawing the retained slice."""

    def __init__(self, data):
        for key in ("bounds", "spacing", "levels", "leaf_ids"):
            setattr(self, key, data[key])
        self.block_shape = tuple(map(int, data["block_shape"]))
        self.mesh = SimpleNamespace(
            block_shape=tuple(map(int, data["mesh_block_shape"]))
        )
        self.edges = data["cell_edges"]

    def cell_edges(self, row):
        return self.edges[row]


def load_current_slice():
    with np.load(DATA / "figure3/slice.npz", allow_pickle=False) as archive:
        data = dict(archive)
    mesh = SimpleNamespace(lower=data["lower"], upper=data["upper"])
    result = SimpleNamespace(
        values=data["values"], valid=data["valid"], geometry=SavedGeometry(data)
    )
    return mesh, result, float(data["current_scale"])
