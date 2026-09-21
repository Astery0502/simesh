"""Render the official P16 schematic from retained numerical geometry."""

import json
import numpy as np
import matplotlib.pyplot as plt
import _current_geometry as draw

from _paths import data_directory, OUTPUT

DATA_ROOT = data_directory()

DATA = DATA_ROOT / "figure4"


def render_schematic():
    with np.load(DATA / "schematic-geometry.npz", allow_pickle=False) as saved:
        geometry = json.loads(
            str(saved["structure_json"]),
            object_hook=lambda item: (
                saved[item["array"]] if set(item) == {"array"} else item
            ),
        )
    summary = json.loads((DATA / "schematic-summary.json").read_text())
    draw.DISPLAY_Y_MIN = summary["display_y_min_Mm"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure = plt.figure(figsize=(8.6, 6.0))
        axis = figure.add_axes(
            [0.005, 0.015, 0.99, 1.03], projection="3d", computed_zorder=False
        )
        surface = draw.draw_geometry(axis, geometry)
        cax = axis.inset_axes([0.36, 0.265, 0.34, 0.022])
        colorbar = figure.colorbar(
            surface,
            cax=cax,
            orientation="horizontal",
            ticks=[0, 100, 200, 300, 400],
            extend="max",
        )
        colorbar.set_label(r"Native $|J|$ [$\mu$A m$^{-2}$]", fontsize=11, labelpad=2)
        colorbar.ax.tick_params(labelsize=10, pad=2)
        colorbar.outline.set_linewidth(0.5)
        axis.text2D(0.08, 0.84, "(a)", transform=axis.transAxes, fontsize=16)
        figure.savefig(
            OUTPUT / "supporting/schematic.png",
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(figure)
