"""Render Figure 3 from the saved native AMR slice and geometry."""

import json
from _paths import OUTPUT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

import _slice as overview
import _amr_patch as patch

OUT = OUTPUT
PDF_OUT = OUTPUT
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)
FIGURE_STEM = "figure3-native-amr-analysis"


def draw_overview(figure, mesh, result, current_scale, bounds, *, bottom=0):
    """Keep panel (a)'s native data, display crop and current normalization."""
    rectangles, values = overview.cell_data(result, current_scale)
    blocks = overview.block_rectangles(result.geometry)
    levels = result.geometry.levels
    domain = np.asarray((mesh.lower[:2], mesh.upper[:2])) * overview.MM
    x_center = np.mean(domain[:, 0])
    x_half = np.ptp(domain[:, 0]) * overview.HORIZONTAL_FRACTION / 2
    x_limits = (x_center - x_half, x_center + x_half)
    centers = rectangles[:, :2].mean(axis=1)
    finite = (
        (centers >= x_limits[0])
        & (centers <= x_limits[1])
        & np.isfinite(values)
        & (values > 0)
    )
    limits = np.percentile(values[finite], (2, 99.8))
    norm = LogNorm(*limits)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#b9bdc4")

    patch.text_inches(
        figure,
        0.12,
        bottom + 6.43,
        "(a) 3D simulation: current slice and native AMR",
        fontsize=13,
        fontweight="bold",
        color=patch.INK,
    )
    axes = [
        patch.axes_inches(figure, 0.62 + index * 2.4, bottom + 1.15, 2.4, 4.8)
        for index in range(2)
    ]
    for index, (ax, title) in enumerate(
        zip(axes, ("Current density", "Native leaf blocks"))
    ):
        ax.set(xlim=x_limits, ylim=domain[:, 1], xlabel="x [Mm]", aspect="equal")
        ax.set_xticks((-5, -2.5, 0, 2.5, 5))
        if index == 0:
            ax.set_ylabel("y [Mm]")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
        patch.text_inches(
            figure,
            1.82 + index * 2.4,
            bottom + 6.08,
            title,
            ha="center",
            color=patch.INK,
        )

    collection = PolyCollection(
        overview.polygons(rectangles),
        array=np.ma.masked_invalid(values),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
        antialiaseds=False,
    )
    axes[0].add_collection(collection)
    axes[0].add_collection(
        LineCollection(
            overview.block_segments(blocks),
            colors="#e7edf0",
            linewidths=0.24,
            alpha=0.55,
        )
    )
    overview.draw_level_blocks(axes[1], blocks, levels)
    cax = patch.axes_inches(
        figure, 0.62 + (2.4 - 1.824) / 2, bottom + 0.40, 1.824, 0.13
    )
    colorbar = figure.colorbar(
        collection, cax=cax, orientation="horizontal", extend="both"
    )
    colorbar.ax.tick_params(labelsize=11, pad=1.5)
    colorbar.set_label(r"$|J|$ [A m$^{-2}$]", fontsize=12, labelpad=2)
    visible = (blocks[:, 1] > x_limits[0]) & (blocks[:, 0] < x_limits[1])
    handles = [
        Patch(
            facecolor=overview.LEVEL_COLORS[int(level) - 3],
            edgecolor=patch.INK,
            label=f"L{level}",
        )
        for level in np.unique(levels[visible])
    ]
    fw, fh = figure.get_size_inches()
    figure.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(4.22 / fw, (bottom + 0.465) / fh),
        ncols=3,
        frameon=False,
        columnspacing=1,
        handlelength=1.3,
        handleheight=0.7,
        handletextpad=0.65,
        fontsize=12,
    )

    # This box exactly encloses the three original leaves enlarged in (b).
    lo, hi = bounds
    for axis in axes:
        roi = Rectangle(
            lo, *(hi - lo), fill=False, edgecolor=patch.INTERFACE, linewidth=1.5, zorder=10
        )
        roi.set_path_effects(
            [path_effects.Stroke(linewidth=3.3, foreground="white"), path_effects.Normal()]
        )
        axis.add_patch(roi)
        axis.annotate(
            "(b)", xy=(lo[0], hi[1]), xytext=(0, 5), textcoords="offset points",
            color=patch.INK, fontsize=12, ha="left", va="bottom", zorder=11,
            bbox={"fc": "white", "ec": "none", "pad": 1.2},
        )
    return axes, {
        "color_limits_A_m2": limits.tolist(),
        "displayed_x_limits_Mm": list(x_limits),
        "native_cells_rendered": len(values),
    }


def main():
    patch.style()
    mesh, result, current_scale = overview.load_current_slice()
    geometry = result.geometry
    rows = patch.patch_rows(geometry)
    bounds = patch.patch_bounds(geometry, rows)
    figure = plt.figure(figsize=(12.05, patch.PANEL_HEIGHT))
    axes, summary = draw_overview(
        figure, mesh, result, current_scale, bounds
    )
    zoom = patch.draw_panel(figure, 5.85, 0, geometry, rows)
    for source, target in (
        ((bounds[1, 0], bounds[1, 1]), (0, 1)),
        ((bounds[1, 0], bounds[0, 1]), (0, 0)),
    ):
        options = dict(
            xyA=source,
            coordsA="data",
            axesA=axes[1],
            xyB=target,
            coordsB="axes fraction",
            axesB=zoom,
            color=patch.INTERFACE,
            linewidth=0.7,
            linestyle=(0, (3, 3)),
            alpha=0.6,
        )
        # Keep zoom guides behind labels, with a clipped overlay inside (a).
        figure.add_artist(ConnectionPatch(**options, clip_on=False, zorder=0))
        local_guide = ConnectionPatch(**options, clip_on=True, zorder=9)
        local_guide.set_clip_path(axes[1].patch)
        axes[1].add_artist(local_guide)
    for label in zoom.get_yticklabels():
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            (PDF_OUT if suffix == "pdf" else OUT) / f"{FIGURE_STEM}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(figure)
    summary.update(patch.metadata(geometry, rows))
    summary["figure_number"] = 3
    summary["purpose"] = (
        "Native AMR structure and ghost-cell reconstruction"
    )
    (OUT / f"{FIGURE_STEM}.json").write_text(json.dumps(summary, indent=2) + "\n")
    caption = (
        "Figure 3. Native-AMR structure and ghost-cell reconstruction. "
        "(a) Current density magnitude and the matching native leaf-block "
        "structure on a slice at z = 5.052 Mm, selected to visualize the AMR "
        "structure of the three-dimensional WENO509 simulation. Current density is evaluated "
        "as norm(curl(B))/mu0 on native AMR cells using exact-phase "
        "preparation and solar MHD units. Both views retain the full y "
        "extent and the central half of x; logarithmic color limits use "
        "the 2nd and 99.8th percentiles of positive finite current values. "
        "The outlined patch spans x = [-5, -2.5] Mm and y = [0, 5/3] Mm. "
        "(b) " + patch.CAPTION
    )
    (OUT / f"{FIGURE_STEM}.txt").write_text(caption + "\n")
    patch.save_standalone(geometry, rows)
    print(
        "Rendered Figure 3 with panels (a) and (b), standalone details, and numerical provenance."
    )


if __name__ == "__main__":
    main()
