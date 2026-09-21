"""Draw native AMR preparation neighborhoods for Figure 3."""

from _paths import OUTPUT
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from _slice import (
    COORDINATE,
    MM,
    LEVEL_COLORS,
    EDGE_COLOR,
)

OUT = OUTPUT / "supporting"
PATCH_LEAF_IDS = (11765, 11947, 11949)
INK = EDGE_COLOR
INTERFACE = "#a46920"
TRANSFER_COLORS = ("#b65e2f", "#76539b", "#2675a0")
PANEL_WIDTH = 6.1
PANEL_HEIGHT = 6.65
DETAIL_WINDOWS = (
    ((6, 5), (9, 8)),
    ((7, 0), (10, 3)),
    ((9.5, 3), (11.5, 5)),
)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def axes_inches(figure, left, bottom, width, height):
    fw, fh = figure.get_size_inches()
    return figure.add_axes((left / fw, bottom / fh, width / fw, height / fh))


def text_inches(figure, x, y, text, **kwargs):
    fw, fh = figure.get_size_inches()
    return figure.text(x / fw, y / fh, text, **kwargs)


def patch_rows(geometry):
    """Resolve and check the three identified, adjacent native leaf sections."""
    lookup = {int(leaf): row for row, leaf in enumerate(geometry.leaf_ids)}
    rows = np.asarray([lookup[leaf] for leaf in PATCH_LEAF_IDS])
    boxes = geometry.bounds[rows] * MM
    coarse = boxes[0]
    extent = coarse[1] - coarse[0]
    expected = np.asarray(
        (
            coarse,
            (coarse[0] + (extent[0], 0), coarse[1] + (extent[0] / 2, -extent[1] / 2)),
            (coarse[0] + (extent[0], extent[1] / 2), coarse[1] + (extent[0] / 2, 0)),
        )
    )
    if not np.allclose(boxes, expected, rtol=0, atol=1e-12):
        raise ValueError(
            "selected native leaves do not form the expected adjacent patch"
        )
    if not np.array_equal(geometry.levels[rows], (4, 5, 5)):
        raise ValueError("selected patch must contain native levels L4, L5, L5")
    if geometry.block_shape != (8, 8):
        raise ValueError("the WENO509 patch must retain 8 x 8 native cells per block")
    return rows


def patch_bounds(geometry, rows):
    boxes = geometry.bounds[rows] * MM
    return np.asarray((boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)))


def local_positions(geometry, rows, coordinates):
    """Map coarse-cell coordinates to the original physical slice positions."""
    coarse = geometry.bounds[rows[0]] * MM
    spacing = geometry.spacing[rows[0]] * MM
    return coarse[0] + spacing * np.asarray(coordinates)


def draw_native_grid(ax, geometry, rows, *, marker_size=4.3):
    """Draw original cell edges, centers, and leaf boundaries on any viewport."""
    coarse = geometry.bounds[rows[0]] * MM
    all_centers = []
    for row in rows:
        edges = [axis_edges * MM for axis_edges in geometry.cell_edges(int(row))]
        lo, hi = geometry.bounds[row] * MM
        color = LEVEL_COLORS[int(geometry.levels[row]) - 3]
        ax.add_patch(
            Rectangle(lo, *(hi - lo), facecolor=color, edgecolor="none", zorder=1)
        )
        segments = [((x, lo[1]), (x, hi[1])) for x in edges[0][1:-1]]
        segments += [((lo[0], y), (hi[0], y)) for y in edges[1][1:-1]]
        ax.add_collection(
            LineCollection(segments, colors=INK, linewidths=0.38, alpha=0.55, zorder=2)
        )
        centers = [(e[:-1] + e[1:]) / 2 for e in edges]
        xx, yy = np.meshgrid(*centers)
        all_centers.append(np.column_stack((xx.ravel(), yy.ravel())))
        ax.scatter(
            xx.ravel(), yy.ravel(), s=marker_size, color=INK, alpha=0.7, zorder=3
        )
        ax.add_patch(
            Rectangle(
                lo, *(hi - lo), fill=False, edgecolor=INK, linewidth=1.2, zorder=4
            )
        )
    ax.plot((coarse[1, 0],) * 2, coarse[:, 1], color=INTERFACE, linewidth=2.0, zorder=5)
    return np.concatenate(all_centers)


def draw_patch(ax, geometry, rows):
    """Draw actual native cell edges and centers without resampling."""
    bounds = patch_bounds(geometry, rows)
    spacing = geometry.spacing[rows[0]] * MM
    all_centers = draw_native_grid(ax, geometry, rows)

    def physical(point):
        return local_positions(geometry, rows, point)

    # Numbered areas locate transfer classes on the actual native cell grid.
    cases = (
        ((7, 6), (1, 1), (6.25, 6.5)),
        ((8, 1), (1, 1), (9.7, 1.5)),
        ((10, 3.5), (1, 1), (11.55, 4)),
    )
    for number, ((origin, extent, label), color) in enumerate(
        zip(cases, TRANSFER_COLORS), 1
    ):
        ax.add_patch(
            Rectangle(
                physical(origin),
                *(spacing * extent),
                facecolor="white",
                alpha=0.8,
                edgecolor=color,
                linewidth=1.4,
                zorder=6,
            )
        )
        local = (all_centers - physical(origin)) / spacing
        inside = np.all((local > 0) & (local < extent), axis=1)
        ax.scatter(*all_centers[inside].T, s=7, color=color, zorder=7)
        if number > 1:
            lower = physical(origin)
            upper = physical(np.asarray(origin) + extent)
            middle = (lower + upper) / 2
            ax.plot(
                (middle[0], middle[0]),
                (lower[1], upper[1]),
                color=INK,
                linewidth=0.38,
                alpha=0.7,
                zorder=7,
            )
            ax.plot(
                (lower[0], upper[0]),
                (middle[1], middle[1]),
                color=INK,
                linewidth=1.2 if number == 3 else 0.38,
                alpha=0.7,
                zorder=7,
            )
        ax.annotate(
            str(number),
            xy=physical(np.asarray(origin) + np.asarray(extent) / 2),
            xytext=physical(label),
            ha="center",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": color, "lw": 1},
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.8,
                "shrinkA": 3,
                "shrinkB": 7,
            },
            zorder=8,
        )

    # An ordinary centered neighborhood lies entirely inside the coarse block.
    center = np.asarray((2.5, 4.5))
    for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        pair = np.asarray((physical(center), physical(center + delta)))
        ax.plot(pair[:, 0], pair[:, 1], color=INK, linewidth=1.0, zorder=5)
    ax.annotate(
        "Local stencil",
        xy=physical((2.5, 5.5)),
        xytext=physical((2.5, 6.7)),
        ha="center",
        va="center",
        color=INK,
        fontsize=9,
        bbox={"fc": LEVEL_COLORS[1], "ec": "none", "pad": 2},
        arrowprops={
            "arrowstyle": "-",
            "color": INK,
            "lw": 0.7,
            "shrinkA": 3,
            "shrinkB": 4,
        },
        zorder=8,
    )

    ax.set(
        xlim=bounds[:, 0],
        ylim=bounds[:, 1],
        aspect="equal",
        xlabel="x [Mm]",
        ylabel="y [Mm]",
    )
    ax.set_xticks((-5, -4, -3, -2.5))
    ax.set_yticks((0, 0.5, 1, 1.5))
    ax.tick_params(labelsize=9, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def detail_centers(kind):
    """Return the depicted native and halo centers in coarse-cell coordinates."""
    if kind == 0:
        native = ((7.5, 6.5),)
        halo = ((7.25, 6.25), (7.75, 6.25), (7.25, 6.75), (7.75, 6.75))
    elif kind == 1:
        native = ((8.25, 1.25), (8.75, 1.25), (8.25, 1.75), (8.75, 1.75))
        halo = ((8.5, 1.5),)
    else:
        native = ((10.25, 3.75), (10.75, 3.75), (10.25, 4.25), (10.75, 4.25))
        halo = native
    return np.asarray(native), np.asarray(halo)


def draw_transfer(ax, kind, geometry, rows):
    """Enlarge one interface neighborhood with spatially aligned halo centers."""
    color = TRANSFER_COLORS[kind]
    bounds = local_positions(geometry, rows, DETAIL_WINDOWS[kind])
    ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], aspect="equal")
    ax.axis("off")
    draw_native_grid(ax, geometry, rows, marker_size=7)

    def line(start, stop, *, dashed=True, width=0.95):
        points = local_positions(geometry, rows, (start, stop))
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "-",
            zorder=6,
        )

    def box(origin, extent, *, dashed=True):
        spacing = geometry.spacing[rows[0]] * MM
        ax.add_patch(
            Rectangle(
                local_positions(geometry, rows, origin),
                *(spacing * extent),
                fill=False,
                edgecolor=color,
                linewidth=1.15,
                linestyle=(0, (3, 2)) if dashed else "-",
                zorder=6,
            )
        )

    if kind == 0:
        # Two fine halo layers extend leftward from the L5 block into L4 space.
        box((7, 6), (1, 1), dashed=False)
        line((7.5, 6), (7.5, 7))
        line((7, 6.5), (8, 6.5))
    elif kind == 1:
        # The coarse halo cell occupies the same area as these four fine cells.
        box((8, 1), (1, 1))
    else:
        # Neighbor halo centers coincide with original same-level cell centers.
        box((10, 3.5), (1, 0.5))
        box((10, 4), (1, 0.5))

    native, halo = (
        local_positions(geometry, rows, points) for points in detail_centers(kind)
    )
    ax.scatter(*native.T, s=24, facecolors=color, edgecolors=color, zorder=8)
    ax.scatter(
        *halo.T,
        s=62 if kind == 2 else 38,
        facecolors="white",
        edgecolors=color,
        linewidths=1.2,
        zorder=9,
    )
    if kind == 2:
        ax.scatter(*native.T, s=14, facecolors=color, edgecolors=color, zorder=10)
    ax.add_patch(
        Rectangle(
            bounds[0],
            *(bounds[1] - bounds[0]),
            fill=False,
            edgecolor=INK,
            linewidth=0.45,
            clip_on=False,
            zorder=10,
        )
    )


def draw_panel(figure, left, bottom, geometry, rows):
    text_inches(
        figure,
        left + 0.12,
        bottom + 6.43,
        "(b) Local patch and interface preparation",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    ax = axes_inches(figure, left + 0.64, bottom + 2.75, 4.8, 3.2)
    draw_patch(ax, geometry, rows)
    text_inches(
        figure,
        left + 2.24,
        bottom + 6.08,
        "Coarse leaf block · L4",
        ha="center",
        color=INK,
    )
    text_inches(
        figure,
        left + 4.64,
        bottom + 6.08,
        "Fine leaf blocks · L5",
        ha="center",
        color=INK,
    )
    text_inches(
        figure,
        left + 3.04,
        bottom + 2.18,
        r"Native $8 \times 8$ cells per block  ·  $\Delta_c = 2\Delta_f$",
        ha="center",
        fontsize=9,
        color=INK,
    )

    titles = ("1  Coarse → fine", "2  Fine → coarse", "3  Same level")
    for kind, title in enumerate(titles):
        x = left + 0.16 + kind * 1.97
        text_inches(
            figure,
            x + 0.90,
            bottom + 1.93,
            title,
            ha="center",
            fontsize=10,
            color=TRANSFER_COLORS[kind],
        )
        draw_transfer(
            axes_inches(figure, x, bottom + 0.14, 1.8, 1.65), kind, geometry, rows
        )
    return ax


def metadata(geometry, rows):
    return {
        "slice_axis": "z",
        "coordinate_code": COORDINATE,
        "coordinate_Mm": COORDINATE * MM,
        "original_leaf_ids": geometry.leaf_ids[rows].tolist(),
        "native_levels": geometry.levels[rows].tolist(),
        "native_block_shape_3d": list(geometry.mesh.block_shape),
        "native_block_shape_in_slice": list(geometry.block_shape),
        "block_bounds_Mm": (geometry.bounds[rows] * MM).tolist(),
        "cell_spacing_Mm": (geometry.spacing[rows] * MM).tolist(),
        "patch_bounds_Mm": patch_bounds(geometry, rows).tolist(),
        "geometry_provenance": "Original AxisSlice cell edges and leaf identities; no resampling or rotation.",
        "transfer_diagrams": "Spatial neighborhoods with native and halo centers in shared physical coordinates; out-of-plane support omitted.",
        "detail_bounds_Mm": [
            local_positions(geometry, rows, window).tolist()
            for window in DETAIL_WINDOWS
        ],
        "detail_halo_centers_Mm": [
            local_positions(geometry, rows, detail_centers(kind)[1]).tolist()
            for kind in range(3)
        ],
    }


CAPTION = (
    "The marked region of the WENO509 slice at z = 5.052 Mm is enlarged using "
    "its original block bounds, native cell edges, and cell-center positions. "
    "One L4 leaf block adjoins two L5 leaf blocks, each with an 8 x 8 cell "
    "section and a refinement ratio of two. Dark thick lines mark block "
    "boundaries; the gold line marks the coarse–fine interface. An interior "
    "stencil uses uniformly spaced local values. Numbered regions identify "
    "three neighborhoods enlarged below, with native cells and receiving halo "
    "positions kept in the same spatial view. (1) Four fine halo centers "
    "lie within the neighboring coarse cell and align with the adjacent fine "
    "grid. Dashed subdivisions show the fine halo lattice. (2) A coarse halo "
    "center lies at the center of a 2 x 2 group of native fine cells; the "
    "dashed box marks the coarse halo cell footprint. (3) Same-level native "
    "and neighboring halo centers coincide, shown by open rings around filled "
    "markers on both sides of the block boundary. Filled colored markers "
    "denote native source values and open markers denote halo positions "
    "owned by the receiving block. Halo overlays do not alter donor interiors. "
    "Only in-plane geometry is shown, not complete numerical stencils. "
    "Exact-phase preparation uses ratio-two minmod-limited reconstruction, "
    "2 x 2 x 2 averaging in three dimensions, and same-level copies."
)


def save_standalone(geometry, rows):
    figure = plt.figure(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    draw_panel(figure, 0, 0, geometry, rows)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"amr-patch.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.05
        )
    plt.close(figure)
    (OUT / "amr-patch.json").write_text(
        json.dumps(metadata(geometry, rows), indent=2) + "\n"
    )
    (OUT / "amr-patch.txt").write_text(CAPTION + "\n")
