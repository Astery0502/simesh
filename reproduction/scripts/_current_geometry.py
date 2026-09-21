"""Draw the saved three-dimensional comparison geometry for Figure 4."""

import matplotlib.patheffects as effects
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgba
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import itertools

COLORS = ("#cf68a7", "#1fbfb4")
INK = "#243746"
DISPLAY_Z_MIN = 1.15
DISPLAY_Y_MIN = -2.9
CURRENT_MAX = 400
CURRENT_GAMMA = 1.5
CONTOUR_LEVELS = (150, 300)
CURRENT_CMAP = LinearSegmentedColormap.from_list(
    "white_yellow_crimson",
    [
        (0, "#ffffff"),
        (0.12, "#ffffff"),
        (0.22, "#ffe37a"),
        (0.42, "#ff9b42"),
        (0.65, "#ef482d"),
        (1, "#aa1230"),
    ],
    N=256,
)


def clip_edges(edges, minimum_z):
    """Truncate context wires without drawing an artificial bottom block face."""
    result = []
    for segment in edges:
        a, b = segment.copy()
        if max(a[2], b[2]) < minimum_z:
            continue
        if a[2] < minimum_z:
            a = a + (b - a) * (minimum_z - a[2]) / (b[2] - a[2])
        elif b[2] < minimum_z:
            b = b + (a - b) * (minimum_z - b[2]) / (a[2] - b[2])
        result.append([a, b])
    return np.asarray(result)


def box_edges(bounds):
    corners = np.array(list(itertools.product(*zip(*bounds))))
    return np.array(
        [
            [a, b]
            for i, a in enumerate(corners)
            for b in corners[i + 1 :]
            if np.count_nonzero(a != b) == 1
        ]
    )


def draw_geometry(ax, geometry, *, compact=False):
    ax.computed_zorder = False
    ax.set_proj_type("persp", focal_length=1.05)
    ax.view_init(elev=24, azim=-118)
    norm = PowerNorm(CURRENT_GAMMA, vmin=0, vmax=CURRENT_MAX)
    cmap = CURRENT_CMAP
    for patch in geometry["current_slices"]:
        surface = Poly3DCollection(
            patch["polygons_Mm"],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            linewidths=0,
            antialiaseds=False,
            rasterized=True,
            zorder=0.5,
        )
        surface.set_array(patch["current_microA_m2"])
        ax.add_collection3d(surface)
        for contour in patch["contours"]:
            ax.add_collection3d(
                Line3DCollection(
                    contour["segments_Mm"],
                    colors="#9d1935",
                    linewidths=0.5 if compact else 0.7,
                    alpha=0.8,
                    zorder=1.5,
                )
            )
    boxes = geometry["context_boxes_Mm"]
    edges = clip_edges(np.concatenate([box_edges(box) for box in boxes]), DISPLAY_Z_MIN)
    ax.add_collection3d(
        Line3DCollection(
            edges,
            colors=to_rgba("#728495", 0.14),
            linewidths=0.5 if compact else 0.7,
            zorder=0.25,
        )
    )
    for index, face in enumerate(geometry["faces"]):
        color = COLORS[index]
        transverse = [axis for axis in range(3) if axis != face["axis"]]
        lo, hi = face["bounds_Mm"][:, transverse]
        vertices = np.empty((4, 3))
        vertices[:, face["axis"]] = face["coordinate_Mm"]
        vertices[:, transverse] = [
            (lo[0], lo[1]),
            (hi[0], lo[1]),
            (hi[0], hi[1]),
            (lo[0], hi[1]),
        ]
        ax.add_collection3d(
            Poly3DCollection(
                [vertices],
                facecolors=to_rgba(color, 0.12),
                edgecolors=to_rgba(color, 0.65),
                linewidths=0.7,
                zorder=2 + index,
            )
        )
    for patch in geometry["current_slices"]:
        for index, face in enumerate(geometry["faces"]):
            selected_edges = []
            for block in patch["block_boundaries"]:
                if block["leaf_id"] != face["coarse_leaf"]:
                    continue
                vertices = block["vertices_Mm"]
                for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                    if (
                        abs(a[face["axis"]] - face["coordinate_Mm"]) < 1e-10
                        and abs(b[face["axis"]] - face["coordinate_Mm"]) < 1e-10
                        and np.linalg.norm(b - a) > 1e-10
                    ):
                        selected_edges.append([a, b])
            if not selected_edges:
                raise ValueError("Selected interface must intersect the current slice")
            crossing = face["crossing_Mm"]
            distances = []
            for a, b in selected_edges:
                fraction = np.clip(
                    np.dot(crossing - a, b - a) / np.dot(b - a, b - a), 0, 1
                )
                distances.append(np.linalg.norm(crossing - (a + fraction * (b - a))))
            assert min(distances) < 1e-10
    endpoints = geometry["endpoints_Mm"]
    ax.plot(*endpoints.T, color="white", lw=5 if compact else 6, zorder=7)
    ax.plot(*endpoints.T, color=INK, lw=2.5 if compact else 3.2, zorder=9)
    ax.scatter(
        *endpoints.T,
        c=INK,
        edgecolors="white",
        linewidths=0.8,
        s=18 if compact else 32,
        depthshade=False,
        zorder=10,
    )
    for index, face in enumerate(geometry["faces"]):
        p = face["crossing_Mm"]
        ax.scatter(
            *p,
            c=COLORS[index],
            edgecolors=INK,
            s=42 if compact else 70,
            marker="s",
            linewidths=1,
            depthshade=False,
            zorder=11,
        )
        if index == 0:
            label_point = p + np.array([-0.55, 0.40, 0.25])
        else:
            label_point = p + np.array([0.18, 0.48, 0.30])
        ax.plot(*np.array([p, label_point]).T, color=COLORS[index], lw=0.9, zorder=12)
        ax.text(
            *label_point,
            face["label"],
            color="#99396f" if index == 0 else "#087e79",
            fontsize=9 if compact else 14,
            fontweight="bold",
            zorder=13,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )
    for interval in geometry["intervals"]:
        fraction = (
            0.5
            * (interval["start_Mm"] + interval["stop_Mm"])
            / geometry["intervals"][-1]["stop_Mm"]
        )
        p = endpoints[0] + fraction * (endpoints[1] - endpoints[0])
        offset = {
            4: np.array([-0.70, 0.25, -0.12]),
            5: np.array([0.18, -0.20, 0]),
            6: np.array([0.25, -0.30, -0.15]),
        }[interval["level"]]
        if not compact:
            ax.text(
                *(p + offset),
                f"L{interval['level']}",
                color=INK,
                fontsize=12.5,
                zorder=14,
                path_effects=[effects.withStroke(linewidth=2.5, foreground="white")],
            )
    low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
    all_corners = np.concatenate([p["corners_Mm"] for p in geometry["current_slices"]])
    low, high = (
        np.minimum(low, all_corners.min(axis=0)),
        np.maximum(high, all_corners.max(axis=0)),
    )
    low[2] = min(DISPLAY_Z_MIN, all_corners[:, 2].min())
    pad = 0.16
    xlim = (low[0] - pad, high[0] + pad)
    ylim = (DISPLAY_Y_MIN, high[1] + pad)
    zlim = (low[2] - 0.08, high[2] + pad)
    assert all_corners[:, 1].min() > DISPLAY_Y_MIN
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    ax.set_box_aspect(
        np.ptp(np.array([xlim, ylim, zlim]), axis=1), zoom=0.92 if compact else 1
    )
    ax.set_xlabel(
        "x [Mm]", labelpad=0 if compact else 9, fontsize=11 if compact else 12.5
    )
    ax.set_ylabel(
        "y [Mm]", labelpad=0 if compact else 9, fontsize=11 if compact else 12.5
    )
    ax.set_zlabel(
        "z [Mm]", labelpad=0 if compact else 9, fontsize=11 if compact else 12.5
    )
    ax.tick_params(labelsize=7 if compact else 11, pad=0 if compact else 2)
    ax.set_xticks([-5, -4, -3, -2])
    ax.set_yticks([-2.5, -2, -1, 0])
    ax.set_zticks([1.5, 2, 2.5, 3])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color("#a9b2ba")
    for index, point in enumerate(geometry["highlights"]):
        p, value = point["position_Mm"], point["current_microA_m2"]
        ax.scatter(
            *p,
            facecolors="white",
            edgecolors=INK,
            s=22 if compact else 42,
            depthshade=False,
            zorder=15,
        )
        x, y, _ = proj3d.proj_transform(*p, ax.get_proj())
        offset = (
            ((8, -27) if index == 0 else (21, -25))
            if compact
            else ((12, -48) if index == 0 else (28, -37))
        )
        text = f"{value:.0f}" if compact else f"{value:.1f}" + r" $\mu$A m$^{-2}$"
        ax.annotate(
            text,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8 if compact else 12.5,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2},
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#647886"},
            zorder=20,
        )
    return surface


def draw_current_colorbar(ax, surface, *, compact=False):
    cax = ax.inset_axes(
        [0.37, 0.265, 0.34, 0.022] if compact else [0.36, 0.265, 0.34, 0.022]
    )
    colorbar = ax.figure.colorbar(
        surface,
        cax=cax,
        orientation="horizontal",
        ticks=[0, 150, 300, 400] if compact else [0, 100, 200, 300, 400],
    )
    colorbar.set_label(
        r"Native $|J|$ [$\mu$A m$^{-2}$], $\gamma=1.5$",
        fontsize=7 if compact else 11,
        labelpad=2,
    )
    colorbar.ax.tick_params(labelsize=6 if compact else 10, pad=2)
    colorbar.outline.set_linewidth(0.5)
    return colorbar
