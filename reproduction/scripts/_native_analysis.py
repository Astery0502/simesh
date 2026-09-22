"""Draw saved scientific products for Figure 3; no simulation is read."""

import json
from _paths import data_directory, OUTPUT

DATA = data_directory()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import _amr_patch as patch
import _slice as overview

OUT = OUTPUT / "supporting"
STEM = "native-analysis"
VOLUME_LEAVES = np.asarray((11765, 11947, 11949, 11951, 11953), dtype=np.int64)
COLORS = ("#76539b", "#bc632f", "#377a55", "#2675a0")
WIDTH, HEIGHT = 12.05, patch.PANEL_HEIGHT


def inside(points, bounds):
    return np.all((points >= bounds[0]) & (points <= bounds[1]), axis=-1)


def load_products():
    with np.load(DATA / "figure3/native-analysis.npz") as archive:
        data = dict(archive)
    summary = json.loads((DATA / "figure3/native-analysis.json").read_text())
    return data, summary


def box_faces(bounds):
    lo, hi = bounds
    return [
        np.asarray(
            (
                (lo[0], lo[1], lo[2]),
                (hi[0], lo[1], lo[2]),
                (hi[0], hi[1], lo[2]),
                (lo[0], hi[1], lo[2]),
            )
        ),
        np.asarray(
            (
                (lo[0], lo[1], hi[2]),
                (hi[0], lo[1], hi[2]),
                (hi[0], hi[1], hi[2]),
                (lo[0], hi[1], hi[2]),
            )
        ),
        np.asarray(
            (
                (lo[0], lo[1], lo[2]),
                (hi[0], lo[1], lo[2]),
                (hi[0], lo[1], hi[2]),
                (lo[0], lo[1], hi[2]),
            )
        ),
        np.asarray(
            (
                (lo[0], hi[1], lo[2]),
                (hi[0], hi[1], lo[2]),
                (hi[0], hi[1], hi[2]),
                (lo[0], hi[1], hi[2]),
            )
        ),
        np.asarray(
            (
                (lo[0], lo[1], lo[2]),
                (lo[0], hi[1], lo[2]),
                (lo[0], hi[1], hi[2]),
                (lo[0], lo[1], hi[2]),
            )
        ),
        np.asarray(
            (
                (hi[0], lo[1], lo[2]),
                (hi[0], hi[1], lo[2]),
                (hi[0], hi[1], hi[2]),
                (hi[0], lo[1], hi[2]),
            )
        ),
    ]


def box_edges(bounds):
    lo, hi = bounds
    edges = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for a in (0, 1):
            for b in (0, 1):
                start, stop = lo.copy(), lo.copy()
                start[other] = stop[other] = (bounds[a, other[0]], bounds[b, other[1]])
                stop[axis] = hi[axis]
                edges.append((start, stop))
    return edges


def draw_volume(ax, data, norm):
    bounds = data["bounds_Mm"]
    grid = []
    for block, level, spacing in zip(
        data["block_bounds_Mm"], data["block_levels"], data["block_spacing_Mm"]
    ):
        lo, hi = block
        faces = box_faces(block)
        visible_faces = []
        for axis, side, face in ((1, 0, 2), (0, 1, 5), (2, 1, 1)):
            if not np.isclose(block[side, axis], bounds[side, axis]):
                continue
            visible_faces.append(faces[face])
            tangents = [a for a in range(3) if a != axis]
            for along in tangents:
                across = next(a for a in tangents if a != along)
                for offset in range(1, 8):
                    start, stop = lo.copy(), hi.copy()
                    start[axis] = stop[axis] = block[side, axis]
                    start[across] = stop[across] = lo[across] + spacing[across] * offset
                    grid.append((start, stop))
        ax.add_collection3d(
            Poly3DCollection(
                visible_faces,
                facecolors=overview.LEVEL_COLORS[int(level) - 3],
                edgecolors="none",
                alpha=0.06,
                zorder=1,
            )
        )
        ax.add_collection3d(
            Line3DCollection(
                box_edges(block), colors=patch.INK, linewidths=0.8, alpha=0.7, zorder=7
            )
        )
    ax.add_collection3d(
        Line3DCollection(grid, colors="#64858b", linewidths=0.3, alpha=0.48, zorder=6)
    )

    rectangles = data["native_rectangles_Mm"]
    xy = overview.polygons(rectangles)
    plane = np.concatenate(
        (xy, np.full((*xy.shape[:-1], 1), data["plane_z_Mm"])), axis=-1
    )
    ax.add_collection3d(
        Poly3DCollection(
            plane,
            facecolors=plt.get_cmap("magma")(norm(data["native_current_A_m2"] * 1e6)),
            edgecolors="#f4efea",
            linewidths=0.14,
            alpha=0.94,
            zorder=4,
        )
    )

    region = data["region_Mm"]
    ax.add_collection3d(
        Poly3DCollection(
            box_faces(region),
            facecolors=COLORS[2],
            edgecolors="none",
            alpha=0.045,
            zorder=5,
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            box_edges(region),
            colors=COLORS[2],
            linewidths=1.0,
            linestyles="dashed",
            alpha=0.85,
            zorder=9,
        )
    )
    ax.text(*region[1], r"$\Omega$", color=COLORS[2], fontsize=11, zorder=15)

    for index, (start, stop) in enumerate(
        zip(data["ray_start_Mm"], data["ray_end_Mm"])
    ):
        ray = np.asarray((start, stop))
        ax.plot(
            *ray.T,
            color=COLORS[3],
            linewidth=1.3 if index == 1 else 0.85,
            alpha=1.0 if index == 1 else 0.7,
            zorder=10,
        )
    ax.text(
        *(data["ray_end_Mm"][1] + (0.05, 0.02, 0.04)),
        "r",
        color=COLORS[3],
        fontsize=10,
        zorder=15,
    )
    for branch, (start, stop) in enumerate(
        zip(data["line_offsets"][:-1], data["line_offsets"][1:])
    ):
        points = data["line_positions_Mm"][start:stop][data["line_inside"][start:stop]]
        ax.plot(
            *points.T,
            color=COLORS[1],
            linewidth=1.8 if branch // 2 == 1 else 1.0,
            alpha=1.0 if branch // 2 == 1 else 0.65,
            zorder=12,
        )
    seed = data["seed_positions_Mm"][1]
    ax.scatter(*seed, s=13, color=COLORS[1], depthshade=False, zorder=14)
    ax.text(
        *(seed + (0.03, -0.10, 0.07)),
        r"$\ell$",
        color=COLORS[1],
        fontsize=12,
        zorder=15,
    )
    point = data["point_Mm"]
    ax.scatter(
        *point,
        s=30,
        color=COLORS[0],
        edgecolors="white",
        linewidths=0.7,
        depthshade=False,
        zorder=14,
    )
    ax.text(
        *(point + (0.04, -0.07, 0.04)), "P", color=COLORS[0], fontsize=10, zorder=15
    )

    ax.set(
        xlim=bounds[:, 0],
        ylim=bounds[:, 1],
        zlim=bounds[:, 2],
        xlabel="x [Mm]",
        ylabel="y [Mm]",
        zlabel="z [Mm]",
    )
    ax.set_box_aspect(bounds[1] - bounds[0])
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks((-5, -4, -3))
    ax.set_yticks((0, 0.5, 1, 1.5))
    ax.set_zticks((5, 5.5, 6, 6.5))
    ax.tick_params(labelsize=9, pad=0)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.label.set_fontsize(10)
        axis.labelpad = 5
        axis.line.set_color("#8b9b9d")


def scientific_label(value):
    exponent = int(np.floor(np.log10(abs(value))))
    return rf"{value / 10**exponent:.2f}\times10^{{{exponent}}}"


def draw_panel(figure, data, *, left=0, bottom=0):
    patch.text_inches(
        figure,
        left + 0.12,
        bottom + 6.43,
        "(c) Native analysis on the local AMR volume",
        fontsize=11,
        fontweight="bold",
        color=patch.INK,
    )
    combined_j = (
        np.r_[data["plane_current_A_m2"].ravel(), data["native_current_A_m2"]] * 1e6
    )
    norm = Normalize(float(combined_j.min()), float(combined_j.max()))
    fw, fh = figure.get_size_inches()
    volume = figure.add_axes(
        ((left + 0.05) / fw, (bottom + 0.95) / fh, 6.55 / fw, 5.10 / fh),
        projection="3d",
        computed_zorder=False,
    )
    draw_volume(volume, data, norm)
    patch.text_inches(
        figure,
        left + 3.25,
        bottom + 6.05,
        "One L4 block and four L5 blocks",
        ha="center",
        color=patch.INK,
    )
    figure.legend(
        handles=(
            Line2D(
                [],
                [],
                marker="o",
                color=COLORS[0],
                linestyle="none",
                markersize=4,
                label="1  Point / plane",
            ),
            Line2D([], [], color=COLORS[1], linewidth=1.5, label="2  Field lines"),
            Line2D(
                [], [], color=COLORS[2], linestyle="dashed", label="3  Selected region"
            ),
            Line2D([], [], color=COLORS[3], linewidth=1.1, label="4  LOS rays"),
        ),
        loc="center",
        bbox_to_anchor=((left + 3.25) / fw, (bottom + 0.62) / fh),
        ncols=2,
        frameon=False,
        fontsize=9,
        handlelength=1.7,
        columnspacing=2.0,
        labelspacing=0.7,
    )

    bounds = data["bounds_Mm"]
    titles = (
        "1  Sampling & diagnostics",
        "2  Along-line profile",
        "3  Regional statistics",
        "4  LOS column density",
    )
    positions = ((7.05, 3.91), (9.79, 3.91), (7.05, 1.04), (9.79, 1.04))
    axes = []
    for number, ((x, y), title) in enumerate(zip(positions, titles)):
        ax = patch.axes_inches(figure, left + x, bottom + y, 2.0, 1.72)
        ax.tick_params(labelsize=8.5, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        patch.text_inches(
            figure,
            left + x + 1,
            bottom + y + 1.88,
            title,
            ha="center",
            fontsize=10,
            color=COLORS[number],
        )
        axes.append(ax)

    ax = axes[0]
    extent = (bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1])
    picture = ax.imshow(
        data["plane_current_A_m2"].T * 1e6,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    for block in data["block_bounds_Mm"]:
        if block[0, 2] <= data["plane_z_Mm"] < block[1, 2]:
            rect = plt.Rectangle(
                block[0, :2],
                *(block[1, :2] - block[0, :2]),
                fill=False,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.add_patch(rect)
    ax.plot(
        *data["point_Mm"][:2],
        "o",
        color=COLORS[0],
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )
    ax.text(
        data["point_Mm"][0] + 0.06,
        data["point_Mm"][1] + 0.07,
        "P",
        color="white",
        fontsize=9,
    )
    ax.set(xlabel="x [Mm]", ylabel="y [Mm]")
    ax.set_xticks((-5, -4, -3))
    ax.set_yticks((0, 0.5, 1, 1.5))
    cax = patch.axes_inches(figure, left + 7.08, bottom + 3.49, 1.94, 0.075)
    colorbar = figure.colorbar(picture, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=8, pad=1)
    colorbar.set_label(r"$|J|$ [$\mu$A m$^{-2}$]", fontsize=9, labelpad=2)

    ax = axes[1]
    distance, values = data["profile_arclength_Mm"], data["profile_B_gauss"]
    ax.axvspan(
        distance.min(), 0, color=overview.LEVEL_COLORS[2], alpha=0.15, linewidth=0
    )
    ax.axvspan(
        0, distance.max(), color=overview.LEVEL_COLORS[1], alpha=0.18, linewidth=0
    )
    ax.axvline(0, color=patch.INTERFACE, linewidth=0.8)
    ax.plot(distance, values, color=COLORS[1], linewidth=1.5)
    ax.text(0.06, 0.08, "L5", transform=ax.transAxes, fontsize=8.5, color=patch.INK)
    ax.text(0.83, 0.08, "L4", transform=ax.transAxes, fontsize=8.5, color=patch.INK)
    ax.set(xlabel="s [Mm]", ylabel=r"$|B|$ [G]", xlim=(distance.min(), distance.max()))
    ax.grid(axis="y", color="#e3e8e8", linewidth=0.5)

    ax = axes[2]
    ax.stairs(
        data["histogram_volume_fraction"],
        data["histogram_edges_gauss"],
        fill=True,
        color=COLORS[2],
        alpha=0.65,
        linewidth=0.8,
    )
    ax.set(xlabel=r"$|B|$ [G]", ylabel="Volume fraction")
    ax.set_ylim(0, max(data["histogram_volume_fraction"]) * 1.35)
    ax.text(
        0.04,
        0.95,
        r"$E_B=" + scientific_label(float(data["region_energy_J"])) + r"$ J",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=patch.INK,
    )
    ax.grid(axis="y", color="#e3e8e8", linewidth=0.5)

    ax = axes[3]
    extent = (bounds[0, 1], bounds[1, 1], bounds[0, 2], bounds[1, 2])
    picture = ax.imshow(
        data["column_density_g_cm2"].T * 1e7,
        origin="lower",
        extent=extent,
        cmap="cividis",
        interpolation="nearest",
        aspect="equal",
    )
    ax.plot(
        data["ray_start_Mm"][:, 1],
        data["ray_start_Mm"][:, 2],
        "o",
        markersize=3.2,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.text(
        data["ray_start_Mm"][1, 1] + 0.08,
        data["ray_start_Mm"][1, 2] + 0.04,
        "r",
        color="white",
        fontsize=9,
    )
    ax.set(xlabel="y [Mm]", ylabel="z [Mm]")
    ax.set_xticks((0, 0.5, 1, 1.5))
    ax.set_yticks((5, 5.5, 6, 6.5))
    cax = patch.axes_inches(figure, left + 9.82, bottom + 0.54, 1.94, 0.075)
    colorbar = figure.colorbar(picture, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=8, pad=1)
    colorbar.set_label(r"$\Sigma$ [$10^{-7}$ g cm$^{-2}$]", fontsize=9, labelpad=2)
    return volume


CAPTION = (
    "Native analyses of the same local AMR neighborhood in three dimensions. "
    "The volume spans x = [-5, -2.5], y = [0, 5/3], and z = [5, 20/3] Mm "
    "and contains one L4 and four L5 leaves, each with 8 x 8 x 8 native cells. "
    "The colored native-cell plane is the z = 5.052 Mm section used above. "
    "(1) Current magnitude, computed by native centered curl and explicitly "
    "normalized in SI, is sampled on a 96 x 64 local plane and at P. "
    "(2) The highlighted field line provides a sampled magnetic-strength "
    "profile across the L5/L4 interface; s = 0 is its seed. The three field "
    "lines are actual RK4 traces, with only their stored in-volume samples "
    "displayed. Integration was limited to 1.5 Mm per branch on explicitly "
    "prepared surrounding leaves; endpoints are not physical footpoints. "
    "(3) The selected region Omega is reduced using exact native-cell overlap "
    "volumes, yielding a volume-weighted field-strength distribution and "
    "magnetic energy. (4) An 80 x 80 set of x-directed rays is integrated "
    "only over the 2.5 Mm local depth to obtain mass column density. The "
    "three displayed rays correspond to marked image pixels. All results "
    "use simesh native consumers and the solar unit factors of panel (a); "
    "the LOS image is a density integral, with no assumed temperature or "
    "thermal response. Output pixels sample local native fields without "
    "preparing a globally uniform volume."
)


def save_standalone(data):
    figure = plt.figure(figsize=(WIDTH, HEIGHT))
    draw_panel(figure, data)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"{STEM}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.05
        )
    plt.close(figure)
    (OUT / f"{STEM}.txt").write_text(CAPTION + "\n")
