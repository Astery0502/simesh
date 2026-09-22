"""Display retained lines colored by local curl magnitude over a B3 plane."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from _paths import data_directory

DATA = data_directory()


def build_figure():
    OUT = DATA / "figure6"

    with np.load(OUT / "lines.npz") as d:
        positions = d["positions"]
        offsets = d["offsets"]
        current = d["current"]
        ids = d["ids"]
    with np.load(OUT / "bottom.npz") as d:
        bz = d["bz"]
        lo = d["lower"]
        hi = d["upper"]
        dx = d["spacing"]
        z = float(d["z"])
    LENGTH_MM_PER_CODE = 10.0
    positions = positions * LENGTH_MM_PER_CODE
    lo = lo * LENGTH_MM_PER_CODE
    hi = hi * LENGTH_MM_PER_CODE
    dx = dx * LENGTH_MM_PER_CODE
    z = z * LENGTH_MM_PER_CODE
    viewlo = np.maximum(lo[:2], positions[:, :2].min(0) - 12.0)
    viewhi = np.minimum(hi[:2], positions[:, :2].max(0) + 12.0)
    x = lo[0] + (np.arange(bz.shape[0]) + 0.5) * dx[0]
    y = lo[1] + (np.arange(bz.shape[1]) + 0.5) * dx[1]
    ix = np.flatnonzero((x >= viewlo[0]) & (x <= viewhi[0]))[::3]
    iy = np.flatnonzero((y >= viewlo[1]) & (y <= viewhi[1]))[::3]
    xx, yy = np.meshgrid(x[ix], y[iy], indexing="ij")
    base = bz[np.ix_(ix, iy)]
    bmax = float(np.percentile(np.abs(base), 99.5))
    bnorm = Normalize(-bmax, bmax)
    cmin, cmax = np.percentile(current[current > 0], [1, 99.5])
    cnorm = LogNorm(cmin, cmax)
    # The tube radius is a display parameter, not a physical flux-tube radius.
    TUBE_RADIUS_MM = 0.42
    SIDES = 12
    cmap = LinearSegmentedColormap.from_list(
        "white_gold_orange_brown",
        ["#ffffff", "#fff0cf", "#f5c36b", "#e78324", "#9a4b16", "#4a260d"],
    )
    faces = []
    facecolors = []
    light = np.array([0.6, -0.8, 1.5])
    light /= np.linalg.norm(light)
    for a, b in zip(offsets[:-1], offsets[1:]):
        raw = positions[a:b]
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        keep = np.r_[True, np.diff(distance) > 1e-12]
        raw = raw[keep]
        distance = distance[keep]
        values = current[a:b][keep]
        sample = np.linspace(
            0, distance[-1], min(900, max(24, int(distance[-1] / 0.3) + 1))
        )
        p = np.column_stack(
            [np.interp(sample, distance, raw[:, axis]) for axis in range(3)]
        )
        value = np.interp(sample, distance, values)
        tangent = np.gradient(p, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = np.empty_like(tangent)
        ref = np.eye(3)[np.argmin(np.abs(tangent[0]))]
        normal[0] = np.cross(tangent[0], ref)
        normal[0] /= np.linalg.norm(normal[0])
        for k in range(1, len(p)):
            n = normal[k - 1] - np.dot(normal[k - 1], tangent[k]) * tangent[k]
            if np.linalg.norm(n) < 1e-10:
                n = np.cross(tangent[k], np.eye(3)[np.argmin(np.abs(tangent[k]))])
            normal[k] = n / np.linalg.norm(n)
        binormal = np.cross(tangent, normal)
        angles = np.arange(SIDES) * 2 * np.pi / SIDES
        radial = (
            normal[:, None, :] * np.cos(angles)[None, :, None]
            + binormal[:, None, :] * np.sin(angles)[None, :, None]
        )
        rings = p[:, None, :] + TUBE_RADIUS_MM * radial
        quads = np.stack(
            (
                rings[:-1],
                np.roll(rings[:-1], -1, axis=1),
                np.roll(rings[1:], -1, axis=1),
                rings[1:],
            ),
            axis=2,
        )
        fn = (
            radial[:-1]
            + np.roll(radial[:-1], -1, axis=1)
            + radial[1:]
            + np.roll(radial[1:], -1, axis=1)
        )
        fn /= np.linalg.norm(fn, axis=2)[:, :, None]
        diffuse = np.clip(fn @ light, 0, 1)
        rgba = cmap(cnorm(0.5 * (value[:-1] + value[1:])))[:, None, :]
        rgba = np.broadcast_to(rgba, (*diffuse.shape, 4)).copy()
        rgba[..., :3] *= (0.58 + 0.42 * diffuse)[..., None]
        rgba[..., :3] = np.clip(rgba[..., :3] + 0.10 * diffuse[..., None] ** 18, 0, 1)
        faces.append(quads.reshape(-1, 4, 3))
        facecolors.append(rgba.reshape(-1, 4))
    # Re-render all panels from retained numerical products, keeping text editable.
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
        }
    )
    with np.load(OUT / "maps.npz", allow_pickle=False) as data:
        q = data["q"]
        tw = data["twist"]
    qpalette = LinearSegmentedColormap.from_list(
        "pale_blue_teal_yellow_red",
        [
            (0.0, "#ffffff"),
            (0.25, "#f5f9fd"),
            (0.43, "#e0edf7"),
            (0.57, "#c4e9e6"),
            (0.68, "#fff0ae"),
            (0.84, "#f6a45b"),
            (1.0, "#b2182b"),
        ],
    )
    qpalette.set_bad("#bcbcbc")
    tpalette = plt.get_cmap("RdBu_r").copy()
    tpalette.set_bad("#bcbcbc")
    tmax = float(np.nanpercentile(np.abs(tw), 99.5))
    fig = plt.figure(figsize=(12, 12.5))
    for bounds, values, title, palette, norm, cbar_bounds, label in (
        (
            [0.08, 0.595, 0.37, 0.355],
            q,
            "(a) QSL",
            qpalette,
            Normalize(np.log10(2), 3.2),
            [0.09, 0.535, 0.35, 0.014],
            r"$\log_{10}Q$",
        ),
        (
            [0.57, 0.595, 0.37, 0.355],
            tw,
            "(b) Twist",
            tpalette,
            Normalize(-tmax, tmax),
            [0.58, 0.535, 0.35, 0.014],
            r"$T_w$",
        ),
    ):
        a = fig.add_axes(bounds)
        im = a.imshow(
            values.T,
            origin="lower",
            extent=(lo[0], hi[0], lo[1], hi[1]),
            cmap=palette,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        a.set(xlabel="x [Mm]", ylabel="y [Mm]")
        a.text(
            0.025,
            0.975,
            title[:3],
            transform=a.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
        )
        a.set_xticks([-150, -75, 0, 75, 150])
        a.set_yticks([-150, -75, 0, 75, 150])
        bar = fig.colorbar(
            im,
            cax=fig.add_axes(cbar_bounds),
            orientation="horizontal",
            extend="max" if title.startswith("(a)") else "both",
        )
        bar.set_label(label)
        if title.startswith("(a)"):
            bar.set_ticks([np.log10(2), 1, 2, 3.2])
            bar.set_ticklabels(["0.30", "1.0", "2.0", "3.2"])
        else:
            bar.set_ticks([-2, -1, 0, 1, 2])
    ax = fig.add_axes(
        [0.115, 0.055, 0.77, 0.445], projection="3d", computed_zorder=False
    )
    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, z),
        facecolors=plt.cm.RdBu_r(bnorm(base)),
        rstride=1,
        cstride=1,
        shade=False,
        alpha=0.83,
        linewidth=0,
        antialiased=False,
        zorder=1,
    )
    collection = Poly3DCollection(
        np.concatenate(faces),
        facecolors=np.concatenate(facecolors),
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        zorder=3,
        rasterized=True,
    )
    ax.add_collection3d(collection)
    zlim = (float(lo[2]), float(positions[:, 2].max() + 7.0))
    ax.set(
        xlim=(viewlo[0], viewhi[0]),
        ylim=(viewlo[1], viewhi[1]),
        zlim=zlim,
        xlabel="x [Mm]",
        ylabel="y [Mm]",
        zlabel="z [Mm]",
    )
    ax.set_box_aspect(
        (viewhi[0] - viewlo[0], viewhi[1] - viewlo[1], zlim[1] - zlim[0]), zoom=1.3
    )
    ax.view_init(elev=28, azim=35)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.set_major_locator(plt.MaxNLocator(5))
    ax.text2D(
        0.10, 0.74, "(c)", transform=ax.transAxes, va="top", ha="left", fontsize=14
    )
    for bounds, norm, palette, label in (
        ([0.065, 0.145, 0.018, 0.26], bnorm, "RdBu_r", r"$B_3$"),
        ([0.925, 0.145, 0.018, 0.26], cnorm, cmap, r"$|J|$"),
    ):
        bar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=palette),
            cax=fig.add_axes(bounds),
            orientation="vertical",
            extend="both",
        )
        bar.set_label(label)
        bar.ax.minorticks_off()
        if label == r"$B_3$":
            bar.set_ticks([-150, 0, 150])
            bar.ax.yaxis.set_ticks_position("left")
            bar.ax.yaxis.set_label_position("left")
        else:
            bar.set_ticks([2, 10, 100])
            bar.set_ticklabels(["2", "10", "100"])
    return fig
