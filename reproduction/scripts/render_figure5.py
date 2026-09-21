"""Render x=0 diagnostic maps and 20 spatially separated high-twist field lines."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize, LogNorm, rgb_to_hsv, hsv_to_rgb

from _paths import data_directory, OUTPUT

DATA = data_directory()

CACHE = DATA / "figure5"
OUT = OUTPUT
PDF = OUTPUT
D = np.load(CACHE / "diagnostics.npz")
P = np.load(CACHE / "sampling.npz")
A = np.load(CACHE / "amr-slice.npz")
L = np.load(CACHE / "lines.npz")
S = json.loads((CACHE / "selection.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def raster(rect, values):
    lower, upper = A["lower"], A["upper"]
    dy = np.min(rect[:, 1] - rect[:, 0])
    dz = np.min(rect[:, 3] - rect[:, 2])
    image = np.full(
        (round((upper[2] - lower[2]) / dz), round((upper[1] - lower[1]) / dy)), np.nan
    )
    ys = np.rint((rect[:, :2] - lower[1]) / dy).astype(int)
    zs = np.rint((rect[:, 2:] - lower[2]) / dz).astype(int)
    for y, z, value in zip(ys, zs, values):
        image[z[0] : z[1], y[0] : y[1]] = value
    return image


def main():
    fig = plt.figure(figsize=(11.4, 5.8))
    extent = np.array([A["lower"][1], A["upper"][1], A["lower"][2], A["upper"][2]]) * 10
    ratio = A["ratio"]
    pos = ratio[np.isfinite(ratio) & (ratio > 0)]
    panels = [
        (
            P["rectangles"],
            np.where(D["valid"], D["log10_q"], np.nan),
            "inferno",
            Normalize(np.log10(2), 3),
            r"$\log_{10} Q$",
            "max",
        ),
        (
            P["rectangles"],
            np.where(D["complete"], D["twist"], np.nan),
            "RdBu_r",
            Normalize(-3, 3),
            r"$T_w$",
            "both",
        ),
        (
            A["rectangles"],
            ratio,
            "magma",
            LogNorm(*np.percentile(pos, [1, 99.8])),
            r"$|J|/|B|$ [A m$^{-2}$ T$^{-1}$]",
            "both",
        ),
    ]
    locations = [(0.055, 0.535), (0.54, 0.535), (0.055, 0.12)]
    panel_width = 0.395
    panel_height = panel_width * 11.4 / 5.8 / 2
    colors = plt.get_cmap("tab20")(np.arange(20))
    # Strengthen pale categorical colors for legibility at publication size.
    hsv = rgb_to_hsv(colors[:, :3])
    hsv[:, 1] = np.maximum(hsv[:, 1], 0.65)
    hsv[:, 2] = np.minimum(hsv[:, 2], 0.78)
    colors[:, :3] = hsv_to_rgb(hsv)
    for i, ((rect, values, cmap, norm, label, extend), (left, bottom)) in enumerate(
        zip(panels, locations)
    ):
        ax = fig.add_axes([left, bottom, panel_width, panel_height])
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#b9bdc4")
        im = ax.imshow(
            raster(rect, values),
            origin="lower",
            extent=extent,
            cmap=cm,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
            rasterized=True,
        )
        ax.set(xlim=extent[:2], ylim=extent[2:])
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([0, 5, 10])
        if i == 2:
            ax.set_xlabel("y [Mm]")
        else:
            ax.tick_params(labelbottom=False)
        if i != 1:
            ax.set_ylabel("z [Mm]")
        else:
            ax.tick_params(labelleft=False)
        ax.text(
            0.025,
            0.965,
            f"({chr(97 + i)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
        )
        cb = fig.colorbar(
            im,
            cax=fig.add_axes([left + panel_width + 0.006, bottom, 0.009, panel_height]),
            orientation="vertical",
            extend=extend,
        )
        cb.set_label(label, fontsize=10, labelpad=3)
        cb.ax.tick_params(labelsize=9, pad=2)
        cb.set_ticks(([1, 2, 3], [-3, 0, 3], [0.01, 1, 100])[i])
        if i == 1:
            ax.scatter(
                L["seeds"][:, 1] * 10,
                L["seeds"][:, 2] * 10,
                s=18,
                facecolors="none",
                edgecolors=colors,
                linewidths=0.8,
            )
    # Orthographic projection along x uses the same y-z frame as the slices.
    ax = fig.add_axes([0.54, 0.12, panel_width, panel_height])
    ax.set_facecolor("#F7F9FB")
    for i, color in enumerate(colors):
        for side in range(2):
            a, b = L["offsets"][2 * i + side : 2 * i + side + 2]
            xyz = L["positions"][a:b] * 10
            ax.plot(
                xyz[:, 1],
                xyz[:, 2],
                color=color,
                lw=1.35,
                solid_capstyle="round",
                solid_joinstyle="round",
                path_effects=[
                    path_effects.Stroke(linewidth=2.05, foreground="white"),
                    path_effects.Normal(),
                ],
            )
    ax.scatter(
        L["seeds"][:, 1] * 10,
        L["seeds"][:, 2] * 10,
        s=14,
        facecolors=colors,
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="y [Mm]", aspect="equal")
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([0, 5, 10])
    ax.tick_params(labelleft=False)
    ax.text(
        0.025,
        0.965,
        "(d)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
    )
    for suffix in ("png", "pdf", "svg"):
        target = (PDF if suffix == "pdf" else OUT) / f"figure5-current-sheet.{suffix}"
        temporary = target.with_name(f"{target.stem}.tmp.{suffix}")
        fig.savefig(temporary, dpi=240, facecolor="white")
        temporary.replace(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
