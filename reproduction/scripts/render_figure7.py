"""Build a reproducible thermal-analysis and twelve-band overview."""

import json, string
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from _paths import data_directory, OUTPUT

DATA = data_directory()

BASE = DATA / "figure7"
CONFIG = json.loads((BASE / "configuration.json").read_text())
WAVES = CONFIG["wavelengths_angstrom"]
BOXES = [np.array([[-4, -20, 2], [4, 20, 6]]), np.array([[-4, -20, 6], [4, 20, 10]])]
COLORS = ["#40e0d0", "#ff8c42"]


def render():
    d = np.load(BASE / "diagnostics.npz", allow_pickle=False)
    bands = np.load(BASE / "bands.npz", allow_pickle=False)
    palettes = np.load(BASE / "colormaps.npz", allow_pickle=False)
    settings = json.loads((BASE / "plot-settings.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.4, 19.8))
    width = (0.915 - 2 * 0.006) / 3
    height = width * 12.4 * 0.8 / 19.8
    axes = np.array(
        [
            fig.add_axes(
                [
                    0.07 + c * (width + 0.006),
                    0.985 - (r + 1) * height - r * 0.003,
                    width,
                    height,
                ]
            )
            for r in range(6)
            for c in range(3)
        ]
    ).reshape(6, 3)

    def tag(ax, i, label):
        ax.text(
            0.035,
            0.96,
            f"({string.ascii_lowercase[i]}) {label}",
            transform=ax.transAxes,
            va="top",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.25, edgecolor="none", pad=2),
        )

    def spatial(ax, i):
        ax.set(
            xlim=(-100, 100),
            ylim=(0, 160),
            xticks=[-100, -50, 0, 50, 100],
            yticks=[0, 40, 80, 120, 160],
            aspect="equal",
        )
        if i // 3 > 0:
            ax.set_yticks([0, 40, 80, 120])
        if i // 3 == 5 and i % 3 > 0:
            ax.set_xticks([-50, 0, 50, 100])
        if i % 3 == 0:
            ax.set_ylabel("z [Mm]")
        else:
            ax.tick_params(labelleft=False)
        if i // 3 == 5:
            ax.set_xlabel("x [Mm]")
        else:
            ax.tick_params(labelbottom=False)

    def bar(im, ax, unit=None, log=True):
        ax.add_patch(
            Rectangle(
                (0.78, 0.48),
                0.205,
                0.39,
                transform=ax.transAxes,
                facecolor="black",
                alpha=0.42,
                edgecolor="none",
                zorder=8,
            )
        )
        cax = ax.inset_axes([0.936, 0.52, 0.019, 0.28], zorder=9)
        cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both")
        cb.set_ticks(
            10.0
            ** np.round(
                np.linspace(
                    np.ceil(np.log10(im.norm.vmin)), np.floor(np.log10(im.norm.vmax)), 3
                )
            )
            if log
            else np.linspace(im.norm.vmin, im.norm.vmax, 3)
        )
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=7, pad=2, length=2, colors="white")
        cb.outline.set_edgecolor("white")
        cb.outline.set_linewidth(0.4)
        if unit:
            ax.text(
                0.972,
                0.85,
                unit,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="white",
                zorder=10,
            )

    vals = d["slice_values"]
    edges = d["slice_edges"] * 10
    specs = [
        ("Density", 0, "magma", LogNorm(1e-16, 1e-12), "g cm⁻³", 1),
        ("Temperature", 1, "inferno", LogNorm(1e4, 4e7), "K", 1),
        ("Speed", 2, "viridis", Normalize(0, 3000), "km s⁻¹", 1e-5),
    ]
    for i, (label, c, cmap, norm, unit, scale) in enumerate(specs):
        ax = axes.flat[i]
        for k, (xe, ze) in enumerate(edges):
            if xe[-1] < -100 or xe[0] > 100 or ze[0] > 160:
                continue
            im = ax.pcolormesh(
                xe,
                ze,
                vals[k, :, :, c].T * scale,
                cmap=cmap,
                norm=norm,
                rasterized=True,
                shading="flat",
            )
        tag(ax, i, label)
        spatial(ax, i)
        bar(im, ax, unit, i != 2)
        ax.text(
            0.96,
            0.035,
            "y = 0 Mm",
            transform=ax.transAxes,
            ha="right",
            color="white",
            fontsize=9,
        )
    # Sample native slice cells on a sparse display lattice for in-plane arrows.
    xx, zz = np.meshgrid(np.linspace(-90, 90, 17), np.linspace(8, 152, 14))
    vx = np.full_like(xx, np.nan)
    vz = vx.copy()
    for k, (xe, ze) in enumerate(edges):
        mask = (xx >= xe[0]) & (xx < xe[-1]) & (zz >= ze[0]) & (zz < ze[-1])
        ix = np.searchsorted(xe, xx[mask], side="right") - 1
        iz = np.searchsorted(ze, zz[mask], side="right") - 1
        vx[mask] = vals[k, ix, iz, 3] * 1e-5
        vz[mask] = vals[k, ix, iz, 5] * 1e-5
    q = axes.flat[2].quiver(xx, zz, vx, vz, color="white", scale=30000, width=0.003)
    axes.flat[2].quiverkey(
        q,
        0.78,
        0.23,
        1000,
        "1000 km s⁻¹",
        labelcolor="white",
        labelpos="S",
        fontproperties={"size": 8},
    )
    for i, key, label, cmap, norm, unit in [
        (3, "density", "Column mass", "magma", LogNorm(1e-5, 1e-1), "g cm⁻²"),
        (
            4,
            "weighted_temperature",
            "Mass-weighted temperature",
            "inferno",
            LogNorm(1e4, 4e7),
            "K",
        ),
    ]:
        ax = axes.flat[i]
        im = ax.imshow(
            d[key],
            origin="lower",
            extent=[-100, 100, 0, 160],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        tag(ax, i, label)
        spatial(ax, i)
        bar(im, ax, unit)
        for j, box in enumerate(BOXES):
            x, z = box[0, [0, 2]] * 10
            w, h = (box[1] - box[0])[[0, 2]] * 10
            ax.add_patch(
                Rectangle((x, z), w, h, fill=False, edgecolor=COLORS[j], linewidth=1.4)
            )
            ax.text(
                x + 3, z + 3, f"R{j + 1}", color=COLORS[j], fontsize=9, weight="bold"
            )
    container = axes.flat[5]
    container.set_facecolor("#161b24")
    container.set_xticks([])
    container.set_yticks([])
    ax = container.inset_axes([0.17, 0.22, 0.79, 0.60])
    ax.set_facecolor("#161b24")
    bins = d["temperature_edges"]
    for j in range(2):
        hist = d[f"hist_{j}"] / np.diff(np.log10(bins))
        mass = float(d[f"mass_{j}"])
        ax.stairs(
            np.where(hist > 0, hist, np.nan),
            bins,
            color=COLORS[j],
            lw=1.5,
            label=f"R{j + 1}: M = {mass:.2e} g",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(1e4, 5e7),
        ylim=(1e7, 1e16),
        xlabel="T [K]",
        ylabel="dM / dlog₁₀T [g]",
    )
    ax.tick_params(colors="white", labelsize=7, direction="in", pad=2)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(alpha=0.12)
    ax.legend(
        loc="lower left",
        fontsize=6.5,
        facecolor="#161b24",
        labelcolor="white",
        framealpha=0.8,
    )
    for spine in ax.spines.values():
        spine.set_color("#888888")
    tag(container, 5, "Temperature distribution")
    for k, wave in enumerate(WAVES):
        i = k + 6
        ax = axes.flat[i]
        image = bands[f"intensity_{wave}"]
        instrument = "AIA" if k < 7 else "IRIS" if k == 7 else "EIS"
        cmap = ListedColormap(palettes[f"rgba_{wave}"])
        cmap.set_bad(cmap(0))
        lo, hi = settings[str(wave)]["color_limits"]
        im = ax.imshow(
            np.ma.masked_less_equal(image, 0),
            origin="lower",
            extent=[-200, 200, 0, 400],
            norm=LogNorm(lo, hi),
            cmap=cmap,
            interpolation="nearest",
        )
        tag(ax, i, f"{instrument} {wave} Å")
        spatial(ax, i)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT / f"figure7-thermal-diagnostics.{ext}",
            dpi=210,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    print("Figure saved", flush=True)


if __name__ == "__main__":
    render()
