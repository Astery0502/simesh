"""Render the manuscript magnetic-topology figure from saved scientific products."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LogNorm

from _paths import data_directory, OUTPUT

DATA = data_directory()
from _topology import build_figure
import matplotlib.pyplot as plt


def main():
    fig = build_figure()
    original_axes = list(fig.axes)
    labels = [
        axis
        for axis in original_axes
        if axis.get_ylabel() == r"$|J|$" or axis.get_xlabel() == r"$|J|$"
    ]
    assert len(labels) == 1, "Expected one current-proxy colorbar"
    axis = labels[0]
    setter = axis.set_ylabel if axis.get_ylabel() == r"$|J|$" else axis.set_xlabel
    setter(r"$j_*$ (code units)", color="black")
    fig.set_size_inches(16, 11.8, forward=False)
    # Compact the map columns and keep the 3D view close to their colorbars.
    bounds = [
        [0.060, 0.595, 0.250, 0.365],
        [0.065, 0.540, 0.240, 0.014],
        [0.380, 0.595, 0.250, 0.365],
        [0.385, 0.540, 0.240, 0.014],
        [0.095, 0.065, 0.500, 0.445],
        [0.055, 0.150, 0.014, 0.260],
        [0.610, 0.150, 0.014, 0.260],
    ]
    for ax, position in zip(original_axes, bounds):
        ax.set_position(position)
    spatial = original_axes[4]
    spatial.patch.set_visible(False)
    panel_c = next(text for text in spatial.texts if text.get_text() == "(c)")
    panel_c.remove()
    # Place the panel letter just above the visible z-axis endpoint.
    spatial.text(
        spatial.get_xlim()[1],
        spatial.get_ylim()[0],
        spatial.get_zlim()[1] + 4,
        "(c)",
        ha="center",
        va="bottom",
        fontsize=18,
    )
    with np.load(DATA / "figure6/projection.npz", allow_pickle=False) as data:
        lower, upper = data["lower_Mm"], data["upper_Mm"]
        projection = data["projection"]
    maximum = float(np.percentile(projection[projection > 0], 99.5))
    image = projection / maximum
    palette = plt.get_cmap("magma").copy()
    palette.set_bad("black")
    proxy_ax = fig.add_axes([0.715, 0.230, 0.265, 0.620])
    proxy_ax.imshow(
        np.ma.masked_less_equal(image.T, 0),
        origin="lower",
        extent=[lower[0], upper[0], lower[1], upper[1]],
        cmap=palette,
        norm=LogNorm(0.001, 1),
        interpolation="nearest",
        aspect="equal",
    )
    proxy_ax.set(xlabel="x [Mm]", ylabel="y [Mm]")
    proxy_ax.text(
        0.025,
        0.975,
        "(d)",
        transform=proxy_ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )
    proxy_ax.text(
        0.975,
        0.975,
        "z = 6–80 Mm",
        transform=proxy_ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )
    # Apply typography to inherited axes as well as the new scientific panels.
    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", labelsize=13)
        for coordinate in ("xaxis", "yaxis", "zaxis"):
            if hasattr(ax, coordinate):
                getattr(ax, coordinate).label.set_fontsize(15)
        for text in ax.texts:
            if text.get_text().startswith("("):
                text.set_fontsize(18)
    spatial.tick_params(axis="z", labelsize=13, pad=2)
    spatial.xaxis.labelpad = 10
    spatial.yaxis.labelpad = 10
    spatial.zaxis.labelpad = 0
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"figure6-magnetic-topology.{ext}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
