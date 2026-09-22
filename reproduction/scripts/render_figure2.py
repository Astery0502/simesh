"""Render Figure 2 from its vector drawing instructions."""

from _paths import OUTPUT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = OUTPUT
PDF_OUT = OUTPUT
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)
STEM = "figure2-software-architecture"
INK = "#233B4B"
EDGE = "#6B8190"
PYTHON = "#335F80"
GREEN = "#527E67"


def main():
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    fig, ax = plt.subplots(figsize=(12, 8.4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1000), ylim=(700, 0))
    ax.axis("off")

    def box(x, y, w, h, fill="white", stroke=EDGE):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=10",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=1,
                zorder=2,
            )
        )

    def text(x, y, value, size=10, color=INK, bold=False, code=False, **kwargs):
        ax.text(
            x,
            y,
            value,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans Mono" if code else "DejaVu Sans",
            zorder=5,
            **kwargs,
        )

    def arrow(points, color=EDGE):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=color, lw=1.1, zorder=3)
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=10,
                color=color,
                lw=1.1,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    # Balanced input panels retain the two-sided flow around the shared mesh.
    box(65, 45, 250, 275)
    box(685, 45, 250, 275)
    text(190, 70, "Discrete fields", 12, bold=True)
    text(190, 103, "Source", 11, PYTHON, code=True)
    text(190, 132, "open_amrvac()", 9.5, PYTHON, code=True)
    text(190, 155, "source_from_arrays()", 9.5, PYTHON, code=True)
    arrow([(190, 177), (190, 253)])
    text(
        190,
        213,
        "read_fields() / prepare()",
        9,
        PYTHON,
        code=True,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 4},
    )
    text(190, 277, "Fields", 12, PYTHON, bold=True, code=True)
    text(190, 302, "Native simulation and derived fields", 9)

    text(810, 70, "Analysis geometry", 12, bold=True)
    text(810, 108, "PointSet", 11, PYTHON, code=True)
    text(810, 130, "Sampling positions", 9)
    text(810, 152, "PointSet() / PointSet.from_plane()", 8, PYTHON, code=True)
    text(810, 202, "RaySet", 11, PYTHON, code=True)
    text(810, 224, "Line-of-sight rays", 9)
    text(810, 274, "LineSet", 11, PYTHON, code=True)
    text(810, 296, "Traced paths", 9)

    box(365, 150, 270, 170, "#F0F4F7", "#AABBC6")
    text(500, 177, "Native AMR representation", 11, bold=True)
    text(500, 216, "Mesh", 13, PYTHON, bold=True, code=True)
    text(500, 247, "AMR topology & geometry", 10)
    text(500, 291, "select_region()", 9.5, PYTHON, code=True)

    # Geometry is independent; operations associate it with native fields.
    for x in (190, 500, 810):
        arrow([(x, 326), (x, 354)])

    box(65, 360, 870, 150, "#EDF4FA", "#6886A3")
    text(500, 379, "Scientific applications", 12, bold=True)
    centers = (174, 391, 609, 826)
    for x in (282.5, 500, 717.5):
        ax.plot([x, x], [399, 472], color="#B9CBD9", lw=0.8)
    for x, heading, entry1, entry2 in (
        (174, "Field diagnostics", "derive()", "derivative()"),
        (
            391,
            "Statistics and\nregional integrals",
            "volume_integral()",
            "weighted_mean()",
        ),
        (609, "Tracing and\nmagnetic-topology", "app.trace()", "app.connectivity()"),
        (826, "Synthetic observations", "app.los()", "app.radiative_los()"),
    ):
        text(x, 414, heading, 8.5, INK, bold=True)
        text(x, 441, entry1, 9, PYTHON, code=True)
        text(x, 464, entry2, 9, PYTHON, code=True)
    ax.plot([80, 920], [479, 479], color="#B9CBD9", lw=0.8)
    text(
        500,
        495,
        "Sampling: app.sample() → SampledPoints   ·   sample_line_profiles() → LineProfiles",
        9,
        PYTHON,
        code=True,
    )

    # One product region keeps representative return types and reuse together.
    for x in centers:
        arrow([(x, 516), (x, 539)])
    box(65, 545, 870, 125, "#EAF4EE", GREEN)
    text(500, 565, "Reusable products", 12, GREEN, bold=True)
    for x, row1, row2 in (
        (174, "Fields", ""),
        (391, "ScalarResult", ""),
        (609, "LineSet", "ConnectivityMap"),
        (826, "RayResult", "RadiationResult"),
    ):
        text(x, 594 if row2 else 605, row1, 10, GREEN, code=True)
        if row2:
            text(x, 615, row2, 10, GREEN, code=True)
    ax.plot([80, 920], [632, 632], color="#BBD1C3", lw=0.8)
    text(282, 651, "derive() / derivative() → Fields", 8.5, GREEN, code=True)
    text(718, 651, "ConnectivityMap.threshold() → PointSet", 8.5, GREEN, code=True)

    # Symmetric return paths distinguish field reuse from geometry reuse.
    arrow([(65, 605), (30, 605), (30, 277), (65, 277)], GREEN)
    arrow([(935, 605), (970, 605), (970, 277), (935, 277)], GREEN)
    text(15, 441, "New fields", 9, GREEN, bold=True, rotation=90)
    text(985, 441, "New geometries", 9, GREEN, bold=True, rotation=90)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(
            (PDF_OUT if suffix == "pdf" else OUT) / f"{STEM}.{suffix}",
            dpi=220,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
