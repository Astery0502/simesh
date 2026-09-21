"""Render Figure 1 from its vector drawing instructions."""

from _paths import OUTPUT
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as DrawingPath

OUT = OUTPUT
PDF_OUT = OUTPUT
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"}
)
fig, ax = plt.subplots(figsize=(8, 7.87))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 900), ylim=(885, 0))
ax.axis("off")
ink, edge = "#233B4B", "#6B8190"


# Tighten the untitled top margin and product panel while retaining ring topology.
def compact_y(y):
    return float(
        np.interp(
            y,
            [0, 80, 130, 190, 215, 275, 355, 425, 520, 675, 790, 935, 960, 1030, 1080],
            [0, 80, 115, 125, 140, 200, 260, 330, 395, 550, 635, 745, 765, 835, 885],
        )
    )


def box(x, y, w, h, fill="white", stroke=edge, radius=8, lw=1):
    y, h = compact_y(y), compact_y(y + h) - compact_y(y)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
            zorder=2,
        )
    )


def text(x, y, value, size=10, weight="normal", color=ink, **kw):
    ax.text(
        x,
        compact_y(y),
        value,
        fontsize=size * 1.08,
        fontweight=weight,
        color=color,
        ha=kw.pop("ha", "center"),
        va="center",
        zorder=5,
        **kw,
    )


def arrow(start, end, both=False):
    start = (start[0], compact_y(start[1]))
    end = (end[0], compact_y(end[1]))
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<->" if both else "-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=edge,
            zorder=3,
            shrinkA=0,
            shrinkB=0,
        )
    )


def line(points, color=edge, lw=1.1):
    xs, ys = zip(*[(x, compact_y(y)) for x, y in points])
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)


def curve(vertices, color=edge, arrowhead=False):
    vertices = [(x, compact_y(y)) for x, y in vertices]
    path = DrawingPath(
        vertices, [DrawingPath.MOVETO] + [DrawingPath.CURVE4] * (len(vertices) - 1)
    )
    if arrowhead:
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color=color,
                zorder=3,
            )
        )
    else:
        ax.add_patch(
            PathPatch(path, fill=False, linewidth=1.1, edgecolor=color, zorder=3)
        )


# Centered entry and symmetric inputs retain the reference diagram's hierarchy.
box(50, 130, 800, 830, "#FAFCFD", "#AABBC6", radius=12)
box(315, 20, 270, 60, "#FFFFFF")
text(450, 50, "3D MHD simulation", 12)
arrow((450, 87), (450, 120))

box(140, 215, 230, 60, "#FFFFFF")
text(255, 245, "Discrete fields", 11)
box(530, 215, 230, 60, "#FFFFFF")
text(645, 245, "Analysis geometry", 11)
curve([(255, 282), (255, 325), (285, 313), (350, 350)])
curve([(645, 282), (645, 325), (615, 313), (550, 350)])
box(285, 355, 330, 70, "#F0F4F7", "#AABBC6", radius=14)
text(450, 390, "Native AMR representation", 11)

# The left operation panel and right return path form an open, readable ring.
curve([(355, 432), (282, 478), (280, 465), (280, 510)], arrowhead=True)
box(75, 520, 410, 155, "#EDF4FA", "#6886A3", radius=14, lw=1.2)
text(280, 545, "Scientific applications", 12, "bold")
line([(89, 568), (471, 568)], "#B9CBD9", 0.7)
line([(280, 578), (280, 661)], "#B9CBD9", 0.7)
line([(89, 617), (471, 617)], "#B9CBD9", 0.7)
text(184, 593, "Field\ndiagnostics", 10.5, linespacing=1.12)
text(376, 593, "Tracing and\nmagnetic-topology", 8.0, color=ink, linespacing=1.12)
text(184, 641, "Statistics and\nregional integrals", 10, color=ink, linespacing=1.12)
text(376, 641, "Synthetic\nobservations", 10.5, linespacing=1.12)
curve([(280, 682), (280, 730), (280, 731), (345, 780)], arrowhead=True)

# A compact rounded panel aligns product labels in two left-aligned columns.
box(230, 790, 440, 145, "#EAF4EE", "#648B74", radius=14, lw=1.2)
text(450, 819, "Reusable products", 12, "bold", "#264735")
text(260, 861, "Derived fields", 10, fontstyle="italic", ha="left")
text(520, 861, "Thresholds", 10, fontstyle="italic", ha="left")
text(260, 900, "Isosurfaces / isovolumes", 10, fontstyle="italic", ha="left")
text(520, 900, "Selections", 10, fontstyle="italic", ha="left")
# Mirror the left flow around the centerline, with feedback directed upward.
curve(
    [
        (555, 780),
        (620, 731),
        (620, 730),
        (620, 682),
        (620, 620),
        (620, 566),
        (620, 510),
        (620, 465),
        (618, 478),
        (545, 432),
    ],
    "#9B7736",
    arrowhead=True,
)
text(
    620,
    598,
    "New fields\nNew geometries",
    9.5,
    color="#795B29",
    linespacing=1.4,
    bbox={"facecolor": "#FAFCFD", "edgecolor": "none", "pad": 6},
)

# Framework-level persistence ports match the reference's uncluttered footer.
# Export/reload applies to native fields; save/load applies to supported results.
arrow((300, 972), (300, 1018), both=True)
arrow((600, 972), (600, 1018), both=True)
text(
    300,
    995,
    "Export / reload",
    10,
    bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
)
text(
    600,
    995,
    "Save / load",
    10,
    bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
)
box(190, 1030, 220, 44, "#F5F5F5", "#9CA6AD")
text(300, 1052, "AMR snapshot", 10.5)
box(490, 1030, 220, 44, "#F5F5F5", "#9CA6AD")
text(600, 1052, "Result files", 10.5)

for suffix in ("svg", "pdf", "png"):
    fig.savefig(
        (PDF_OUT if suffix == "pdf" else OUT) / f"figure1-framework.{suffix}",
        dpi=200,
        facecolor="white",
    )
plt.close(fig)
