"""Render grouped magnetic/current profiles with a disclosed piecewise distance scale."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from _figure4_schematic import render_schematic

from _paths import data_directory, OUTPUT

DATA_ROOT = data_directory()

DATA = DATA_ROOT / "figure4"
OUT = OUTPUT
PDF_OUT = OUTPUT
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)
SUPPORT = OUT / "supporting"
SUPPORT.mkdir(exist_ok=True)
render_schematic()
report = json.loads((DATA / "summary.json").read_text())
line = report["line"]
with np.load(DATA / "profiles.npz") as saved:
    current = {k: saved[k] for k in saved.files}
with np.load(DATA / "magnetic-profiles.npz") as saved:
    magnetic = {k: saved[k] for k in saved.files}
np.testing.assert_array_equal(current["s_Mm"], magnetic["s_Mm"])
s = current["s_Mm"]
faces = np.array(line["interfaces_Mm"])
length = line["bands"][-1]["stop_Mm"]
compression = faces.copy()
breaks = np.r_[0.0, faces, length]
widths = np.diff(breaks)
scales = widths[0] / widths
ratio = float(scales[1])
assert np.all(widths > 0)


def forward(x):
    x = np.asarray(x)
    return np.where(
        x < faces[0],
        x,
        np.where(
            x <= faces[1],
            widths[0] + scales[1] * (x - faces[0]),
            2 * widths[0] + scales[2] * (x - faces[1]),
        ),
    )


def inverse(x):
    x = np.asarray(x)
    return np.where(
        x < widths[0],
        x,
        np.where(
            x <= 2 * widths[0],
            faces[0] + (x - widths[0]) / scales[1],
            faces[1] + (x - 2 * widths[0]) / scales[2],
        ),
    )


np.testing.assert_allclose(inverse(forward(s)), s, atol=1e-14, rtol=0)
np.testing.assert_allclose(
    np.diff(forward(breaks)), np.full(3, widths[0]), atol=1e-14, rtol=0
)
assert np.all(np.diff(forward(s)) > 0)
STYLES = {
    "native": dict(color="#40576A", lw=1.45, ls="-", label="Native AMR"),
    "vtk": dict(color="#C76D52", lw=1.6, ls="-", label="VTK PointData"),
    "uniform_l5": dict(color="#218AC4", lw=1.6, ls="-", label="Uniform L5"),
    "uniform_l6": dict(color="#D94F70", lw=1.7, ls=(0, (5, 3)), label="Uniform L6"),
}
MARKERS = {"native": "o", "vtk": "D", "uniform_l5": "s"}
MARKER_INDICES = {}
# Stagger identifiers in display space, while keeping their actual query values.
for key, phase in zip(MARKERS, (0.25, 0.58, 0.82)):
    target = (np.arange(7) + phase) / 7 * float(forward(length))
    allowed = np.all(
        abs(forward(s)[:, None] - forward(faces)[None, :]) > 0.012 * forward(length),
        axis=1,
    )
    candidates = np.flatnonzero(allowed)
    indices = candidates[
        np.argmin(abs(forward(s[candidates])[:, None] - target[None, :]), axis=0)
    ]
    assert len(np.unique(indices)) == 7
    MARKER_INDICES[key] = indices


def legend_handle(key):
    extra = (
        {}
        if key not in MARKERS
        else dict(
            marker=MARKERS[key],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=0.95,
        )
    )
    return Line2D([], [], **STYLES[key], **extra)


GROUPS = [("native", "vtk"), ("native", "uniform_l5", "uniform_l6")]
LEVEL_COLORS = {4: "#e5ecef", 5: "#94b9c7", 6: "#356981"}


def panel(
    ax, group, quantity, component=0, *, bottom=False, letter=None, residual=False
):
    ax.set_xscale("function", functions=(forward, inverse))
    ax.set_xlim(0, length)
    if quantity == "B":
        vals = {k: magnetic[k + "_B_G"][:, component] for k in STYLES}
        all_values = np.concatenate(list(vals.values()))
        lo, hi = all_values.min(), all_values.max()
        pad = 0.10 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f"$B_{'xyz'[component]}$ [G]")
    else:
        vals = {k: current[k + "_uA_m2"] for k in STYLES}
        ax.set_ylim(0, 190)
        ax.set_ylabel(r"$|J|$ [$\mu$A m$^{-2}$]")
    if residual:
        reference = vals["native"]
        multiplier = 1000.0 if quantity == "B" else 1.0
        vals = {k: (v - reference) * multiplier for k, v in vals.items()}
        all_values = np.concatenate([vals[k] for k in STYLES if k != "native"])
        lo, hi = min(0.0, all_values.min()), max(0.0, all_values.max())
        pad = max(0.1 * (hi - lo), 1e-8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(
            r"$\Delta B_x$ [mG]"
            if quantity == "B"
            else r"$\Delta |J|$ [$\mu$A m$^{-2}$]"
        )
        ax.axhline(0, color="#59636C", lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    for band in line["bands"]:
        ax.axvspan(
            band["start_Mm"],
            band["stop_Mm"],
            color=LEVEL_COLORS[band["level"]],
            alpha=0.14,
            lw=0,
            zorder=0,
        )
        if quantity == "J" and not residual:
            center = inverse(
                0.5 * (forward(band["start_Mm"]) + forward(band["stop_Mm"]))
            )
            ax.text(
                center,
                10,
                f"Native L{band['level']}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=8,
            )
    ax.axvspan(
        *compression,
        facecolor="none",
        edgecolor="#a5adb5",
        hatch="///",
        lw=0,
        alpha=0.17,
        zorder=0,
    )
    for key in group:
        if residual and key == "native":
            continue
        if residual or key in ("native", "vtk"):
            for lo, hi in zip(np.r_[-np.inf, faces], np.r_[faces, np.inf]):
                mask = (s > lo) & (s < hi)
                ax.plot(
                    s[mask],
                    vals[key][mask],
                    **STYLES[key],
                    zorder=4 if key == "native" else 5,
                )
        else:
            ax.plot(s, vals[key], **STYLES[key], zorder=5)
    if not residual:
        # Draw sparse marker overlays last so an overlapping line cannot hide the reference.
        for key in group:
            if key not in MARKERS:
                continue
            idx = MARKER_INDICES[key]
            ax.plot(
                s[idx],
                vals[key][idx],
                ls="None",
                marker=MARKERS[key],
                ms=4.2,
                markerfacecolor="white",
                markeredgecolor=STYLES[key]["color"],
                markeredgewidth=0.95,
                zorder=8,
            )
    for face, color in zip(faces, ["#704166", "#16766e"]):
        ax.axvline(face, color=color, ls=":", lw=1.15, zorder=6)
    ticks = [0, 0.5, 1.0, 2.0, 2.5, length]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FixedFormatter(["0", "0.5", "1.0", "2.0", "2.5", f"{length:.2f}"])
    )
    if bottom:
        ax.set_xlabel("Distance along line, s [Mm]")
    else:
        ax.tick_params(labelbottom=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7a8790")
    ax.grid(axis="y", color="#a1adb5", lw=0.5, alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_fontsize(10.5)
    ax.yaxis.label.set_fontsize(9.5 if residual else 11)
    if letter:
        ax.text(
            0.025,
            0.96,
            f"({letter})",
            transform=ax.transAxes,
            va="top",
            fontsize=15,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.8),
            zorder=9,
        )


with plt.rc_context(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
):
    fig = plt.figure(figsize=(15, 8.7))
    axis = fig.add_axes([0.002, 0.095, 0.44, 0.81])
    image = plt.imread(OUTPUT / "supporting/schematic.png")
    yy, xx = np.nonzero(np.any(image[:, :, :3] < 0.995, axis=2))
    axis.imshow(
        image[
            max(0, yy.min() - 18) : min(image.shape[0], yy.max() + 19),
            max(0, xx.min() - 18) : min(image.shape[1], xx.max() + 19),
        ],
        interpolation="none",
    )
    axis.set_axis_off()
    for col, group in enumerate(GROUPS):
        x = [0.50, 0.735][col]
        width = 0.22
        top = fig.add_axes([x, 0.67, width, 0.145])
        panel(top, group, "B", letter="bc"[col])
        bdiff = fig.add_axes([x, 0.568, width, 0.09])
        panel(bdiff, group, "B", residual=True)
        bot = fig.add_axes([x, 0.332, width, 0.18])
        panel(bot, group, "J", letter="de"[col])
        jdiff = fig.add_axes([x, 0.23, width, 0.09])
        panel(jdiff, group, "J", bottom=True, residual=True)
        # Corresponding rows use identical limits; keep vertical labels only at left.
        if col == 1:
            for ax in (top, bdiff, bot, jdiff):
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)
        handles = [legend_handle(k) for k in group]
        fig.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=(x + width / 2, 0.892),
            ncol=len(group),
            frameon=False,
            fontsize=9,
            handlelength=1.5,
            columnspacing=0.7,
        )
        # The ruler explicitly marks retained and compressed display scales.
        for lo, hi, label in [
            (breaks[i], breaks[i + 1], f"{scales[i]:.2f}×") for i in range(3)
        ]:
            middle = inverse(0.5 * (forward(lo) + forward(hi)))
            top.plot(
                [lo, hi],
                [1.065, 1.065],
                transform=top.get_xaxis_transform(),
                color="#8998a2",
                lw=0.8,
                clip_on=False,
            )
            top.text(
                middle,
                1.085,
                label,
                transform=top.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#596b77",
            )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            (PDF_OUT if ext == "pdf" else OUT) / f"figure4-p16-field-current.{ext}",
            dpi=240,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)
    # Retain all three field components as a separate, equally scaled companion.
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.10, top=0.91, wspace=0.24, hspace=0.15
    )
    for i in range(3):
        for col, group in enumerate(GROUPS):
            panel(axes[i, col], group, "B", component=i, bottom=(i == 2))
    for col, group in enumerate(GROUPS):
        fig.legend(
            handles=[legend_handle(k) for k in group],
            loc="center",
            bbox_to_anchor=(0.28 if col == 0 else 0.77, 0.96),
            ncol=len(group),
            frameon=False,
            fontsize=10,
        )
    fig.text(
        0.5,
        0.025,
        "Native L4, L5 and L6 have equal display widths; distance ticks retain physical values.",
        ha="center",
        fontsize=9,
        color="#596b77",
    )
    fig.savefig(SUPPORT / "p16-magnetic-components.png", dpi=200, facecolor="white")
    fig.savefig(SUPPORT / "p16-magnetic-components.pdf", facecolor="white")
    plt.close(fig)
settings = {
    "display_methods": list(STYLES),
    "groups": GROUPS,
    "primary_field": "Bx",
    "magnetic_unit": "G",
    "compression_interval_Mm": compression.tolist(),
    "display_scale_inside": ratio,
    "display_scale_by_level": {str(4 + i): float(v) for i, v in enumerate(scales)},
    "equal_native_band_widths": True,
    "tick_values_are_physical": True,
    "samples_removed": False,
    "current_recomputed": False,
    "current_ylim": [0, 190],
    "figure_size_inches": [15, 8.7],
    "export_bbox": "tight",
    "export_padding_inches": 0.08,
    "profile_width_inches": 3.3,
    "right_column_y_labels": False,
    "main_difference_gap_fraction": 0.012,
    "profile_vertical_extent_fraction": [0.23, 0.815],
    "tick_fontsize": 9,
    "axis_title_fontsize": 11,
    "legend_fontsize": 9,
    "difference_definition": "method minus Native AMR",
    "difference_B_unit": "mG",
    "difference_J_unit": "microampere per square metre",
    "colors": {k: v["color"] for k, v in STYLES.items()},
    "markers": MARKERS,
    "marker_s_Mm": {k: s[idx].tolist() for k, idx in MARKER_INDICES.items()},
    "main_output": "figure4-p16-field-current.png",
}
(OUTPUT / "grouped-plot-settings.json").write_text(
    json.dumps(settings, indent=2) + "\n"
)
(OUT / "figure4-p16-field-current.txt").write_text(
    "Magnetic-field input and derived current on the P16_11 probe. Panel (a) retains the three-dimensional native-current context. Panels (b,d) compare Native AMR and VTK PointData; panels (c,e) compare Native AMR with Uniform L5 and L6. The upper main panels show the reconstructed Cartesian magnetic component Bx in gauss, and the lower main panels show the complete three-dimensional current magnitude. Each main panel is followed by a linear signed-difference strip: method minus Native AMR. Magnetic differences are displayed in milligauss and current differences in microampere per square metre. The two columns share the same vertical range for each corresponding main panel and difference strip; redundant right-column vertical labels are omitted. Each main panel and its difference strip are closely stacked. Native AMR is a comparison reference, not an analytic truth; differences are not absolute accuracy errors. Slate blue, terracotta, blue and rose distinguish Native AMR, VTK, Uniform L5 and Uniform L6, respectively. Native lines are thinner and L6 remains dashed. Seven staggered open circles, diamonds and squares identify Native AMR, VTK and Uniform L5 on the main curves, respectively. They are selected existing line-query samples, distributed approximately uniformly in display space, not raw cell-center measurements or extra observations. Difference strips have no markers. Bx is tangential to both selected interfaces (z-normal L4/L5 and y-normal L5/L6), so its normal derivatives contribute to curl at both interfaces; the plotted line profile alone does not determine the full three-dimensional curl. All three magnetic components are provided in the companion figure. Magnetic profiles are evaluated at the same physical query coordinates as the current profiles. Shading denotes native levels. A continuous piecewise-linear horizontal map gives native L4, L5 and L6 equal display widths. The actual L5 interval is s=%.6f–%.6f Mm. All samples remain present and ticks retain physical distances. The ruler above each column states the per-region display scales relative to L4; the L5 band is hatched. The horizontal scale changes at the actual refinement interfaces. Visual slopes across scale zones are not directly comparable physical gradients. Curves represent reconstructed field queries, not raw stored cell-center values. All current definitions and unit conventions follow the accompanying method report.\n"
    % tuple(compression)
)
print("Rendered Figure 4 and magnetic-component companion.")

differences = {}
for key in STYLES:
    if key == "native":
        continue
    db = (magnetic[key + "_B_G"][:, 0] - magnetic["native_B_G"][:, 0]) * 1000
    dj = current[key + "_uA_m2"] - current["native_uA_m2"]
    differences[key] = {
        "Bx_difference_range_mG": [float(db.min()), float(db.max())],
        "current_difference_range_uA_m2": [float(dj.min()), float(dj.max())],
    }
(OUTPUT / "difference-summary.json").write_text(
    json.dumps(differences, indent=2) + "\n"
)
