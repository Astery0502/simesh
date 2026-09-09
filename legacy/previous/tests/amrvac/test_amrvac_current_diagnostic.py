import os
from pathlib import Path
from struct import pack
import zlib

import numpy as np

from simesh.amrvac import open_dataset


FIXTURE_PATH = Path("data/weno509_sub_0000.dat")
REPORT_DIR = Path("report/amrvac-current-diagnostic")
REPORT_PATH = REPORT_DIR / "report.md"
BASELINE_SLICE_PNGS = {
    "mid_x": REPORT_DIR / "current_mid_x.png",
    "mid_y": REPORT_DIR / "current_mid_y.png",
    "mid_z": REPORT_DIR / "current_mid_z.png",
}
FIELD_INDICES = [4, 5, 6]
GHOST_WIDTH = 2
CURRENT_THRESHOLD = 300.0
CURRENT_EPSILON = np.finfo(np.double).eps
EXISTING_SMOOTHING_PASSES = 2
SYNTHETIC_TOLERANCE = 1e-11
CONVENTION_SENSITIVITY_RATIO = 2.0
ENABLE_COMMAND = (
    "SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src "
    ".venv/bin/python tests/amrvac/test_amrvac_current_diagnostic.py"
)


def _format_number(value):
    if value is None:
        return "not evaluated"
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.12g}"


def _format_tuple(values):
    return tuple(float(value) for value in np.asarray(values, dtype=np.double).ravel())


def _format_percentiles(percentiles):
    keys = ("p0", "p50", "p95", "p99", "p100")
    return " / ".join(_format_number(percentiles.get(key)) for key in keys)


def _metric_passed(metric):
    return bool(np.isfinite(metric) and metric <= CURRENT_THRESHOLD)


def _metric_ratio(metric, baseline):
    metric = float(metric)
    baseline = float(baseline)
    if not np.isfinite(metric) or not np.isfinite(baseline):
        return float("inf")
    if metric == 0.0 and baseline == 0.0:
        return 1.0
    if metric <= 0.0 or baseline <= 0.0:
        return float("inf")
    return max(metric / baseline, baseline / metric)


def _smooth_current_plane(values, passes):
    smoothed = np.asarray(values, dtype=np.double).copy()
    for _ in range(passes):
        previous = smoothed.copy()
        smoothed[1:-1, 1:-1] = (
            0.25 * previous[1:-1, 1:-1]
            + 0.125
            * (
                previous[:-2, 1:-1]
                + previous[2:, 1:-1]
                + previous[1:-1, :-2]
                + previous[1:-1, 2:]
            )
            + 0.0625
            * (
                previous[:-2, :-2]
                + previous[:-2, 2:]
                + previous[2:, :-2]
                + previous[2:, 2:]
            )
        )
    return smoothed


def _axis_interpreted_bfield(bfield, spacing, axis_interpretation):
    spacing = np.asarray(spacing, dtype=np.double)
    if axis_interpretation == "standard_xyz":
        return bfield[0], bfield[1], bfield[2], spacing
    if axis_interpretation == "swap_xy":
        return (
            np.swapaxes(bfield[1], 0, 1),
            np.swapaxes(bfield[0], 0, 1),
            np.swapaxes(bfield[2], 0, 1),
            spacing[[1, 0, 2]],
        )
    raise ValueError(f"Unknown axis interpretation: {axis_interpretation}")


def _current_center_components(bfield, spacing, smoothing_passes=0, axis_interpretation="standard_xyz"):
    bfield = np.asarray(bfield, dtype=np.double)
    if bfield.shape[0] != 3:
        raise ValueError(f"Expected bfield shape (3, nx, ny, nz), got {bfield.shape}")

    bx, by, bz, spacing = _axis_interpreted_bfield(bfield, spacing, axis_interpretation)
    if min(bx.shape) < 3:
        raise ValueError(f"Current-density diagnostic needs at least three samples per axis, got {bx.shape}")

    center_z = bx.shape[2] // 2
    if center_z == 0 or center_z == bx.shape[2] - 1:
        raise ValueError("Current-density diagnostic needs at least three z samples.")

    dx, dy, dz = spacing
    bx_slice = bx[:, :, center_z]
    bx_zlo = bx[:, :, center_z - 1]
    bx_zhi = bx[:, :, center_z + 1]
    by_slice = by[:, :, center_z]
    by_zlo = by[:, :, center_z - 1]
    by_zhi = by[:, :, center_z + 1]
    bz_slice = bz[:, :, center_z]

    if smoothing_passes > 0:
        bx_slice = _smooth_current_plane(bx_slice, smoothing_passes)
        bx_zlo = _smooth_current_plane(bx_zlo, smoothing_passes)
        bx_zhi = _smooth_current_plane(bx_zhi, smoothing_passes)
        by_slice = _smooth_current_plane(by_slice, smoothing_passes)
        by_zlo = _smooth_current_plane(by_zlo, smoothing_passes)
        by_zhi = _smooth_current_plane(by_zhi, smoothing_passes)
        bz_slice = _smooth_current_plane(bz_slice, smoothing_passes)

    jx = np.gradient(bz_slice, dy, axis=1, edge_order=2) - (by_zhi - by_zlo) / (2.0 * dz)
    jy = (bx_zhi - bx_zlo) / (2.0 * dz) - np.gradient(bz_slice, dx, axis=0, edge_order=2)
    jz = np.gradient(by_slice, dx, axis=0, edge_order=2) - np.gradient(
        bx_slice,
        dy,
        axis=1,
        edge_order=2,
    )
    components = np.stack((jx, jy, jz), axis=0)
    magnitude = np.sqrt(np.sum(components * components, axis=0))
    return center_z, components, magnitude


def _current_smoothness_metric(jmag_slice, spacing):
    dx, dy = np.asarray(spacing, dtype=np.double)[:2]
    grad_x = np.gradient(jmag_slice, dx, axis=0, edge_order=2)
    grad_y = np.gradient(jmag_slice, dy, axis=1, edge_order=2)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    numerator = np.percentile(grad_mag, 99.0)
    denominator = np.median(grad_mag) + CURRENT_EPSILON
    return float(numerator / denominator)


def _write_grayscale_png(path, image):
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape

    def chunk(chunk_type, payload):
        return (
            pack(">I", len(payload))
            + chunk_type
            + payload
            + pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_current_slice_png(path, image_values, title, xlabel, ylabel, colorbar_label, *, vmin=None, vmax=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (path.parent / ".cache").mkdir(exist_ok=True)
        (path.parent / ".matplotlib-cache").mkdir(exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(path.parent / ".cache"))
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        finite = np.asarray(image_values[np.isfinite(image_values)])
        if finite.size == 0:
            image = np.zeros(image_values.shape, dtype=np.uint8)
        else:
            lo = float(np.min(finite)) if vmin is None else float(vmin)
            hi = float(np.max(finite)) if vmax is None else float(vmax)
            if hi <= lo:
                hi = lo + 1.0
            image = np.clip((image_values - lo) / (hi - lo), 0.0, 1.0)
            image = (255.0 * image).astype(np.uint8)
        _write_grayscale_png(path, image.T)
        return

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(
        image_values.T,
        origin="lower",
        cmap="plasma",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _center_plane_jmag(bfield, spacing, normal_axis, smoothing_passes=0, axis_interpretation="standard_xyz"):
    bfield = np.asarray(bfield, dtype=np.double)
    bx, by, bz, spacing = _axis_interpreted_bfield(bfield, spacing, axis_interpretation)
    fields = (bx, by, bz)
    shape = bx.shape
    if min(shape) < 3:
        raise ValueError(f"Current-density diagnostic needs at least three samples per axis, got {shape}")

    normal_axis = int(normal_axis)
    plane_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    center_index = shape[normal_axis] // 2
    if center_index == 0 or center_index == shape[normal_axis] - 1:
        raise ValueError("Current-density diagnostic needs at least three samples on the normal axis.")

    def plane(field, offset=0):
        selection = [slice(None), slice(None), slice(None)]
        selection[normal_axis] = center_index + offset
        return np.asarray(field[tuple(selection)], dtype=np.double)

    center_planes = [plane(field) for field in fields]
    lo_planes = [plane(field, -1) for field in fields]
    hi_planes = [plane(field, 1) for field in fields]
    if smoothing_passes > 0:
        center_planes = [_smooth_current_plane(item, smoothing_passes) for item in center_planes]
        lo_planes = [_smooth_current_plane(item, smoothing_passes) for item in lo_planes]
        hi_planes = [_smooth_current_plane(item, smoothing_passes) for item in hi_planes]

    derivatives = {}
    for component, center_plane in enumerate(center_planes):
        derivatives[(component, plane_axes[0])] = np.gradient(
            center_plane,
            spacing[plane_axes[0]],
            axis=0,
            edge_order=2,
        )
        derivatives[(component, plane_axes[1])] = np.gradient(
            center_plane,
            spacing[plane_axes[1]],
            axis=1,
            edge_order=2,
        )
        derivatives[(component, normal_axis)] = (
            hi_planes[component] - lo_planes[component]
        ) / (2.0 * spacing[normal_axis])

    jx = derivatives[(2, 1)] - derivatives[(1, 2)]
    jy = derivatives[(0, 2)] - derivatives[(2, 0)]
    jz = derivatives[(1, 0)] - derivatives[(0, 1)]
    return center_index, np.sqrt(jx * jx + jy * jy + jz * jz)


def _write_baseline_slice_pngs(paths, bfield, spacing, fixture_path, metric, smoothing_passes):
    mid_x, jmag_x = _center_plane_jmag(bfield, spacing, 0, smoothing_passes=smoothing_passes)
    mid_y, jmag_y = _center_plane_jmag(bfield, spacing, 1, smoothing_passes=smoothing_passes)
    mid_z, jmag_z = _center_plane_jmag(bfield, spacing, 2, smoothing_passes=smoothing_passes)
    log_floor = CURRENT_EPSILON
    display_x = np.log10(jmag_x + log_floor)
    display_y = np.log10(jmag_y + log_floor)
    display_z = np.log10(jmag_z + log_floor)

    finite = np.concatenate(
        [
            np.asarray(display_x[np.isfinite(display_x)], dtype=np.double),
            np.asarray(display_y[np.isfinite(display_y)], dtype=np.double),
            np.asarray(display_z[np.isfinite(display_z)], dtype=np.double),
        ]
    )
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(finite, [1.0, 99.0])
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0

    metadata = {
        "mid_x": {
            "path": paths["mid_x"],
            "index": int(mid_x),
            "axes": ("y index", "z index"),
            "display": display_x,
        },
        "mid_y": {
            "path": paths["mid_y"],
            "index": int(mid_y),
            "axes": ("x index", "z index"),
            "display": display_y,
        },
        "mid_z": {
            "path": paths["mid_z"],
            "index": int(mid_z),
            "axes": ("x index", "y index"),
            "display": display_z,
        },
    }
    for name, item in metadata.items():
        axis = name[-1]
        title = (
            f"{fixture_path.name} mid-{axis} |J| index={item['index']} "
            f"smoothness={metric:.6g}"
        )
        _write_current_slice_png(
            item["path"],
            item["display"],
            title,
            item["axes"][0],
            item["axes"][1],
            "log10(|J| + eps)",
            vmin=vmin,
            vmax=vmax,
        )
    return {
        name: {
            "path": item["path"].name,
            "index": item["index"],
            "vmin": vmin,
            "vmax": vmax,
            "log_floor": log_floor,
        }
        for name, item in metadata.items()
    }


def _spacing_from_domain(xmin, xmax, resolution, convention):
    xmin = np.asarray(xmin, dtype=np.double)
    xmax = np.asarray(xmax, dtype=np.double)
    resolution = np.asarray(resolution, dtype=np.double)
    if convention == "domain_over_resolution_minus_one":
        denominator = resolution - 1.0
    elif convention == "domain_over_resolution":
        denominator = resolution
    else:
        raise ValueError(f"Unknown spacing convention: {convention}")
    if np.any(denominator <= 0.0):
        raise ValueError(f"Invalid resolution for spacing convention {convention}: {tuple(resolution)}")
    return (xmax - xmin) / denominator


def _synthetic_case_fields():
    shape = (9, 8, 7)
    spacing = np.array([0.25, 0.375, 0.5], dtype=np.double)
    xmin = np.array([-0.75, 1.25, -0.5], dtype=np.double)
    x = xmin[0] + spacing[0] * np.arange(shape[0], dtype=np.double)
    y = xmin[1] + spacing[1] * np.arange(shape[1], dtype=np.double)
    z = xmin[2] + spacing[2] * np.arange(shape[2], dtype=np.double)
    xx, yy, _ = np.meshgrid(x, y, z, indexing="ij")

    b_z_equals_x = np.zeros((3, *shape), dtype=np.double)
    b_z_equals_x[2] = xx
    expected_z_equals_x = np.zeros((3, shape[0], shape[1]), dtype=np.double)
    expected_z_equals_x[1] = -1.0

    b_rotation = np.zeros((3, *shape), dtype=np.double)
    b_rotation[0] = -0.5 * yy
    b_rotation[1] = 0.5 * xx
    expected_rotation = np.zeros((3, shape[0], shape[1]), dtype=np.double)
    expected_rotation[2] = 1.0

    return [
        {
            "name": "B_z_equals_x",
            "field_definition": "B = (0, 0, x)",
            "expected_curl": "(0, -1, 0)",
            "bfield": b_z_equals_x,
            "spacing": spacing,
            "expected_components": expected_z_equals_x,
        },
        {
            "name": "xy_rotation",
            "field_definition": "B = (-y / 2, x / 2, 0)",
            "expected_curl": "(0, 0, 1)",
            "bfield": b_rotation,
            "spacing": spacing,
            "expected_components": expected_rotation,
        },
    ]


def _evaluate_synthetic_controls():
    results = []
    for case in _synthetic_case_fields():
        center_z, components, magnitude = _current_center_components(case["bfield"], case["spacing"])
        expected_components = case["expected_components"]
        expected_magnitude = np.sqrt(np.sum(expected_components * expected_components, axis=0))
        component_error = np.abs(components - expected_components)
        magnitude_error = np.abs(magnitude - expected_magnitude)
        max_component_error = float(np.max(component_error))
        mean_component_error = float(np.mean(component_error))
        max_magnitude_error = float(np.max(magnitude_error))
        mean_magnitude_error = float(np.mean(magnitude_error))
        results.append(
            {
                "name": case["name"],
                "field_definition": case["field_definition"],
                "expected_curl": case["expected_curl"],
                "shape": tuple(int(value) for value in case["bfield"].shape),
                "spacing": tuple(float(value) for value in case["spacing"]),
                "center_z": int(center_z),
                "max_component_error": max_component_error,
                "mean_component_error": mean_component_error,
                "max_magnitude_error": max_magnitude_error,
                "mean_magnitude_error": mean_magnitude_error,
                "tolerance": SYNTHETIC_TOLERANCE,
                "passed": bool(
                    max_component_error <= SYNTHETIC_TOLERANCE
                    and max_magnitude_error <= SYNTHETIC_TOLERANCE
                ),
            }
        )
    return results


def _current_percentiles(jmag_slice):
    finite = np.asarray(jmag_slice[np.isfinite(jmag_slice)], dtype=np.double)
    percentiles = {"finite_count": int(finite.size)}
    if finite.size == 0:
        for key in ("p0", "p50", "p95", "p99", "p100"):
            percentiles[key] = float("nan")
        return percentiles

    values = np.percentile(finite, [0.0, 50.0, 95.0, 99.0, 100.0])
    for key, value in zip(("p0", "p50", "p95", "p99", "p100"), values):
        percentiles[key] = float(value)
    return percentiles


def _variant_definitions():
    return [
        {
            "name": "minus_one_no_smoothing",
            "spacing_convention": "domain_over_resolution_minus_one",
            "smoothing_passes": 0,
            "axis_interpretation": "standard_xyz",
        },
        {
            "name": "minus_one_two_pass_smoothing",
            "spacing_convention": "domain_over_resolution_minus_one",
            "smoothing_passes": EXISTING_SMOOTHING_PASSES,
            "axis_interpretation": "standard_xyz",
            "baseline": True,
        },
        {
            "name": "cell_width_no_smoothing",
            "spacing_convention": "domain_over_resolution",
            "smoothing_passes": 0,
            "axis_interpretation": "standard_xyz",
        },
        {
            "name": "cell_width_two_pass_smoothing",
            "spacing_convention": "domain_over_resolution",
            "smoothing_passes": EXISTING_SMOOTHING_PASSES,
            "axis_interpretation": "standard_xyz",
        },
        {
            "name": "swap_xy_two_pass_smoothing",
            "spacing_convention": "domain_over_resolution_minus_one",
            "smoothing_passes": EXISTING_SMOOTHING_PASSES,
            "axis_interpretation": "swap_xy",
        },
    ]


def _classify_variant(variant, baseline_metric):
    metric = variant["smoothness_ratio"]
    threshold_status = "passes existing threshold" if _metric_passed(metric) else "exceeds existing threshold"
    if not np.isfinite(metric):
        return "invalid metric"
    if variant.get("baseline"):
        return f"baseline; {threshold_status}"

    ratio = _metric_ratio(metric, baseline_metric)
    if _metric_passed(metric) != _metric_passed(baseline_metric):
        return f"{threshold_status}; threshold outcome changes versus baseline"
    if ratio >= CONVENTION_SENSITIVITY_RATIO:
        return f"{threshold_status}; metric shifts {_format_number(ratio)}x versus baseline"
    return f"{threshold_status}; comparable to baseline"


def _evaluate_weno_variants(fixture_path):
    ds = open_dataset(str(fixture_path), ghost_width=GHOST_WIDTH)
    ds.load_data(field_indices=FIELD_INDICES)
    ds.exchange_ghost_cells()

    resolution = ds.domain_nx.astype(np.int64) * (2 ** (int(ds.levmax) - 1))
    physical_domain = np.asarray(ds.physical_domain, dtype=np.double)
    bfield = ds.uniform_grid(resolution, field_indices=FIELD_INDICES, interpolation="linear")

    variants = []
    for definition in _variant_definitions():
        spacing = _spacing_from_domain(
            physical_domain[0],
            physical_domain[1],
            resolution,
            definition["spacing_convention"],
        )
        slice_index, _, jmag_slice = _current_center_components(
            bfield,
            spacing,
            smoothing_passes=definition["smoothing_passes"],
            axis_interpretation=definition["axis_interpretation"],
        )
        effective_spacing = spacing
        if definition["axis_interpretation"] == "swap_xy":
            effective_spacing = spacing[[1, 0, 2]]
        metric = _current_smoothness_metric(jmag_slice, effective_spacing)
        variant = {
            **definition,
            "center_z": int(slice_index),
            "finite_shape": tuple(int(value) for value in jmag_slice.shape),
            "spacing": tuple(float(value) for value in spacing),
            "percentiles": _current_percentiles(jmag_slice),
            "smoothness_ratio": metric,
        }
        if definition.get("baseline"):
            variant["slice_pngs"] = _write_baseline_slice_pngs(
                BASELINE_SLICE_PNGS,
                bfield,
                spacing,
                fixture_path,
                metric,
                definition["smoothing_passes"],
            )
        variants.append(variant)

    baseline_metric = next(
        variant["smoothness_ratio"] for variant in variants if variant.get("baseline")
    )
    for variant in variants:
        variant["classification"] = _classify_variant(variant, baseline_metric)

    return {
        "status": "evaluated",
        "fixture_path": str(fixture_path),
        "field_indices": list(FIELD_INDICES),
        "field_names": [ds.wnames[index] for index in FIELD_INDICES],
        "ghost_width": GHOST_WIDTH,
        "resolution": tuple(int(value) for value in resolution),
        "physical_domain": (
            tuple(float(value) for value in physical_domain[0]),
            tuple(float(value) for value in physical_domain[1]),
        ),
        "variants": variants,
    }


def _find_variant(weno, spacing_convention, smoothing_passes, axis_interpretation):
    if weno.get("status") != "evaluated":
        return None
    for variant in weno["variants"]:
        if (
            variant["spacing_convention"] == spacing_convention
            and variant["smoothing_passes"] == smoothing_passes
            and variant["axis_interpretation"] == axis_interpretation
        ):
            return variant
    return None


def _diagnose(synthetic_results, weno):
    if not synthetic_results or not all(result.get("passed", False) for result in synthetic_results):
        return (
            "synthetic curl controls failed; diagnose a current math or axis/spacing "
            "convention bug before evaluating the WENO fixture"
        )

    if weno.get("status") == "not evaluated":
        return "partial diagnosis: synthetic curl controls pass; WENO fixture was not evaluated"
    if weno.get("status") != "evaluated":
        return f"diagnosis blocked: WENO setup failed with {weno.get('exception_type', 'unknown error')}"

    baseline = _find_variant(
        weno,
        "domain_over_resolution_minus_one",
        EXISTING_SMOOTHING_PASSES,
        "standard_xyz",
    )
    cell_width = _find_variant(
        weno,
        "domain_over_resolution",
        EXISTING_SMOOTHING_PASSES,
        "standard_xyz",
    )
    no_smoothing = _find_variant(
        weno,
        "domain_over_resolution_minus_one",
        0,
        "standard_xyz",
    )
    axis_swapped = _find_variant(
        weno,
        "domain_over_resolution_minus_one",
        EXISTING_SMOOTHING_PASSES,
        "swap_xy",
    )

    convention_sensitive = False
    if baseline is not None:
        for variant in (cell_width, axis_swapped):
            if variant is None:
                continue
            ratio = _metric_ratio(variant["smoothness_ratio"], baseline["smoothness_ratio"])
            if (
                _metric_passed(variant["smoothness_ratio"])
                != _metric_passed(baseline["smoothness_ratio"])
                or ratio >= CONVENTION_SENSITIVITY_RATIO
            ):
                convention_sensitive = True

    smoothing_sensitive = False
    if baseline is not None and no_smoothing is not None:
        smoothing_sensitive = (
            _metric_passed(no_smoothing["smoothness_ratio"])
            != _metric_passed(baseline["smoothness_ratio"])
        )

    if convention_sensitive and smoothing_sensitive:
        return (
            "synthetic curl controls pass; WENO metrics are convention-sensitive and "
            "smoothing changes threshold behavior"
        )
    if convention_sensitive:
        return "synthetic curl controls pass; WENO failure is convention-sensitive"
    if smoothing_sensitive:
        return "synthetic curl controls pass; WENO failure is smoothing or metric-sensitive"

    evaluated_metrics = [variant["smoothness_ratio"] for variant in weno["variants"]]
    if evaluated_metrics and all(not _metric_passed(metric) for metric in evaluated_metrics):
        return (
            "synthetic curl controls pass; WENO metrics remain high across controlled "
            "variants, pointing to a real field/sampling artifact or a smoothness "
            "metric that is not aligned with intended visual quality"
        )

    return "synthetic curl controls pass; at least one WENO variant satisfies the existing smoothness threshold"


def _synthetic_report_table(synthetic_results):
    lines = [
        "| field | expected curl | shape | center z | max component error | mean component error | max magnitude error | result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in synthetic_results:
        status = "PASS" if result.get("passed", False) else "FAIL"
        lines.append(
            "| "
            f"{result.get('field_definition', result.get('name', 'unknown'))} | "
            f"{result.get('expected_curl', 'unknown')} | "
            f"{result.get('shape', 'unknown')} | "
            f"{result.get('center_z', 'not evaluated')} | "
            f"{_format_number(result.get('max_component_error'))} | "
            f"{_format_number(result.get('mean_component_error'))} | "
            f"{_format_number(result.get('max_magnitude_error'))} | "
            f"{status} |"
        )
    return "\n".join(lines)


def _weno_report_section(weno):
    if weno.get("status") == "not evaluated":
        return f"""## WENO Fixture Variants
- status: `not evaluated`
- reason: `{weno.get('reason', 'fixture not available')}`
"""

    if weno.get("status") != "evaluated":
        return f"""## WENO Fixture Variants
- status: `error`
- exception: `{weno.get('exception_type', 'unknown')}: {weno.get('exception_message', '')}`
"""

    lines = [
        "## WENO Fixture Variants",
        f"- field indices: `{weno['field_indices']}`",
        f"- field names: `{weno['field_names']}`",
        f"- ghost width: `{weno['ghost_width']}`",
        f"- uniform-grid resolution: `{weno['resolution']}`",
        f"- physical domain: `{weno['physical_domain']}`",
        "",
        "| variant | spacing convention | smoothing passes | axis interpretation | finite count | J magnitude p0/p50/p95/p99/max | smoothness ratio | classification |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant in weno["variants"]:
        lines.append(
            "| "
            f"{variant['name']} | "
            f"{variant['spacing_convention']} | "
            f"{variant['smoothing_passes']} | "
            f"{variant['axis_interpretation']} | "
            f"{variant['percentiles']['finite_count']} | "
            f"{_format_percentiles(variant['percentiles'])} | "
            f"{_format_number(variant['smoothness_ratio'])} | "
            f"{variant['classification']} |"
        )

    baseline = next((variant for variant in weno["variants"] if variant.get("baseline")), None)
    if baseline is not None and "slice_pngs" in baseline:
        pngs = baseline["slice_pngs"]
        first_png = pngs["mid_x"]
        lines.extend(
            [
                "",
                "## Baseline Central |J| Slices",
                f"- variant: `{baseline['name']}`",
                f"- smoothing passes: `{baseline['smoothing_passes']}`",
                f"- smoothness ratio from mid-z metric: `{_format_number(baseline['smoothness_ratio'])}`",
                "- display transform: `log10(|J| + eps)` for the images only",
                "- colormap: `plasma` with shared p1-p99 clipping across the three displayed slices",
                f"- log display range: `{_format_number(first_png['vmin'])}` to `{_format_number(first_png['vmax'])}`",
                "",
                "| slice | index | image path |",
                "| --- | --- | --- |",
            ]
        )
        for name in ("mid_x", "mid_y", "mid_z"):
            item = pngs[name]
            lines.append(f"| `{name}` | `{item['index']}` | `{item['path']}` |")
        lines.extend(
            [
                "",
                f"![baseline mid-x J magnitude]({pngs['mid_x']['path']})",
                "",
                f"![baseline mid-y J magnitude]({pngs['mid_y']['path']})",
                "",
                f"![baseline mid-z J magnitude]({pngs['mid_z']['path']})",
            ]
        )
    return "\n".join(lines)


def _write_report(synthetic_results, weno, diagnosis):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""# AMRVAC Current-Density Diagnostic

## Command Gate
- fixture path: `{FIXTURE_PATH}`
- fixture available: `{FIXTURE_PATH.exists()}`
- enable command: `{ENABLE_COMMAND}`
- heavy gate: `SIMESH_RUN_HEAVY_TESTS=1`
- existing current threshold: `{CURRENT_THRESHOLD:.12g}`
- convention sensitivity ratio: `{CONVENTION_SENSITIVITY_RATIO:.12g}`

## Synthetic Curl Controls
- layout: `(3, nx, ny, nz)` for `(Bx, By, Bz)`
- spacing: synthetic cases use explicit Cartesian spacings and the same center-slice curl helper shape as the heavy validation
- tolerance: `{SYNTHETIC_TOLERANCE:.12g}`

{_synthetic_report_table(synthetic_results)}

{_weno_report_section(weno)}

## Final Diagnosis
{diagnosis}
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def test_heavy_amrvac_current_diagnostic():
    if os.environ.get("SIMESH_RUN_HEAVY_TESTS") != "1":
        print("Skipping AMRVAC current diagnostic: set SIMESH_RUN_HEAVY_TESTS=1 to enable.")
        return

    synthetic_exception = None
    weno_exception = None
    try:
        synthetic_results = _evaluate_synthetic_controls()
    except Exception as exc:
        synthetic_exception = exc
        synthetic_results = [
            {
                "name": "synthetic_controls",
                "field_definition": "synthetic controls",
                "expected_curl": "not evaluated",
                "passed": False,
                "max_component_error": float("inf"),
                "mean_component_error": float("inf"),
                "max_magnitude_error": float("inf"),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        ]

    synthetic_passed = synthetic_exception is None and all(result["passed"] for result in synthetic_results)
    if not synthetic_passed:
        weno = {
            "status": "not evaluated",
            "fixture_path": str(FIXTURE_PATH),
            "reason": "synthetic controls failed",
        }
    elif not FIXTURE_PATH.exists():
        weno = {
            "status": "not evaluated",
            "fixture_path": str(FIXTURE_PATH),
            "reason": "fixture missing",
        }
    else:
        try:
            weno = _evaluate_weno_variants(FIXTURE_PATH)
        except Exception as exc:
            weno_exception = exc
            weno = {
                "status": "error",
                "fixture_path": str(FIXTURE_PATH),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }

    diagnosis = _diagnose(synthetic_results, weno)
    _write_report(synthetic_results, weno, diagnosis)

    if synthetic_exception is not None:
        raise AssertionError(f"Synthetic current diagnostic errored. See {REPORT_PATH}.") from synthetic_exception

    failed_synthetic = [result["name"] for result in synthetic_results if not result["passed"]]
    assert not failed_synthetic, (
        f"Synthetic current diagnostic failed for {failed_synthetic}. "
        f"See {REPORT_PATH}."
    )

    if weno_exception is not None:
        raise AssertionError(f"WENO current diagnostic setup failed. See {REPORT_PATH}.") from weno_exception


def run_tests():
    test_heavy_amrvac_current_diagnostic()
    print("test_heavy_amrvac_current_diagnostic passed or skipped")


if __name__ == "__main__":
    run_tests()
