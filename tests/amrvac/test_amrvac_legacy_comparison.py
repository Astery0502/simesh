import os
from pathlib import Path

import numpy as np

from simesh.amrvac import open_dataset
from simesh.legacy.frontends.amrvac.io import amr_loader


FIXTURE_PATH = Path("data/weno509_sub_0000.dat")
REPORT_DIR = Path("report/amrvac-legacy-comparison")
REPORT_PATH = REPORT_DIR / "report.md"
FIELD_INDICES = [4, 5, 6]
FIELD_SLICE = slice(4, 7)
REJECTED_GHOST_WIDTH = 1
CURRENT_GHOST_WIDTH = 2
LEGACY_GHOST_WIDTH = 2
GHOST_RTOL = 1e-12
GHOST_ATOL = 1e-12
UNIFORM_RTOL = 1e-10
UNIFORM_ATOL = 1e-10
ENABLE_COMMAND = (
    "SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src "
    ".venv/bin/python tests/amrvac/test_amrvac_legacy_comparison.py"
)

AXIS_NAMES = ("x", "y", "z")
NEIGHBOR_TYPE_NAMES = {
    1: "boundary",
    2: "coarse",
    3: "sibling",
    4: "fine",
}


def _format_tuple(values):
    return tuple(float(value) for value in np.asarray(values).ravel())


def _comparison_metrics(current, legacy, *, rtol, atol):
    current = np.asarray(current, dtype=np.double)
    legacy = np.asarray(legacy, dtype=np.double)

    if current.shape != legacy.shape:
        return {
            "shape_current": tuple(int(value) for value in current.shape),
            "shape_legacy": tuple(int(value) for value in legacy.shape),
            "finite_current": int(np.isfinite(current).sum()),
            "finite_legacy": int(np.isfinite(legacy).sum()),
            "finite_both": 0,
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "legacy_scale": float("nan"),
            "relative_error": float("inf"),
            "rtol": rtol,
            "atol": atol,
            "passed": False,
        }

    finite_current = np.isfinite(current)
    finite_legacy = np.isfinite(legacy)
    finite_both = finite_current & finite_legacy

    if np.any(finite_both):
        abs_error = np.abs(current[finite_both] - legacy[finite_both])
        max_abs_error = float(np.max(abs_error))
        mean_abs_error = float(np.mean(abs_error))
        legacy_scale = float(np.max(np.abs(legacy[finite_both])))
    else:
        max_abs_error = float("inf")
        mean_abs_error = float("inf")
        legacy_scale = float("nan")

    if legacy_scale > 0.0:
        relative_error = max_abs_error / legacy_scale
    elif max_abs_error == 0.0:
        relative_error = 0.0
    else:
        relative_error = float("inf")

    all_finite = bool(np.all(finite_current) and np.all(finite_legacy))
    tolerance = atol + rtol * (legacy_scale if legacy_scale > 0.0 else 0.0)
    passed = all_finite and max_abs_error <= tolerance

    return {
        "shape_current": tuple(int(value) for value in current.shape),
        "shape_legacy": tuple(int(value) for value in legacy.shape),
        "finite_current": int(finite_current.sum()),
        "finite_legacy": int(finite_legacy.sum()),
        "finite_both": int(finite_both.sum()),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "legacy_scale": legacy_scale,
        "relative_error": relative_error,
        "rtol": rtol,
        "atol": atol,
        "passed": bool(passed),
    }


def _legacy_neighbor_type_as_current_layout(legacy_forest):
    nleafs = legacy_forest.neighbor_type.shape[3]
    neighbor_type = np.zeros((nleafs, 27), dtype=np.uint32)
    for i, j, k in np.ndindex(3, 3, 3):
        neighbor_type[:, i + j * 3 + k * 9] = legacy_forest.neighbor_type[i, j, k, :]
    return neighbor_type


def _field_names(current_ds):
    return [current_ds.wnames[index] for index in FIELD_INDICES]


def _evaluate_width_policy():
    try:
        current_ds = open_dataset(str(FIXTURE_PATH), ghost_width=REJECTED_GHOST_WIDTH)
    except ValueError as exc:
        message = str(exc)
        passed = "ghost_width must be 0 or >= 2" in message
        return {
            "name": "Width-1 ghost policy check",
            "status": "evaluated",
            "ghost_width": REJECTED_GHOST_WIDTH,
            "has_coarse_or_fine_interfaces": None,
            "exception_type": type(exc).__name__,
            "exception_message": message,
            "passed": bool(passed),
        }
    except Exception as exc:
        return {
            "name": "Width-1 ghost policy check",
            "status": "evaluated",
            "ghost_width": REJECTED_GHOST_WIDTH,
            "has_coarse_or_fine_interfaces": None,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "passed": False,
        }

    neighbor_type = np.asarray(current_ds.forest.neighbor_type)
    has_coarse_or_fine = bool(np.any((neighbor_type == 2) | (neighbor_type == 4)))

    try:
        current_ds.load_data(field_indices=FIELD_INDICES)
    except ValueError as exc:
        message = str(exc)
        passed = has_coarse_or_fine and "ghost_width >= 2" in message
        return {
            "name": "Width-1 ghost policy check",
            "status": "evaluated",
            "ghost_width": REJECTED_GHOST_WIDTH,
            "has_coarse_or_fine_interfaces": has_coarse_or_fine,
            "exception_type": type(exc).__name__,
            "exception_message": message,
            "passed": bool(passed),
        }
    except Exception as exc:
        return {
            "name": "Width-1 ghost policy check",
            "status": "evaluated",
            "ghost_width": REJECTED_GHOST_WIDTH,
            "has_coarse_or_fine_interfaces": has_coarse_or_fine,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "passed": False,
        }

    return {
        "name": "Width-1 ghost policy check",
        "status": "evaluated",
        "ghost_width": REJECTED_GHOST_WIDTH,
        "has_coarse_or_fine_interfaces": has_coarse_or_fine,
        "exception_type": "none",
        "exception_message": "ghost_width=1 was accepted",
        "passed": False,
    }


def _load_legacy_with_getbc():
    try:
        legacy_ds = amr_loader(str(FIXTURE_PATH), nghostcells=LEGACY_GHOST_WIDTH, load_ghost=False)
        legacy_ds.mesh.getbc()
    except Exception as exc:
        failure = f"legacy `getbc()` failed with `nghostcells={LEGACY_GHOST_WIDTH}`: {type(exc).__name__}: {exc}"
        return None, [], [failure]

    notes = [f"legacy comparison uses `nghostcells={LEGACY_GHOST_WIDTH}` to match the current refined ghost-width policy"]
    return legacy_ds, notes, []


def _validate_setup(current_ds, legacy_ds):
    failures = []
    current_header = current_ds.metadata
    legacy_header = legacy_ds.header

    for key in ("ndim", "nw", "nleafs", "nparents", "levmax", "geometry", "staggered"):
        if current_header[key] != legacy_header[key]:
            failures.append(f"header `{key}` differs: {current_header[key]!r} != {legacy_header[key]!r}")

    for key in ("domain_nx", "block_nx", "xmin", "xmax", "periodic"):
        if not np.array_equal(np.asarray(current_header[key]), np.asarray(legacy_header[key])):
            failures.append(f"header `{key}` differs")

    if list(current_header["w_names"]) != list(legacy_header["w_names"]):
        failures.append("header `w_names` differs")

    if not np.array_equal(current_ds.is_leaf.astype(bool), np.asarray(legacy_ds.forest, dtype=bool)):
        failures.append("forest leaf flags differ")

    for name, current_value, legacy_value in (
        ("block levels", current_ds.tree_info[0], legacy_ds.tree[0]),
        ("block ids", current_ds.tree_info[1], legacy_ds.tree[1]),
        ("block offsets", current_ds.tree_info[2], legacy_ds.tree[2]),
    ):
        if not np.array_equal(np.asarray(current_value), np.asarray(legacy_value)):
            failures.append(f"tree {name} differ")

    current_neighbor_type = np.asarray(current_ds.forest.neighbor_type)
    legacy_neighbor_type = _legacy_neighbor_type_as_current_layout(legacy_ds.mesh.forest)
    if not np.array_equal(current_neighbor_type, legacy_neighbor_type):
        failures.append("current and legacy neighbor connectivity differ")

    current_interior = current_ds.blocks(include_ghosts=False)
    legacy_interior = np.transpose(
        legacy_ds.mesh.data[
            :,
            legacy_ds.mesh.ixMmin[0] : legacy_ds.mesh.ixMmax[0] + 1,
            legacy_ds.mesh.ixMmin[1] : legacy_ds.mesh.ixMmax[1] + 1,
            legacy_ds.mesh.ixMmin[2] : legacy_ds.mesh.ixMmax[2] + 1,
            FIELD_SLICE,
        ],
        (0, 4, 1, 2, 3),
    )
    if not np.array_equal(current_interior, legacy_interior):
        failures.append("loaded interior block values differ for fields [4, 5, 6]")

    return failures


def _neighbor_index(axis, side):
    offset = [1, 1, 1]
    offset[axis] = 0 if side == 0 else 2
    return offset[0] + offset[1] * 3 + offset[2] * 9


def _adjacent_ghost_index(block_cells, ghost_width, side):
    if side == 0:
        return int(ghost_width) - 1
    return int(ghost_width) + int(block_cells)


def _select_region(current_ds, legacy_ds):
    neighbor_type = np.asarray(current_ds.forest.neighbor_type)
    current_width = int(current_ds.ghost_width)
    legacy_width = int(legacy_ds.mesh.nghostcells)

    for preferred_type in (2, 4, 3):
        for ileaf in range(neighbor_type.shape[0]):
            for axis in range(int(current_ds.ndim)):
                for side in (0, 1):
                    index = _neighbor_index(axis, side)
                    found_type = int(neighbor_type[ileaf, index])
                    if found_type != preferred_type:
                        continue

                    current_ghost_index = _adjacent_ghost_index(current_ds.block_nx[axis], current_width, side)
                    legacy_ghost_index = _adjacent_ghost_index(legacy_ds.mesh.block_nx[axis], legacy_width, side)
                    block_bounds = np.asarray(current_ds.mesh.rnode)[ileaf]
                    return {
                        "leaf_index": int(ileaf),
                        "axis": int(axis),
                        "axis_name": AXIS_NAMES[axis],
                        "side": "lo" if side == 0 else "hi",
                        "neighbor_type": found_type,
                        "neighbor_type_name": NEIGHBOR_TYPE_NAMES[found_type],
                        "neighbor_index": int(index),
                        "current_ghost_index": int(current_ghost_index),
                        "legacy_ghost_index": int(legacy_ghost_index),
                        "current_ghost_width": int(current_width),
                        "legacy_ghost_width": int(legacy_width),
                        "block_nx": np.asarray(current_ds.block_nx, dtype=np.int64).copy(),
                        "block_xmin": block_bounds[0:3].copy(),
                        "block_xmax": block_bounds[3:6].copy(),
                        "block_dx": block_bounds[6:9].copy(),
                    }

    raise AssertionError("comparison setup invalid: no internal block face was available for comparison")


def _current_ghost_plane(current_blocks, region):
    selection = [region["leaf_index"], slice(None), slice(None), slice(None), slice(None)]
    selection[2 + region["axis"]] = region["current_ghost_index"]
    return current_blocks[tuple(selection)]


def _legacy_ghost_plane(legacy_data, region):
    selection = [region["leaf_index"], slice(None), slice(None), slice(None), slice(None)]
    crop_start = int(region["legacy_ghost_width"]) - int(region["current_ghost_width"])
    for axis in range(3):
        if axis == region["axis"]:
            selection[1 + axis] = region["legacy_ghost_index"]
        else:
            crop_stop = crop_start + int(region["block_nx"][axis]) + 2 * int(region["current_ghost_width"])
            selection[1 + axis] = slice(crop_start, crop_stop)
    plane = legacy_data[tuple(selection)]
    return np.moveaxis(plane[..., FIELD_SLICE], -1, 0)


def _uniform_region(current_ds, region):
    axis = region["axis"]
    xmin = region["block_xmin"].copy()
    xmax = region["block_xmax"].copy()
    dx = region["block_dx"]
    nx = np.asarray(current_ds.block_nx, dtype=np.int64).copy()

    if region["side"] == "lo":
        center = region["block_xmin"][axis] + 0.25 * dx[axis]
    else:
        center = region["block_xmax"][axis] - 0.25 * dx[axis]

    xmin[axis] = center - 0.5 * dx[axis]
    xmax[axis] = center + 0.5 * dx[axis]
    nx[axis] = 1
    return xmin, xmax, nx


def _evaluate_stage_a(current_ds, legacy_ds, region):
    current_plane = _current_ghost_plane(current_ds.blocks(include_ghosts=True), region)
    legacy_plane = _legacy_ghost_plane(legacy_ds.mesh.data, region)
    return {
        "name": "Stage A: ghost-cell slice comparison",
        "status": "evaluated",
        "orientation": f"{region['axis_name']}-{region['side']}",
        "slice_index": region["current_ghost_index"],
        "legacy_slice_index": region["legacy_ghost_index"],
        "metrics": _comparison_metrics(current_plane, legacy_plane, rtol=GHOST_RTOL, atol=GHOST_ATOL),
    }


def _evaluate_stage_b(current_ds, legacy_ds, region):
    xmin, xmax, nx = _uniform_region(current_ds, region)

    current_uniform = current_ds.uniform_grid(
        nx,
        xmin=xmin,
        xmax=xmax,
        field_indices=FIELD_INDICES,
        interpolation="linear",
    )

    try:
        legacy_uniform_udata = legacy_ds.mesh.export_uniform(
            legacy_ds.mesh.data[..., FIELD_SLICE],
            xmin,
            xmax,
            int(nx[0]),
            int(nx[1]),
            int(nx[2]),
        )
    except Exception as exc:
        return {
            "name": "Stage B: uniform interpolation slice comparison",
            "status": "not evaluated",
            "reason": f"legacy uniform export failed: {type(exc).__name__}: {exc}",
            "uniform_xmin": xmin,
            "uniform_xmax": xmax,
            "uniform_resolution": nx,
        }

    legacy_uniform = np.moveaxis(legacy_uniform_udata, -1, 0)
    return {
        "name": "Stage B: uniform interpolation slice comparison",
        "status": "evaluated",
        "orientation": f"{region['axis_name']}-{region['side']} near-interface plane",
        "slice_index": 0,
        "uniform_xmin": xmin,
        "uniform_xmax": xmax,
        "uniform_resolution": nx,
        "metrics": _comparison_metrics(current_uniform, legacy_uniform, rtol=UNIFORM_RTOL, atol=UNIFORM_ATOL),
    }


def _diagnosis(policy_stage, setup_failures, stage_a, stage_b):
    if not policy_stage["passed"]:
        return "comparison setup invalid: width-1 refined ghost policy check failed"

    if setup_failures:
        return "comparison setup invalid"

    if not stage_a["metrics"]["passed"]:
        return "first evaluated divergence: ghost-cell exchange"

    if stage_b["status"] == "not evaluated":
        return "partial diagnosis: ghost-cell exchange matches legacy; uniform interpolation was not evaluated"

    if not stage_b["metrics"]["passed"]:
        return "first evaluated divergence: uniform interpolation"

    return "current and legacy paths match through the evaluated interpolation slice; any remaining artifact is later than interpolation"


def _policy_report(policy_stage):
    result = "PASS" if policy_stage["passed"] else "FAIL"
    return f"""## {policy_stage['name']}
- status: `{policy_stage['status']}`
- rejected ghost width: `{policy_stage['ghost_width']}`
- coarse/fine interfaces detected: `{policy_stage['has_coarse_or_fine_interfaces']}`
- observed exception type: `{policy_stage['exception_type']}`
- observed exception message: `{policy_stage['exception_message']}`
- result: `{result}`"""


def _stage_report(stage):
    lines = [f"## {stage['name']}"]
    lines.append(f"- status: `{stage['status']}`")

    if stage["status"] == "not evaluated":
        lines.append(f"- reason: {stage['reason']}")
        if "uniform_resolution" in stage:
            lines.append(f"- uniform resolution: `{tuple(int(value) for value in stage['uniform_resolution'])}`")
            lines.append(f"- uniform xmin: `{_format_tuple(stage['uniform_xmin'])}`")
            lines.append(f"- uniform xmax: `{_format_tuple(stage['uniform_xmax'])}`")
        return "\n".join(lines)

    metrics = stage["metrics"]
    lines.extend(
        [
            f"- orientation: `{stage['orientation']}`",
            f"- slice index: `{stage['slice_index']}`",
            f"- current shape: `{metrics['shape_current']}`",
            f"- legacy shape: `{metrics['shape_legacy']}`",
            f"- finite current values: `{metrics['finite_current']}`",
            f"- finite legacy values: `{metrics['finite_legacy']}`",
            f"- finite comparable values: `{metrics['finite_both']}`",
            f"- maximum absolute error: `{metrics['max_abs_error']:.12g}`",
            f"- mean absolute error: `{metrics['mean_abs_error']:.12g}`",
            f"- legacy scale: `{metrics['legacy_scale']:.12g}`",
            f"- relative error: `{metrics['relative_error']:.12g}`",
            f"- tolerance: `rtol={metrics['rtol']:.1e}, atol={metrics['atol']:.1e}`",
            f"- result: `{'PASS' if metrics['passed'] else 'FAIL'}`",
        ]
    )
    if "legacy_slice_index" in stage:
        lines.append(f"- legacy slice index: `{stage['legacy_slice_index']}`")
    if "uniform_resolution" in stage:
        lines.append(f"- uniform resolution: `{tuple(int(value) for value in stage['uniform_resolution'])}`")
        lines.append(f"- uniform xmin: `{_format_tuple(stage['uniform_xmin'])}`")
        lines.append(f"- uniform xmax: `{_format_tuple(stage['uniform_xmax'])}`")
    return "\n".join(lines)


def _write_report(current_ds, legacy_ds, region, policy_stage, setup_failures, setup_notes, stage_a, stage_b, diagnosis):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected_region = "not selected because comparison setup was invalid"
    if region is not None:
        selected_region = f"""
- selected leaf index: `{region['leaf_index']}`
- selected interface: `{region['axis_name']}-{region['side']}`
- neighbor type: `{region['neighbor_type_name']}` (`{region['neighbor_type']}`)
- current ghost slice index: `{region['current_ghost_index']}`
- legacy ghost slice index: `{region['legacy_ghost_index']}`
- current ghost width: `{region['current_ghost_width']}`
- legacy ghost width: `{region['legacy_ghost_width']}`
- block cell counts: `{tuple(int(value) for value in region['block_nx'])}`
- block xmin: `{_format_tuple(region['block_xmin'])}`
- block xmax: `{_format_tuple(region['block_xmax'])}`
- block cell widths: `{_format_tuple(region['block_dx'])}`
""".strip()

    setup_result = "valid" if not setup_failures else "invalid"
    setup_lines = "\n".join(f"- {failure}" for failure in setup_failures) if setup_failures else "- no setup validation failures"
    setup_note_lines = "\n".join(f"- {note}" for note in setup_notes) if setup_notes else "- no setup notes"
    legacy_width = int(legacy_ds.mesh.nghostcells) if legacy_ds is not None else "unavailable"

    content = f"""# AMRVAC Legacy Comparison

## Fixture
- fixture path: `{FIXTURE_PATH}`
- command gate: `{ENABLE_COMMAND}`
- report path: `{REPORT_PATH}`

## Current Implementation Setup
- loader: `simesh.amrvac.open_dataset`
- ghost width: `{CURRENT_GHOST_WIDTH}`
- loaded field indices: `{FIELD_INDICES}`
- loaded field names: `{_field_names(current_ds)}`
- ghost refresh: `exchange_ghost_cells()`
- ghost-padded access: `blocks(include_ghosts=True)`
- uniform interpolation: `uniform_grid(..., interpolation="linear")`

{_policy_report(policy_stage)}

## Legacy Implementation Setup
- loader: `simesh.legacy.frontends.amrvac.io.amr_loader`
- ghost width: `{legacy_width}`
- loaded fields: all legacy file fields, compared with slice `{FIELD_INDICES}`
- ghost refresh: `mesh.getbc()`
- ghost-padded access: `mesh.data`
- uniform interpolation: `mesh.export_uniform(...)`

## Setup Validation
- result: `{setup_result}`
{setup_lines}
- notes:
{setup_note_lines}

## Selected Diagnostic Region
{selected_region}

{_stage_report(stage_a)}

{_stage_report(stage_b)}

## Final Diagnosis
{diagnosis}
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def test_heavy_amrvac_legacy_comparison():
    if os.environ.get("SIMESH_RUN_HEAVY_TESTS") != "1":
        print("Skipping AMRVAC legacy comparison: set SIMESH_RUN_HEAVY_TESTS=1 to enable.")
        return

    if not FIXTURE_PATH.exists():
        print(f"Skipping AMRVAC legacy comparison: fixture not found at {FIXTURE_PATH}.")
        return

    policy_stage = _evaluate_width_policy()

    current_ds = open_dataset(str(FIXTURE_PATH), ghost_width=CURRENT_GHOST_WIDTH)
    current_ds.load_data(field_indices=FIELD_INDICES)
    current_ds.exchange_ghost_cells()

    legacy_ds, setup_notes, legacy_failures = _load_legacy_with_getbc()
    if legacy_ds is None:
        setup_failures = legacy_failures
    else:
        setup_failures = legacy_failures or _validate_setup(current_ds, legacy_ds)

    region = None
    if setup_failures:
        stage_a = {
            "name": "Stage A: ghost-cell slice comparison",
            "status": "not evaluated",
            "reason": "comparison setup invalid",
        }
        stage_b = {
            "name": "Stage B: uniform interpolation slice comparison",
            "status": "not evaluated",
            "reason": "comparison setup invalid",
        }
    else:
        region = _select_region(current_ds, legacy_ds)
        stage_a = _evaluate_stage_a(current_ds, legacy_ds, region)
        if stage_a["metrics"]["passed"]:
            stage_b = _evaluate_stage_b(current_ds, legacy_ds, region)
        else:
            stage_b = {
                "name": "Stage B: uniform interpolation slice comparison",
                "status": "not evaluated",
                "reason": "Stage A failed; earliest divergence is already ghost-cell exchange",
            }

    diagnosis = _diagnosis(policy_stage, setup_failures, stage_a, stage_b)
    _write_report(current_ds, legacy_ds, region, policy_stage, setup_failures, setup_notes, stage_a, stage_b, diagnosis)

    assert policy_stage["passed"], f"AMRVAC width-1 policy check failed. See {REPORT_PATH}."
    assert not setup_failures, f"AMRVAC legacy comparison setup invalid. See {REPORT_PATH}."
    assert stage_a["metrics"]["passed"], (
        "AMRVAC legacy comparison found ghost-cell divergence. "
        f"See {REPORT_PATH}."
    )
    if stage_b["status"] == "evaluated":
        assert stage_b["metrics"]["passed"], (
            "AMRVAC legacy comparison found uniform interpolation divergence. "
            f"See {REPORT_PATH}."
        )


def run_tests():
    test_heavy_amrvac_legacy_comparison()
    print("test_heavy_amrvac_legacy_comparison passed or skipped")


if __name__ == "__main__":
    run_tests()
