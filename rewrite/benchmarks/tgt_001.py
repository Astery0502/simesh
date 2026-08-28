"""TGT-001 target-box scaling, reuse, and safe current-table evidence."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite._target_boxes import (
    fill_directed_halo_target_boxes_unchecked,
)
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes
from simesh_rewrite.target_boxes_reference import (
    directed_halo_target_boxes_reference,
)
from simesh_rewrite.morton import level1_morton


ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)
FACE_DIRECTIONS = np.asarray(
    [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ],
    dtype=np.int64,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def interleaved_timings(operations, repeats: int):
    samples = {name: [] for name, _ in operations}
    orders = []
    for repeat in range(repeats):
        order = [
            operations[(repeat + offset) % len(operations)]
            for offset in range(len(operations))
        ]
        orders.append([name for name, _ in order])
        for name, operation in order:
            started = time.perf_counter()
            operation()
            samples[name].append(time.perf_counter() - started)
    return (
        {name: statistics.median(values) for name, values in samples.items()},
        orders,
    )


def exact_pair(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> bool:
    return bool(
        np.array_equal(left[0], right[0])
        and np.array_equal(left[1], right[1])
    )


def case(
    label: str,
    directions: np.ndarray,
    repeats: int,
    *,
    trace_checked: bool = False,
) -> tuple[dict, tuple]:
    interior_lower = i3(4, 6, 6)
    interior_upper = i3(9, 8, 10)
    requested_lower = i3(2, 3, 5)
    requested_upper = i3(12, 13, 12)
    checked_lower = np.empty(directions.shape, dtype=np.int64)
    checked_upper = np.empty(directions.shape, dtype=np.int64)
    unchecked_lower = np.empty(directions.shape, dtype=np.int64)
    unchecked_upper = np.empty(directions.shape, dtype=np.int64)
    reference_lower = np.empty(directions.shape, dtype=np.int64)
    reference_upper = np.empty(directions.shape, dtype=np.int64)

    def checked() -> None:
        fill_directed_halo_target_boxes(
            interior_lower,
            interior_upper,
            requested_lower,
            requested_upper,
            directions,
            checked_lower,
            checked_upper,
        )

    def unchecked() -> None:
        fill_directed_halo_target_boxes_unchecked(
            interior_lower,
            interior_upper,
            requested_lower,
            requested_upper,
            directions,
            unchecked_lower,
            unchecked_upper,
        )

    def python_reference() -> None:
        lower, upper = directed_halo_target_boxes_reference(
            interior_lower,
            interior_upper,
            requested_lower,
            requested_upper,
            directions,
        )
        np.copyto(reference_lower, lower)
        np.copyto(reference_upper, upper)

    checked()
    unchecked()
    python_reference()
    checked_pair = (checked_lower, checked_upper)
    exact_unchecked = exact_pair(
        checked_pair, (unchecked_lower, unchecked_upper)
    )
    exact_reference = exact_pair(
        checked_pair, (reference_lower, reference_upper)
    )
    if not exact_unchecked or not exact_reference:
        raise AssertionError("TGT complete output arrays disagree")

    operations = [
        ("checked", checked),
        ("unchecked", unchecked),
        ("python_reference_allocating", python_reference),
    ]
    timings, timing_orders = interleaved_timings(operations, repeats)
    if not exact_pair(checked_pair, (unchecked_lower, unchecked_upper)) or not exact_pair(
        checked_pair, (reference_lower, reference_upper)
    ):
        raise AssertionError("timed TGT complete arrays disagree")

    traced_retained = None
    traced_peak = None
    if trace_checked:
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        checked()
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_retained = after_current - before_current
        traced_peak = peak - before_current

    direction_count = int(directions.shape[0])
    output_bytes = checked_lower.nbytes + checked_upper.nbytes
    result = {
        "label": label,
        "direction_count": direction_count,
        "directions": directions.tolist(),
        "output_bytes": output_bytes,
        "expected_48D_bytes": 48 * direction_count,
        "exact_checked_unchecked_arrays": exact_unchecked,
        "exact_checked_python_reference_arrays": exact_reference,
        "python_reference_allocates_return_arrays": True,
        "timing_orders": timing_orders,
        "seconds": timings,
        "million_directions_per_second": {
            name: (
                0.0 if direction_count == 0 else direction_count / seconds / 1.0e6
            )
            for name, seconds in timings.items()
        },
        "checked_over_unchecked": timings["checked"] / timings["unchecked"],
        "checked_over_python_reference": (
            timings["checked"] / timings["python_reference_allocating"]
        ),
        "checked_traced_retained_bytes": traced_retained,
        "checked_traced_peak_bytes": traced_peak,
    }
    if output_bytes != 48 * direction_count:
        raise AssertionError("TGT output byte formula is not exact")
    inputs = (
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
    )
    return result, inputs


def materialization_reuse(
    inputs: tuple,
    scan_count: int,
    repeats: int,
) -> dict:
    interior_lower, interior_upper, requested_lower, requested_upper, directions = inputs
    artifact_lower = np.empty(directions.shape, dtype=np.int64)
    artifact_upper = np.empty(directions.shape, dtype=np.int64)
    action_lower = np.empty_like(artifact_lower)
    action_upper = np.empty_like(artifact_upper)
    derived_lower = np.empty_like(artifact_lower)
    derived_upper = np.empty_like(artifact_upper)

    started = time.perf_counter()
    fill_directed_halo_target_boxes_unchecked(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
        artifact_lower,
        artifact_upper,
    )
    creation_seconds = time.perf_counter() - started

    def reuse_scan() -> None:
        for _ in range(scan_count):
            np.copyto(action_lower, artifact_lower)
            np.copyto(action_upper, artifact_upper)

    def repeated_derivation() -> None:
        for _ in range(scan_count):
            fill_directed_halo_target_boxes_unchecked(
                interior_lower,
                interior_upper,
                requested_lower,
                requested_upper,
                directions,
                derived_lower,
                derived_upper,
            )

    timings, timing_orders = interleaved_timings(
        [
            ("reuse_np_copyto", reuse_scan),
            ("repeated_unchecked_derivation", repeated_derivation),
        ],
        repeats,
    )
    exact = exact_pair(
        (action_lower, action_upper), (derived_lower, derived_upper)
    ) and exact_pair(
        (action_lower, action_upper), (artifact_lower, artifact_upper)
    )
    if not exact:
        raise AssertionError("TGT reuse and repeated derivation arrays disagree")
    reuse_total = creation_seconds + timings["reuse_np_copyto"]
    return {
        "scan_count": scan_count,
        "artifact_creation_seconds": creation_seconds,
        "artifact_bytes": artifact_lower.nbytes + artifact_upper.nbytes,
        "reuse_copy_seconds": timings["reuse_np_copyto"],
        "reuse_total_including_creation_seconds": reuse_total,
        "repeated_unchecked_derivation_seconds": timings[
            "repeated_unchecked_derivation"
        ],
        "repeated_derivation_over_reuse_total": (
            timings["repeated_unchecked_derivation"] / reuse_total
        ),
        "timing_orders": timing_orders,
        "identical_action_output_shape": list(action_lower.shape),
        "exact_complete_action_arrays": exact,
        "action_output_bytes": action_lower.nbytes + action_upper.nbytes,
    }


def current_level1_behavioral_comparison() -> dict:
    root = i3(3, 3, 3)
    block_shape = i3(4, 6, 8)
    ghost_width = 2
    leaf_count = int(np.prod(root))
    forest = AMRForest(
        3,
        *tuple(int(value) for value in root),
        np.ones(leaf_count, dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block_shape.astype(np.uint32),
        (root * block_shape).astype(np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(ghost_width),
        np.uint32(1),
        forest,
    )
    interior = np.empty((leaf_count, 1, *block_shape), dtype=np.float64)
    for leaf_id in range(leaf_count):
        interior[leaf_id].fill(np.float64(leaf_id) + np.float64(0.25))
    mesh.load_interior_data(interior)
    started = time.perf_counter()
    mesh.apply_ghost_cells()
    current_seconds = time.perf_counter() - started

    interior_lower = i3(ghost_width, ghost_width, ghost_width)
    interior_upper = interior_lower + block_shape
    requested_lower = i3(0, 0, 0)
    requested_upper = block_shape + 2 * ghost_width
    lower = np.empty_like(ALL_DIRECTIONS)
    upper = np.empty_like(ALL_DIRECTIONS)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
        lower,
        upper,
    )
    coord_to_rank, _ = level1_morton(root)
    center_coord = i3(1, 1, 1)
    center_leaf = int(coord_to_rank[1, 1, 1])
    padded = np.asarray(mesh.padded_view())[center_leaf, ..., 0]
    exact = True
    compared_cells = 0
    for row, direction in enumerate(ALL_DIRECTIONS):
        source_coord = center_coord + direction
        source_leaf = int(coord_to_rank[tuple(source_coord)])
        expected = np.float64(source_leaf) + np.float64(0.25)
        target = padded[
            int(lower[row, 0]) : int(upper[row, 0]),
            int(lower[row, 1]) : int(upper[row, 1]),
            int(lower[row, 2]) : int(upper[row, 2]),
        ]
        exact &= target.size > 0 and bool(np.all(target == expected))
        compared_cells += int(target.size)
    if not exact:
        raise AssertionError("current level-one halo values differ from TGT boxes")
    return {
        "description": (
            "canonical current AMRMesh level-one ghost behavior on the fully "
            "interior leaf; unsafe singleton physical-status access is excluded"
        ),
        "root_shape": root.tolist(),
        "block_shape": block_shape.tolist(),
        "ghost_width": ghost_width,
        "direction_count": 26,
        "current_apply_ghost_cells_seconds": current_seconds,
        "compared_target_cells": compared_cells,
        "exact_neighbor_constants_in_all_boxes": bool(exact),
        "tgt_output_bytes": lower.nbytes + upper.nbytes,
        "current_padded_bytes": int(np.asarray(mesh.padded_view()).nbytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--reuse-scans", type=int, default=100_000)
    parser.add_argument("--reuse-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.reuse_scans < 1 or args.reuse_repeats < 1:
        raise ValueError("repeat and scan counts must be positive")

    standard, standard_inputs = case(
        "canonical_26",
        ALL_DIRECTIONS,
        args.repeats,
        trace_checked=True,
    )
    direction_scaling = [
        case("D0", np.empty((0, 3), dtype=np.int64), args.repeats)[0],
        case("D1", ALL_DIRECTIONS[[0]].copy(), args.repeats)[0],
        case("D6_faces", FACE_DIRECTIONS.copy(), args.repeats)[0],
        standard,
    ]
    reordered = np.ascontiguousarray(ALL_DIRECTIONS[::-1])
    repeated_subset = np.ascontiguousarray(
        ALL_DIRECTIONS[[25, 0, 12, 25, 5, 0, 19, 19, 7, 3]]
    )
    repeated_canonical = np.ascontiguousarray(
        np.concatenate((ALL_DIRECTIONS, ALL_DIRECTIONS), axis=0)
    )
    repeat_reorder_cases = [
        case("reordered_26", reordered, args.repeats)[0],
        case("repeated_subset_10", repeated_subset, args.repeats)[0],
        case("repeated_canonical_52", repeated_canonical, args.repeats)[0],
    ]

    report = {
        "capability": "TGT-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats": args.repeats,
        "standard_asymmetric_canonical26": standard,
        "direction_scaling": direction_scaling,
        "repeat_reorder_cases": repeat_reorder_cases,
        "materialize_once_reuse": materialization_reuse(
            standard_inputs,
            args.reuse_scans,
            args.reuse_repeats,
        ),
        "current_level1_behavioral": current_level1_behavioral_comparison(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
