"""SLB-001 source-box scaling, reuse, current, and FND composition evidence."""

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
from simesh_rewrite._same_level_boxes import (
    fill_same_level_source_boxes_unchecked,
)
from simesh_rewrite.foundation import copy_region_into
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.same_level_boxes import fill_same_level_source_boxes
from simesh_rewrite.same_level_boxes_reference import (
    same_level_source_boxes_reference,
)
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes


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
STANDARD_INTERIOR_LOWER = np.asarray((12, 14, 16), dtype=np.int64)
STANDARD_INTERIOR_UPPER = np.asarray((20, 24, 28), dtype=np.int64)
STANDARD_LOWER_WIDTHS = np.asarray((1, 3, 5), dtype=np.int64)
STANDARD_UPPER_WIDTHS = np.asarray((6, 2, 4), dtype=np.int64)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def timing_statistics(samples: list[float]) -> dict:
    median = statistics.median(samples)
    deviations = [abs(value - median) for value in samples]
    return {
        "median_seconds": median,
        "median_absolute_deviation_seconds": statistics.median(deviations),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def interleaved_timings(operations, repeats: int) -> tuple[dict, list[list[str]]]:
    names = [name for name, _ in operations]
    samples = [[] for _ in operations]
    representative_orders: list[list[str]] = []
    for repeat in range(repeats):
        order = [
            (repeat + offset) % len(operations)
            for offset in range(len(operations))
        ]
        if repeat < len(operations):
            representative_orders.append([names[index] for index in order])
        for index in order:
            started = time.perf_counter()
            operations[index][1]()
            samples[index].append(time.perf_counter() - started)
    return (
        {
            names[index]: timing_statistics(samples[index])
            for index in range(len(operations))
        },
        representative_orders,
    )


def requested_box(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    lower_widths: np.ndarray,
    upper_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_shape = interior_upper - interior_lower
    if np.any(lower_widths < 0) or np.any(upper_widths < 0):
        raise ValueError("widths must be nonnegative")
    if np.any(lower_widths > block_shape) or np.any(upper_widths > block_shape):
        raise ValueError("widths must not exceed the interior shape")
    if np.any(lower_widths > interior_lower):
        raise ValueError("lower widths exceed nonnegative storage coordinates")
    return interior_lower - lower_widths, interior_upper + upper_widths


def target_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    lower_widths: np.ndarray,
    upper_widths: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    requested_lower, requested_upper = requested_box(
        interior_lower,
        interior_upper,
        lower_widths,
        upper_widths,
    )
    target_lower = np.empty(directions.shape, dtype=np.int64)
    target_upper = np.empty(directions.shape, dtype=np.int64)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
        target_lower,
        target_upper,
    )
    return requested_lower, requested_upper, target_lower, target_upper


def exact_pair(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> bool:
    return bool(
        np.array_equal(left_lower, right_lower)
        and np.array_equal(left_upper, right_upper)
    )


def total_box_cells(lower: np.ndarray, upper: np.ndarray) -> int:
    total = 0
    for row in range(lower.shape[0]):
        extent = upper[row] - lower[row]
        total += int(extent[0]) * int(extent[1]) * int(extent[2])
    return total


def benchmark_case(
    label: str,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    directions: np.ndarray,
    lower_widths: np.ndarray,
    upper_widths: np.ndarray,
    repeats: int,
    *,
    trace_checked: bool,
) -> tuple[dict, tuple[np.ndarray, ...]]:
    interior_lower = np.ascontiguousarray(interior_lower, dtype=np.int64)
    interior_upper = np.ascontiguousarray(interior_upper, dtype=np.int64)
    directions = np.ascontiguousarray(directions, dtype=np.int64)
    lower_widths = np.ascontiguousarray(lower_widths, dtype=np.int64)
    upper_widths = np.ascontiguousarray(upper_widths, dtype=np.int64)
    requested_lower, requested_upper, target_lower, target_upper = target_boxes(
        interior_lower,
        interior_upper,
        lower_widths,
        upper_widths,
        directions,
    )
    checked_lower = np.empty(directions.shape, dtype=np.int64)
    checked_upper = np.empty(directions.shape, dtype=np.int64)
    unchecked_lower = np.empty(directions.shape, dtype=np.int64)
    unchecked_upper = np.empty(directions.shape, dtype=np.int64)
    reference_lower = np.empty(directions.shape, dtype=np.int64)
    reference_upper = np.empty(directions.shape, dtype=np.int64)

    def checked() -> None:
        fill_same_level_source_boxes(
            interior_lower,
            interior_upper,
            directions,
            target_lower,
            target_upper,
            checked_lower,
            checked_upper,
        )

    def unchecked() -> None:
        fill_same_level_source_boxes_unchecked(
            interior_lower,
            interior_upper,
            directions,
            target_lower,
            target_upper,
            unchecked_lower,
            unchecked_upper,
        )

    def python_reference() -> None:
        lower, upper = same_level_source_boxes_reference(
            interior_lower,
            interior_upper,
            directions,
            target_lower,
            target_upper,
        )
        np.copyto(reference_lower, lower)
        np.copyto(reference_upper, upper)

    checked()
    unchecked()
    python_reference()
    exact_unchecked = exact_pair(
        checked_lower,
        checked_upper,
        unchecked_lower,
        unchecked_upper,
    )
    exact_reference = exact_pair(
        checked_lower,
        checked_upper,
        reference_lower,
        reference_upper,
    )
    if not exact_unchecked or not exact_reference:
        raise AssertionError("SLB complete output arrays disagree")

    timings, representative_orders = interleaved_timings(
        [
            ("checked", checked),
            ("unchecked", unchecked),
            ("python_reference_allocating", python_reference),
        ],
        repeats,
    )
    if not exact_pair(
        checked_lower,
        checked_upper,
        unchecked_lower,
        unchecked_upper,
    ) or not exact_pair(
        checked_lower,
        checked_upper,
        reference_lower,
        reference_upper,
    ):
        raise AssertionError("timed SLB complete output arrays disagree")

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

    row_count = int(directions.shape[0])
    output_bytes = checked_lower.nbytes + checked_upper.nbytes
    extents = target_upper - target_lower
    rows_per_second = {
        name: (
            0.0
            if row_count == 0
            else row_count / summary["median_seconds"]
        )
        for name, summary in timings.items()
    }
    result = {
        "label": label,
        "row_count": row_count,
        "interior_lower": interior_lower.tolist(),
        "interior_upper": interior_upper.tolist(),
        "interior_shape": (interior_upper - interior_lower).tolist(),
        "lower_widths": lower_widths.tolist(),
        "upper_widths": upper_widths.tolist(),
        "requested_lower": requested_lower.tolist(),
        "requested_upper": requested_upper.tolist(),
        "total_target_cells_per_field": total_box_cells(
            target_lower, target_upper
        ),
        "minimum_axis_extent": (
            None if row_count == 0 else int(np.min(extents))
        ),
        "maximum_axis_extent": (
            None if row_count == 0 else int(np.max(extents))
        ),
        "output_bytes": output_bytes,
        "expected_48R_bytes": 48 * row_count,
        "exact_checked_unchecked_complete_arrays": exact_unchecked,
        "exact_checked_reference_complete_arrays": exact_reference,
        "python_reference_allocates_return_arrays": True,
        "timings": timings,
        "representative_timing_orders": representative_orders,
        "rows_per_second": rows_per_second,
        "checked_over_unchecked": (
            timings["checked"]["median_seconds"]
            / timings["unchecked"]["median_seconds"]
        ),
        "checked_over_python_reference": (
            timings["checked"]["median_seconds"]
            / timings["python_reference_allocating"]["median_seconds"]
        ),
        "checked_traced_retained_bytes": traced_retained,
        "checked_traced_peak_bytes": traced_peak,
    }
    if output_bytes != 48 * row_count:
        raise AssertionError("SLB output byte formula is not exact")
    inputs = (
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
    )
    return result, inputs


def materialization_reuse(
    inputs: tuple[np.ndarray, ...],
    scan_count: int,
    repeats: int,
) -> dict:
    interior_lower, interior_upper, directions, target_lower, target_upper = inputs
    artifact_lower = np.empty(directions.shape, dtype=np.int64)
    artifact_upper = np.empty(directions.shape, dtype=np.int64)
    reuse_lower = np.empty_like(artifact_lower)
    reuse_upper = np.empty_like(artifact_upper)
    derived_lower = np.empty_like(artifact_lower)
    derived_upper = np.empty_like(artifact_upper)

    started = time.perf_counter()
    fill_same_level_source_boxes_unchecked(
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
        artifact_lower,
        artifact_upper,
    )
    creation_seconds = time.perf_counter() - started

    def reuse() -> None:
        for _ in range(scan_count):
            np.copyto(reuse_lower, artifact_lower)
            np.copyto(reuse_upper, artifact_upper)

    def derive() -> None:
        for _ in range(scan_count):
            fill_same_level_source_boxes_unchecked(
                interior_lower,
                interior_upper,
                directions,
                target_lower,
                target_upper,
                derived_lower,
                derived_upper,
            )

    timings, representative_orders = interleaved_timings(
        [
            ("materialized_native_copies", reuse),
            ("repeated_unchecked_derivation", derive),
        ],
        repeats,
    )
    exact = bool(
        exact_pair(
            reuse_lower,
            reuse_upper,
            artifact_lower,
            artifact_upper,
        )
        and exact_pair(
            reuse_lower,
            reuse_upper,
            derived_lower,
            derived_upper,
        )
    )
    if not exact:
        raise AssertionError("SLB reuse and repeated derivation arrays disagree")
    copy_seconds = timings["materialized_native_copies"]["median_seconds"]
    derive_seconds = timings["repeated_unchecked_derivation"]["median_seconds"]
    reuse_total = creation_seconds + copy_seconds
    artifact_bytes = artifact_lower.nbytes + artifact_upper.nbytes
    if artifact_bytes != 48 * directions.shape[0]:
        raise AssertionError("SLB reuse artifact byte formula is not exact")
    return {
        "description": (
            "same-shaped native array copies of one unchecked materialization "
            "versus repeated unchecked affine derivation"
        ),
        "scan_count": scan_count,
        "artifact_creation_seconds": creation_seconds,
        "artifact_bytes": artifact_bytes,
        "consumer_output_bytes": reuse_lower.nbytes + reuse_upper.nbytes,
        "timings": timings,
        "representative_timing_orders": representative_orders,
        "reuse_total_including_creation_seconds": reuse_total,
        "repeated_derivation_over_reuse_total": derive_seconds / reuse_total,
        "exact_complete_consumer_arrays": exact,
    }


def boxes_are_pairwise_disjoint(
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    for left in range(lower.shape[0]):
        for right in range(left):
            intersects = True
            for axis in range(3):
                if max(int(lower[left, axis]), int(lower[right, axis])) >= min(
                    int(upper[left, axis]), int(upper[right, axis])
                ):
                    intersects = False
                    break
            if intersects:
                return False
    return True


def fnd_copy_composition(repeats: int) -> dict:
    interior_lower = STANDARD_INTERIOR_LOWER.copy()
    interior_upper = STANDARD_INTERIOR_UPPER.copy()
    directions = ALL_DIRECTIONS.copy()
    requested_lower, requested_upper = requested_box(
        interior_lower,
        interior_upper,
        STANDARD_LOWER_WIDTHS,
        STANDARD_UPPER_WIDTHS,
    )
    target_lower = np.empty(directions.shape, dtype=np.int64)
    target_upper = np.empty(directions.shape, dtype=np.int64)
    source_lower = np.empty(directions.shape, dtype=np.int64)
    source_upper = np.empty(directions.shape, dtype=np.int64)

    def tgt_stage() -> None:
        fill_directed_halo_target_boxes(
            interior_lower,
            interior_upper,
            requested_lower,
            requested_upper,
            directions,
            target_lower,
            target_upper,
        )

    def slb_stage() -> None:
        fill_same_level_source_boxes(
            interior_lower,
            interior_upper,
            directions,
            target_lower,
            target_upper,
            source_lower,
            source_upper,
        )

    tgt_stage()
    slb_stage()
    expected_lower, expected_upper = same_level_source_boxes_reference(
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
    )
    if not exact_pair(
        source_lower,
        source_upper,
        expected_lower,
        expected_upper,
    ):
        raise AssertionError("TGT-to-SLB composition boxes disagree")
    if not boxes_are_pairwise_disjoint(target_lower, target_upper):
        raise AssertionError("canonical 26 target boxes are not disjoint")

    spatial_shape = tuple(int(value) for value in requested_upper)
    field_count = 2
    source = np.empty((1, field_count, *spatial_shape), dtype=np.float64)
    coordinates = np.indices(spatial_shape)
    for field in range(field_count):
        source[0, field] = (
            float(field) * 1_000_000.0
            + coordinates[0] * 10_000.0
            + coordinates[1] * 100.0
            + coordinates[2]
        )
    sentinel = np.float64(-1.0)
    destination = np.full_like(source, sentinel)
    expected = np.full_like(source, sentinel)
    source_before = source.copy()
    for row in range(directions.shape[0]):
        expected[
            :,
            :,
            int(target_lower[row, 0]) : int(target_upper[row, 0]),
            int(target_lower[row, 1]) : int(target_upper[row, 1]),
            int(target_lower[row, 2]) : int(target_upper[row, 2]),
        ] = source[
            :,
            :,
            int(source_lower[row, 0]) : int(source_upper[row, 0]),
            int(source_lower[row, 1]) : int(source_upper[row, 1]),
            int(source_lower[row, 2]) : int(source_upper[row, 2]),
        ]

    def copy_stage() -> None:
        for row in range(directions.shape[0]):
            copy_region_into(
                source,
                source_lower[row],
                destination,
                target_lower[row],
                target_upper[row] - target_lower[row],
            )

    copy_stage()
    exact = bool(
        np.array_equal(destination, expected)
        and np.array_equal(source, source_before)
    )
    interior_preserved = bool(
        np.all(
            destination[
                :,
                :,
                int(interior_lower[0]) : int(interior_upper[0]),
                int(interior_lower[1]) : int(interior_upper[1]),
                int(interior_lower[2]) : int(interior_upper[2]),
            ]
            == sentinel
        )
    )
    copied_cells = field_count * total_box_cells(target_lower, target_upper)
    changed_cells = int(np.count_nonzero(destination != sentinel))
    if not exact or not interior_preserved or changed_cells != copied_cells:
        raise AssertionError("one-slot FND copy composition disagrees")

    timings, representative_orders = interleaved_timings(
        [
            ("tgt_checked", tgt_stage),
            ("slb_checked", slb_stage),
            ("fnd_checked_copy_pass", copy_stage),
        ],
        repeats,
    )
    if not np.array_equal(destination, expected):
        raise AssertionError("timed one-slot FND composition changed output")

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    copy_stage()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    copy_seconds = timings["fnd_checked_copy_pass"]["median_seconds"]
    return {
        "description": (
            "TGT to SLB to 26 disjoint one-slot FND copies on a two-field "
            "axis-coded payload"
        ),
        "slot_count": 1,
        "field_count": field_count,
        "direction_count": int(directions.shape[0]),
        "pairwise_disjoint_target_boxes": True,
        "target_cells_per_field": total_box_cells(target_lower, target_upper),
        "copied_field_cells_per_pass": copied_cells,
        "changed_destination_cells": changed_cells,
        "copied_cells_per_second": copied_cells / copy_seconds,
        "effective_16byte_gb_per_second": (
            16 * copied_cells / copy_seconds / 1.0e9
        ),
        "timings": timings,
        "representative_timing_orders": representative_orders,
        "exact_complete_destination": exact,
        "source_preserved": bool(np.array_equal(source, source_before)),
        "interior_and_outside_targets_preserved": interior_preserved,
        "source_payload_bytes": source.nbytes,
        "destination_payload_bytes": destination.nbytes,
        "target_and_source_box_bytes": int(
            target_lower.nbytes
            + target_upper.nbytes
            + source_lower.nbytes
            + source_upper.nbytes
        ),
        "copy_traced_retained_bytes": after_current - before_current,
        "copy_traced_peak_bytes": peak - before_current,
    }


def current_unmasked_level1_comparison() -> dict:
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
    x, y, z = np.indices(tuple(int(value) for value in block_shape))
    for leaf_id in range(leaf_count):
        interior[leaf_id, 0] = (
            float(leaf_id) * 1_000_000.0 + x * 10_000.0 + y * 100.0 + z
        )
    mesh.load_interior_data(interior)
    started = time.perf_counter()
    mesh.apply_ghost_cells()
    current_seconds = time.perf_counter() - started

    interior_lower = i3(ghost_width, ghost_width, ghost_width)
    interior_upper = interior_lower + block_shape
    requested_lower = i3(0, 0, 0)
    requested_upper = block_shape + 2 * ghost_width
    target_lower = np.empty_like(ALL_DIRECTIONS)
    target_upper = np.empty_like(ALL_DIRECTIONS)
    source_lower = np.empty_like(ALL_DIRECTIONS)
    source_upper = np.empty_like(ALL_DIRECTIONS)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    fill_same_level_source_boxes(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
        source_lower,
        source_upper,
    )
    expected_lower, expected_upper = same_level_source_boxes_reference(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    exact_boxes = exact_pair(
        source_lower,
        source_upper,
        expected_lower,
        expected_upper,
    )
    coord_to_rank, _ = level1_morton(root)
    primary_coord = i3(1, 1, 1)
    primary_leaf = int(coord_to_rank[1, 1, 1])
    padded = np.asarray(mesh.padded_view())[primary_leaf, ..., 0]
    compared_cells = 0
    exact_values = True
    for row, direction in enumerate(ALL_DIRECTIONS):
        source_coord = primary_coord + direction
        source_leaf = int(coord_to_rank[tuple(source_coord)])
        actual = padded[
            int(target_lower[row, 0]) : int(target_upper[row, 0]),
            int(target_lower[row, 1]) : int(target_upper[row, 1]),
            int(target_lower[row, 2]) : int(target_upper[row, 2]),
        ]
        local_lower = source_lower[row] - interior_lower
        local_upper = source_upper[row] - interior_lower
        expected = interior[
            source_leaf,
            0,
            int(local_lower[0]) : int(local_upper[0]),
            int(local_lower[1]) : int(local_upper[1]),
            int(local_lower[2]) : int(local_upper[2]),
        ]
        exact_values &= bool(np.array_equal(actual, expected))
        compared_cells += int(actual.size)
    if not exact_boxes or not exact_values:
        raise AssertionError("current unmasked same-level source regions disagree")
    return {
        "description": (
            "canonical current 3x3x3 level-one AMRMesh central leaf; all rows "
            "are unmasked, positive-even extents with ghost width no larger "
            "than the interior; physical/mixed and degenerate parity excluded"
        ),
        "root_shape": root.tolist(),
        "block_shape": block_shape.tolist(),
        "ghost_width": ghost_width,
        "direction_count": 26,
        "current_apply_ghost_cells_seconds_descriptive": current_seconds,
        "compared_cells": compared_cells,
        "exact_reference_source_boxes": exact_boxes,
        "exact_current_axis_coded_source_slices": bool(exact_values),
        "slb_output_bytes": source_lower.nbytes + source_upper.nbytes,
        "current_padded_bytes": int(np.asarray(mesh.padded_view()).nbytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--reuse-scans", type=int, default=100_000)
    parser.add_argument("--reuse-repeats", type=int, default=5)
    parser.add_argument("--composition-repeats", type=int, default=9)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.reuse_scans < 1
        or args.reuse_repeats < 1
        or args.composition_repeats < 1
    ):
        raise ValueError("repeat and scan counts must be positive")

    standard, standard_inputs = benchmark_case(
        "canonical_26_partial_asymmetric",
        STANDARD_INTERIOR_LOWER,
        STANDARD_INTERIOR_UPPER,
        ALL_DIRECTIONS,
        STANDARD_LOWER_WIDTHS,
        STANDARD_UPPER_WIDTHS,
        args.repeats,
        trace_checked=True,
    )
    row_scaling = [
        benchmark_case(
            label,
            STANDARD_INTERIOR_LOWER,
            STANDARD_INTERIOR_UPPER,
            directions,
            STANDARD_LOWER_WIDTHS,
            STANDARD_UPPER_WIDTHS,
            args.repeats,
            trace_checked=True,
        )[0]
        for label, directions in (
            ("R0", np.empty((0, 3), dtype=np.int64)),
            ("R1", np.ascontiguousarray(ALL_DIRECTIONS[[0]])),
            ("R6_faces", FACE_DIRECTIONS),
            (
                "R52_repeated_canonical",
                np.ascontiguousarray(
                    np.concatenate((ALL_DIRECTIONS, ALL_DIRECTIONS), axis=0)
                ),
            ),
        )
    ]
    block_shape = STANDARD_INTERIOR_UPPER - STANDARD_INTERIOR_LOWER
    width_direction_cases = [
        benchmark_case(
            label,
            STANDARD_INTERIOR_LOWER,
            STANDARD_INTERIOR_UPPER,
            directions,
            lower_widths,
            upper_widths,
            args.repeats,
            trace_checked=False,
        )[0]
        for label, directions, lower_widths, upper_widths in (
            (
                "zero_width_all_axes_R26",
                ALL_DIRECTIONS,
                i3(0, 0, 0),
                i3(0, 0, 0),
            ),
            (
                "unit_width_faces_R6",
                FACE_DIRECTIONS,
                i3(1, 1, 1),
                i3(1, 1, 1),
            ),
            (
                "full_width_all_axes_R26",
                ALL_DIRECTIONS,
                block_shape,
                block_shape,
            ),
            (
                "reordered_partial_R26",
                np.ascontiguousarray(ALL_DIRECTIONS[::-1]),
                STANDARD_LOWER_WIDTHS,
                STANDARD_UPPER_WIDTHS,
            ),
        )
    ]
    degenerate_reference_cases = [
        benchmark_case(
            "singleton_axes_reference",
            i3(3, 4, 5),
            i3(4, 5, 6),
            ALL_DIRECTIONS,
            i3(1, 1, 1),
            i3(1, 1, 1),
            args.repeats,
            trace_checked=False,
        )[0],
        benchmark_case(
            "zero_xz_axes_reference",
            i3(3, 4, 5),
            i3(3, 9, 5),
            ALL_DIRECTIONS,
            i3(0, 2, 0),
            i3(0, 3, 0),
            args.repeats,
            trace_checked=False,
        )[0],
    ]
    traced_peaks = [standard["checked_traced_peak_bytes"]]
    traced_peaks.extend(
        case["checked_traced_peak_bytes"] for case in row_scaling
    )

    report = {
        "capability": "SLB-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "measurement_policy": {
            "correctness_gate": "complete exact lower and upper arrays",
            "timing": "rotating interleaved operations; median and MAD reported",
            "material_regression": (
                "investigate controlled median regression above 10 percent; "
                "reproduced representative regression above 20 percent blocks"
            ),
        },
        "repeats": args.repeats,
        "standard": standard,
        "row_scaling": row_scaling,
        "width_and_direction_cases": width_direction_cases,
        "degenerate_reference_only_cases": degenerate_reference_cases,
        "checked_tracemalloc_fixed_behavior": {
            "row_counts": [26] + [case["row_count"] for case in row_scaling],
            "retained_bytes": [standard["checked_traced_retained_bytes"]]
            + [case["checked_traced_retained_bytes"] for case in row_scaling],
            "peak_bytes": traced_peaks,
            "peak_range_bytes": max(traced_peaks) - min(traced_peaks),
            "output_artifact_bytes_follow_exact_48R_not_tracemalloc": True,
        },
        "materialize_once_reuse": materialization_reuse(
            standard_inputs,
            args.reuse_scans,
            args.reuse_repeats,
        ),
        "one_slot_fnd_copy_composition": fnd_copy_composition(
            args.composition_repeats
        ),
        "current_canonical_unmasked_same_level": (
            current_unmasked_level1_comparison()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
