"""Kernel and bounded composed-path measurements for OPR-002."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.operators import central_difference_into
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def neighbor_views(
    source: np.ndarray,
    field: int,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower_region = [
        slice(int(lower), int(upper))
        for lower, upper in zip(output_lower, output_upper, strict=True)
    ]
    upper_region = list(lower_region)
    lower_region[axis] = slice(
        int(output_lower[axis] - 1),
        int(output_upper[axis] - 1),
    )
    upper_region[axis] = slice(
        int(output_lower[axis] + 1),
        int(output_upper[axis] + 1),
    )
    return (
        source[(slice(None), field, *lower_region)],
        source[(slice(None), field, *upper_region)],
    )


def finite_error_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    local_scale: np.ndarray,
) -> dict:
    finite = np.isfinite(actual) & np.isfinite(reference)
    difference = np.abs(actual - reference)
    max_absolute = float(np.max(difference[finite], initial=0.0))
    relative_mask = finite & (
        np.abs(reference) >= np.sqrt(np.finfo(np.float64).eps) * local_scale
    )
    max_relative = float(
        np.max(
            difference[relative_mask] / np.abs(reference[relative_mask]),
            initial=0.0,
        )
    )
    nonzero = finite & (actual != 0.0) & (reference != 0.0)
    if np.any(nonzero):
        actual_bits = actual[nonzero].view(np.uint64)
        reference_bits = reference[nonzero].view(np.uint64)
        sign_mask = np.uint64(1 << 63)
        actual_ordered = np.where(
            actual_bits & sign_mask,
            ~actual_bits,
            actual_bits | sign_mask,
        )
        reference_ordered = np.where(
            reference_bits & sign_mask,
            ~reference_bits,
            reference_bits | sign_mask,
        )
        ulp = np.where(
            actual_ordered >= reference_ordered,
            actual_ordered - reference_ordered,
            reference_ordered - actual_ordered,
        )
        max_ulp = int(np.max(ulp, initial=np.uint64(0)))
    else:
        max_ulp = 0
    return {
        "max_absolute_error": max_absolute,
        "max_relative_error_away_from_cancellation": max_relative,
        "max_finite_nonzero_ulp_error": max_ulp,
    }


def kernel_case(axis: int, slot_count: int, repeats: int) -> dict:
    rng = np.random.default_rng(1701 + axis)
    source = rng.normal(size=(slot_count, 2, 10, 10, 10))
    destination = np.full((slot_count, 1, 8, 8, 8), -7.0)
    expected = np.empty_like(destination)
    difference = np.empty((slot_count, 8, 8, 8), dtype=np.float64)
    spacing = np.array([0.125, 0.25, 0.5])
    output_lower = i3(1, 1, 1)
    output_upper = i3(9, 9, 9)
    zero = i3(0, 0, 0)

    def compiled() -> None:
        central_difference_into(
            source,
            zero,
            i3(10, 10, 10),
            output_lower,
            output_upper,
            1,
            axis,
            spacing,
            destination,
            0,
            zero,
        )

    def numpy_reference() -> None:
        lower_values, upper_values = neighbor_views(
            source,
            1,
            output_lower,
            output_upper,
            axis,
        )
        np.subtract(
            upper_values,
            lower_values,
            out=difference,
        )
        np.multiply(
            difference,
            np.float64(0.5) / spacing[axis],
            out=expected[:, 0],
        )

    compiled()
    numpy_reference()
    seconds = median_seconds(compiled, repeats)
    numpy_seconds = median_seconds(numpy_reference, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    compiled()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = slot_count * 8**3
    lower_values, upper_values = neighbor_views(
        source,
        1,
        output_lower,
        output_upper,
        axis,
    )
    local_scale = np.maximum.reduce(
        (
            np.ones_like(expected[:, 0]),
            np.abs(lower_values),
            np.abs(upper_values),
        )
    ) / spacing[axis]
    result = {
        "axis": axis,
        "slot_count": slot_count,
        "seconds": seconds,
        "numpy_seconds": numpy_seconds,
        "million_values_per_second": values / seconds / 1e6,
        "effective_gb_per_second": 24 * values / seconds / 1e9,
        "bitwise_equal": bool(
            np.array_equal(destination.view(np.uint64), expected.view(np.uint64))
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }
    result.update(finite_error_metrics(destination[:, 0], expected[:, 0], local_scale))
    return result


def current_comparison(axis: int, repeats: int) -> dict:
    root_shape = (2, 2, 2)
    block = np.array([8, 8, 8], dtype=np.uint32)
    domain = np.array([16, 16, 16], dtype=np.uint32)
    spacing = np.array([1.0 / 16.0] * 3)
    forest = AMRForest(3, *root_shape, np.ones(8, dtype=np.int32))
    mesh = AMRMesh(
        3,
        block,
        domain,
        np.zeros(3),
        np.ones(3),
        np.uint32(2),
        np.uint32(1),
        forest,
    )
    backing = np.empty((8, 1, 8, 8, 8), dtype=np.float64)
    for block_id, node in enumerate(np.asarray(mesh.rnode)):
        x = node[0] + (np.arange(8)[:, None, None] + 0.5) * node[6]
        y = node[1] + (np.arange(8)[None, :, None] + 0.5) * node[7]
        z = node[2] + (np.arange(8)[None, None, :] + 0.5) * node[8]
        backing[block_id, 0] = x + 2.0 * y + 5.0 * z
    mesh.load_interior_data(backing)
    mesh.apply_ghost_cells()
    padded = np.ascontiguousarray(
        np.transpose(mesh.padded_view(), (0, 4, 1, 2, 3))
    )
    rewrite_output = np.empty((8, 1, 10, 10, 10), dtype=np.float64)
    current_output = np.empty((8, 12, 12, 12, 1), dtype=np.float64)
    terms = np.array([0], dtype=np.uint32)
    axes = np.array([axis], dtype=np.uint32)
    coefficients = np.array([1.0])

    def rewrite() -> None:
        central_difference_into(
            padded,
            i3(0, 0, 0),
            i3(12, 12, 12),
            i3(1, 1, 1),
            i3(11, 11, 11),
            0,
            axis,
            spacing,
            rewrite_output,
            0,
            i3(0, 0, 0),
        )

    def current() -> None:
        mesh.first_derivative_fields(
            current_output,
            terms,
            terms,
            axes,
            coefficients,
        )

    rewrite()
    current()
    current_common = np.transpose(current_output[:, 1:11, 1:11, 1:11], (0, 4, 1, 2, 3))
    lower_values, upper_values = neighbor_views(
        padded,
        0,
        i3(1, 1, 1),
        i3(11, 11, 11),
        axis,
    )
    local_scale = np.maximum.reduce(
        (
            np.ones_like(current_common[:, 0]),
            np.abs(lower_values),
            np.abs(upper_values),
        )
    ) / spacing[axis]
    report = {
        "axis": axis,
        "rewrite_seconds": median_seconds(rewrite, repeats),
        "current_seconds": median_seconds(current, repeats),
        "bitwise_equal": bool(
            np.array_equal(
                rewrite_output.view(np.uint64),
                current_common.view(np.uint64),
            )
        ),
    }
    report.update(
        finite_error_metrics(rewrite_output[:, 0], current_common[:, 0], local_scale)
    )
    return report


def bounded_composition(axis: int, capacity: int, repeats: int) -> dict:
    root = i3(8, 6, 4)
    block = i3(8, 8, 8)
    domain = root * block
    spacing = 1.0 / domain.astype(np.float64)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    block_count = faces.shape[0]
    backing = np.empty((block_count, 1, 8, 8, 8), dtype=np.float64)
    local = np.indices((8, 8, 8), dtype=np.float64)
    for block_id, coordinate in enumerate(rank_to_coord):
        global_index = coordinate[:, None, None, None] * block[:, None, None, None] + local
        centers = (global_index + 0.5) * spacing[:, None, None, None]
        backing[block_id, 0] = centers[0] + 2.0 * centers[1] + 5.0 * centers[2]

    ids = np.empty(capacity, dtype=np.int64)
    source = np.empty((capacity, 1, 10, 10, 10), dtype=np.float64)
    destination = np.empty((capacity, 1, 8, 8, 8), dtype=np.float64)
    result = np.empty((block_count, 1, 8, 8, 8), dtype=np.float64)
    field = np.array([0], dtype=np.int64)
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    zero = i3(0, 0, 0)
    lower = i3(1, 1, 1)
    upper = i3(9, 9, 9)

    all_ids = np.arange(block_count, dtype=np.int64)
    full_source = np.empty((block_count, 1, 10, 10, 10), dtype=np.float64)
    full_expected = np.empty((block_count, 1, 8, 8, 8), dtype=np.float64)
    gather_blocks_into(backing, zero, block, all_ids, field, full_source, lower)
    fill_physical_halos(
        full_source,
        lower,
        upper,
        all_ids,
        faces,
        modes,
        normals,
    )
    fill_same_level_halos(
        full_source,
        lower,
        upper,
        all_ids,
        block_count,
        faces,
        modes,
        normals,
    )
    central_difference_into(
        full_source,
        zero,
        i3(10, 10, 10),
        lower,
        upper,
        0,
        axis,
        spacing,
        full_expected,
        0,
        zero,
    )

    def traverse() -> int:
        first = 0
        chunks = 0
        while first < block_count:
            primary_count, selected_count = plan_level1_halo_chunk(
                first,
                faces,
                ids,
            )
            gather_blocks_into(
                backing,
                zero,
                block,
                ids[:selected_count],
                field,
                source[:selected_count],
                lower,
            )
            fill_physical_halos(
                source[:selected_count],
                lower,
                upper,
                ids[:selected_count],
                faces,
                modes,
                normals,
            )
            fill_same_level_halos(
                source[:selected_count],
                lower,
                upper,
                ids[:selected_count],
                primary_count,
                faces,
                modes,
                normals,
            )
            central_difference_into(
                source[:primary_count],
                zero,
                i3(10, 10, 10),
                lower,
                upper,
                0,
                axis,
                spacing,
                destination[:primary_count],
                0,
                zero,
            )
            scatter_blocks_from(
                destination[:primary_count],
                zero,
                block,
                ids[:primary_count],
                field,
                result,
                zero,
            )
            first += primary_count
            chunks += 1
        return chunks

    chunks = traverse()
    seconds = median_seconds(traverse, repeats)
    expected_value = (1.0, 2.0, 5.0)[axis]
    analytic = np.full_like(result[:, 0], expected_value)
    axis_slice = [slice(None), slice(None), slice(None), slice(None)]
    for block_id, coordinate in enumerate(rank_to_coord):
        if coordinate[axis] == 0:
            axis_slice[0] = block_id
            axis_slice[axis + 1] = 0
            analytic[tuple(axis_slice)] = 0.5 * expected_value
            axis_slice[axis + 1] = slice(None)
        if coordinate[axis] + 1 == root[axis]:
            axis_slice[0] = block_id
            axis_slice[axis + 1] = -1
            analytic[tuple(axis_slice)] = 0.5 * expected_value
            axis_slice[axis + 1] = slice(None)
    analytic_error = np.abs(result[:, 0] - analytic)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    traverse()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    managed_bytes = ids.nbytes + source.nbytes + destination.nbytes
    lower_values, upper_values = neighbor_views(
        full_source,
        0,
        lower,
        upper,
        axis,
    )
    local_scale = np.maximum.reduce(
        (
            np.ones_like(full_expected[:, 0]),
            np.abs(lower_values),
            np.abs(upper_values),
        )
    ) / spacing[axis]
    report = {
        "axis": axis,
        "capacity": capacity,
        "chunks": chunks,
        "managed_workspace_bytes": managed_bytes,
        "seconds": seconds,
        "million_values_per_second": result[:, 0].size / seconds / 1e6,
        "max_absolute_boundary_aware_affine_error": float(np.max(analytic_error)),
        "bitwise_equal_to_full": bool(
            np.array_equal(
                result[:, 0].view(np.uint64),
                full_expected[:, 0].view(np.uint64),
            )
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }
    report.update(
        finite_error_metrics(result[:, 0], full_expected[:, 0], local_scale)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "capability": "OPR-002",
                "kernel_cases": [
                    kernel_case(axis, 128, args.repeats) for axis in range(3)
                ],
                "current": [
                    current_comparison(axis, args.repeats) for axis in range(3)
                ],
                "bounded_composition": [
                    bounded_composition(axis, capacity, args.repeats)
                    for axis in range(3)
                    for capacity in (64, 256)
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
