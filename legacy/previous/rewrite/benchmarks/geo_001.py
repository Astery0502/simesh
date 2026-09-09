"""Retained selected-block runtime and memory probe for GEO-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._geometry import (
    fill_level1_block_geometry_unchecked,
    validate_selected_geometry_unchecked,
)
from simesh_rewrite.geometry import fill_level1_block_geometry
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


def median_call(function, arguments, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*arguments)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def ordered_float_bits(value: float) -> int:
    bits = int(np.asarray(value, dtype=np.float64).view(np.uint64))
    sign = 1 << 63
    mask = (1 << 64) - 1
    return ((~bits) & mask) if bits & sign else bits | sign


def current_comparison() -> dict:
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh

    root_shape = np.array([3, 2, 5], dtype=np.int64)
    block_cells = np.array([4, 6, 8], dtype=np.int64)
    domain_cells = root_shape * block_cells
    domain_lower = np.array([-0.7, 1.1, -2.3])
    domain_upper = np.array([1.9, 4.7, 7.2])
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    bounds = np.empty((block_ids.size, 2, 3), dtype=np.float64)
    spacing = np.empty(3, dtype=np.float64)
    fill_level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        block_ids,
        bounds,
        spacing,
    )
    forest = AMRForest(3, *tuple(root_shape), np.ones(block_ids.size, dtype=np.int32))
    mesh = AMRMesh(
        3,
        block_cells.astype(np.uint32),
        domain_cells.astype(np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(0),
        np.uint32(1),
        forest,
    )
    current = np.asarray(mesh.rnode)
    current_bounds = current[:, :6].reshape(-1, 2, 3)
    absolute = np.abs(bounds - current_bounds)
    max_ulp = max(
        abs(ordered_float_bits(actual) - ordered_float_bits(expected))
        for actual, expected in zip(bounds.ravel(), current_bounds.ravel(), strict=True)
    )

    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    current_face_gaps = []
    rewrite_bit_mismatches = 0
    for block_id in range(block_ids.size):
        for axis in range(3):
            neighbor = neighbors[block_id, 2 * axis + 1]
            if neighbor >= 0:
                current_face_gaps.append(
                    abs(
                        current_bounds[block_id, 1, axis]
                        - current_bounds[neighbor, 0, axis]
                    )
                )
                rewrite_bit_mismatches += int(
                    bounds[block_id, 1, axis].tobytes()
                    != bounds[neighbor, 0, axis].tobytes()
                )

    current_endpoint_errors = []
    for block_id, coordinate in enumerate(rank_to_coord):
        for axis in range(3):
            if coordinate[axis] == 0:
                current_endpoint_errors.append(
                    abs(current_bounds[block_id, 0, axis] - domain_lower[axis])
                )
            if coordinate[axis] + 1 == root_shape[axis]:
                current_endpoint_errors.append(
                    abs(current_bounds[block_id, 1, axis] - domain_upper[axis])
                )

    return {
        "bounds_max_abs_by_axis": np.max(absolute, axis=(0, 1)).tolist(),
        "bounds_max_spacing_fraction": float(np.max(absolute / spacing)),
        "bounds_max_ulp": max_ulp,
        "spacing_max_abs": float(np.max(np.abs(current[:, 6:9] - spacing))),
        "current_shared_face_max_abs": float(max(current_face_gaps, default=0.0)),
        "rewrite_shared_face_bit_mismatches": rewrite_bit_mismatches,
        "current_endpoint_max_abs": float(max(current_endpoint_errors, default=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", default="1,8,64,512,4096,65536,all")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    root_shape = np.array([127, 113, 97], dtype=np.int64)
    block_cells = np.array([8, 6, 10], dtype=np.int64)
    domain_cells = root_shape * block_cells
    domain_lower = np.array([-0.7, 1.1, -2.3])
    domain_upper = np.array([1.9, 4.7, 7.2])
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    volume = rank_to_coord.shape[0]
    spacing_values = (domain_upper - domain_lower) / domain_cells

    results = []
    for value in args.slots.split(","):
        slot_count = volume if value == "all" else int(value)
        block_ids = np.arange(slot_count, dtype=np.int64)
        if slot_count > 1:
            block_ids = np.ascontiguousarray((block_ids * 104729) % volume)
        bounds = np.empty((slot_count, 2, 3), dtype=np.float64)
        spacing = np.empty(3, dtype=np.float64)
        checked_arguments = (
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            block_ids,
            bounds,
            spacing,
        )
        fill_level1_block_geometry(*checked_arguments)
        checked_seconds = median_call(
            fill_level1_block_geometry,
            checked_arguments,
            args.repeats,
        )
        validation_seconds = median_call(
            validate_selected_geometry_unchecked,
            (
                domain_lower,
                domain_upper,
                domain_cells,
                block_cells,
                coord_to_rank,
                rank_to_coord,
                block_ids,
                spacing_values,
            ),
            args.repeats,
        )
        fill_seconds = median_call(
            fill_level1_block_geometry_unchecked,
            (
                domain_lower,
                domain_upper,
                domain_cells,
                block_cells,
                rank_to_coord,
                block_ids,
                spacing_values,
                bounds,
                spacing,
            ),
            args.repeats,
        )

        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        fill_level1_block_geometry(*checked_arguments)
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append(
            {
                "slots": slot_count,
                "checked_seconds": checked_seconds,
                "validation_seconds": validation_seconds,
                "fill_seconds": fill_seconds,
                "million_blocks_per_second": slot_count / checked_seconds / 1.0e6,
                "bounds_bytes": bounds.nbytes,
                "full_bounds_bytes": volume * 2 * 3 * 8,
                "repeated_spacing_bytes_avoided": slot_count * 3 * 8,
                "traced_current_delta_bytes": after_current - before_current,
                "traced_peak_delta_bytes": peak - before_current,
            }
        )

    report = {
        "capability": "GEO-001",
        "root_shape": root_shape.tolist(),
        "blocks": volume,
        "repeats": args.repeats,
        "results": results,
        "current_comparison": current_comparison(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
