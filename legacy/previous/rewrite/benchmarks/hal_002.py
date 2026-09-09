"""Kernel and bounded composed-path measurements for HAL-002."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.chunking import (
    plan_level1_halo_chunk,
    workspace_nbytes,
)
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.halos_reference import fill_same_level_halos_reference
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.storage import gather_blocks_into
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


def written_cell_count(
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> int:
    count = 0
    for primary in range(primary_count):
        block_id = int(block_ids[primary])
        for target in np.ndindex(spatial_shape):
            has_sibling = False
            for axis in range(3):
                if target[axis] < interior_lower[axis]:
                    face = 2 * axis
                elif target[axis] >= interior_upper[axis]:
                    face = 2 * axis + 1
                else:
                    continue
                if face_neighbor_ids[block_id, face] >= 0:
                    has_sibling = True
                    break
            count += has_sibling
    return count


def benchmark_case(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    halo: int,
    field_count: int,
    capacity: int,
    repeats: int,
) -> dict:
    root = i3(*root_shape)
    block = i3(*block_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    first = int(coord_to_rank[tuple(int(value // 2) for value in root)])
    ids = np.empty(capacity, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_chunk(first, faces, ids)
    selected_ids = ids[:selected_count]

    backing = np.empty(
        (faces.shape[0], field_count, *block_shape),
        dtype=np.float64,
    )
    x, y, z = np.indices(block_shape, dtype=np.float64)
    cell_values = 100.0 * x + 10.0 * y + z + 1.0
    for block_id in range(backing.shape[0]):
        for field in range(field_count):
            backing[block_id, field] = (
                100000.0 * block_id + 10000.0 * field + cell_values
            )

    lower = i3(halo, halo, halo)
    upper = lower + block
    spatial_shape_array = upper + halo
    spatial_shape = tuple(int(value) for value in spatial_shape_array)
    workspace = np.full(
        (capacity, field_count, *spatial_shape),
        np.nan,
        dtype=np.float64,
    )
    payload = workspace[:selected_count]
    fields = np.arange(field_count, dtype=np.int64)
    modes = np.empty((field_count, 6), dtype=np.uint8)
    for field in range(field_count):
        for face in range(6):
            modes[field, face] = (field + face) % 3
    normals = i3(-1, -1, -1)
    zero = i3(0, 0, 0)

    def gather() -> None:
        gather_blocks_into(
            backing,
            zero,
            block,
            selected_ids,
            fields,
            payload,
            lower,
        )

    def physical() -> None:
        fill_physical_halos(
            payload,
            lower,
            upper,
            selected_ids,
            faces,
            modes,
            normals,
        )

    def same_level() -> None:
        fill_same_level_halos(
            payload,
            lower,
            upper,
            selected_ids,
            primary_count,
            faces,
            modes,
            normals,
        )

    gather()
    physical()
    same_level()
    kernel_seconds = median_seconds(same_level, repeats)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    same_level()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def composed() -> None:
        gather()
        physical()
        same_level()

    composed_seconds = median_seconds(composed, repeats)
    written_cells = written_cell_count(
        selected_ids,
        primary_count,
        faces,
        lower,
        upper,
        spatial_shape,
    )
    written_values = written_cells * field_count
    gathered_bytes = (
        selected_count * field_count * int(np.prod(block)) * 8
    )
    return {
        "capacity": capacity,
        "selected_count": selected_count,
        "primary_count": primary_count,
        "field_count": field_count,
        "halo": halo,
        "spatial_shape": spatial_shape,
        "managed_workspace_bytes": workspace_nbytes(
            capacity,
            field_count,
            spatial_shape_array,
        ),
        "written_cells": written_cells,
        "written_values": written_values,
        "kernel_seconds": kernel_seconds,
        "million_written_values_per_second": written_values / kernel_seconds / 1e6,
        "composed_seconds": composed_seconds,
        "composed_gather_gb_per_second": gathered_bytes / composed_seconds / 1e9,
        "kernel_traced_current_delta_bytes": after_current - before_current,
        "kernel_traced_peak_delta_bytes": peak - before_current,
    }


def reference_comparison(repeats: int) -> dict:
    root = i3(3, 3, 3)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    ids = np.empty(27, dtype=np.int64)
    first = int(coord_to_rank[1, 1, 1])
    primary_count, selected_count = plan_level1_halo_chunk(first, faces, ids)
    selected_ids = ids[:selected_count]
    lower = i3(1, 1, 1)
    upper = i3(5, 5, 5)
    payload = np.full((selected_count, 1, 6, 6, 6), np.nan)
    interior = np.arange(
        faces.shape[0] * 4**3,
        dtype=np.float64,
    ).reshape(faces.shape[0], 1, 4, 4, 4)
    gather_blocks_into(
        interior,
        i3(0, 0, 0),
        i3(4, 4, 4),
        selected_ids,
        np.array([0], dtype=np.int64),
        payload,
        lower,
    )
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    fill_physical_halos(
        payload,
        lower,
        upper,
        selected_ids,
        faces,
        modes,
        normals,
    )
    expected = payload.copy()
    started = time.perf_counter()
    fill_same_level_halos_reference(
        expected,
        lower,
        upper,
        selected_ids,
        primary_count,
        faces,
        modes,
        normals,
    )
    reference_seconds = time.perf_counter() - started
    actual = payload.copy()

    def compiled() -> None:
        fill_same_level_halos(
            actual,
            lower,
            upper,
            selected_ids,
            primary_count,
            faces,
            modes,
            normals,
        )

    compiled_seconds = median_seconds(compiled, repeats)
    return {
        "primary_count": primary_count,
        "selected_count": selected_count,
        "reference_seconds": reference_seconds,
        "compiled_seconds": compiled_seconds,
        "speedup": reference_seconds / compiled_seconds,
        "bitwise_equal": bool(
            np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-shape", default="12,10,8")
    parser.add_argument("--block-shape", default="8,8,8")
    parser.add_argument("--halos", default="1,2,4")
    parser.add_argument("--fields", default="1,4")
    parser.add_argument("--capacities", default="64,256")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    root_shape = tuple(int(value) for value in args.root_shape.split(","))
    block_shape = tuple(int(value) for value in args.block_shape.split(","))
    cases = []
    for capacity in (int(value) for value in args.capacities.split(",")):
        for field_count in (int(value) for value in args.fields.split(",")):
            for halo in (int(value) for value in args.halos.split(",")):
                cases.append(
                    benchmark_case(
                        root_shape,
                        block_shape,
                        halo,
                        field_count,
                        capacity,
                        args.repeats,
                    )
                )
    print(
        json.dumps(
            {
                "capability": "HAL-002",
                "root_shape": root_shape,
                "block_shape": block_shape,
                "cases": cases,
                "reference": reference_comparison(args.repeats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
