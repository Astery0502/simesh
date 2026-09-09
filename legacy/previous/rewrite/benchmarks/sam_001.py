"""Placement and bounded composition measurements for SAM-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import (
    plan_level1_chunk,
    workspace_nbytes,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.sampling import place_level1_blocks
from simesh_rewrite.sampling_reference import place_level1_blocks_reference
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


def backing_data(
    block_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    result = np.empty(
        (block_count, field_count, *block_shape),
        dtype=np.float64,
    )
    x, y, z = np.indices(block_shape, dtype=np.float64)
    local = 100.0 * x + 10.0 * y + z + 1.0
    for block_id in range(block_count):
        for field in range(field_count):
            result[block_id, field] = (
                100000.0 * block_id + 10000.0 * field + local
            )
    return result


def placement_case(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    field_count: int,
    selected_count: int,
    shuffled: bool,
    repeats: int,
) -> dict:
    root = i3(*root_shape)
    block = i3(*block_shape)
    domain = root * block
    coord_to_rank, rank_to_coord = level1_morton(root)
    selected_count = min(selected_count, rank_to_coord.shape[0])
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    if shuffled:
        np.random.default_rng(1701).shuffle(block_ids)
    block_ids = np.ascontiguousarray(block_ids[:selected_count])
    backing = backing_data(rank_to_coord.shape[0], field_count, block_shape)
    payload = np.ascontiguousarray(backing[block_ids])
    output = np.full(
        (field_count, *(int(value) for value in domain)),
        -1.0,
        dtype=np.float64,
    )
    zero = i3(0, 0, 0)

    def place() -> None:
        place_level1_blocks(
            payload,
            zero,
            block,
            block_ids,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            output,
        )

    place()
    seconds = median_seconds(place, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    place()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = selected_count * field_count * int(np.prod(block))
    return {
        "field_count": field_count,
        "selected_count": selected_count,
        "shuffled": shuffled,
        "seconds": seconds,
        "million_values_per_second": values / seconds / 1e6,
        "read_write_gb_per_second": 16 * values / seconds / 1e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def bounded_composition(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    field_count: int,
    capacity: int,
    repeats: int,
) -> dict:
    root = i3(*root_shape)
    block = i3(*block_shape)
    domain = root * block
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    backing = backing_data(rank_to_coord.shape[0], field_count, block_shape)
    ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty(
        (capacity, field_count, *block_shape),
        dtype=np.float64,
    )
    output = np.empty(
        (field_count, *(int(value) for value in domain)),
        dtype=np.float64,
    )
    fields = np.arange(field_count, dtype=np.int64)
    zero = i3(0, 0, 0)

    def traverse() -> int:
        first = 0
        chunks = 0
        while first < faces.shape[0]:
            primary_count, selected_count = plan_level1_chunk(
                first,
                faces,
                False,
                ids,
            )
            gather_blocks_into(
                backing,
                zero,
                block,
                ids[:selected_count],
                fields,
                payload[:selected_count],
                zero,
            )
            place_level1_blocks(
                payload[:primary_count],
                zero,
                block,
                ids[:primary_count],
                domain,
                block,
                coord_to_rank,
                rank_to_coord,
                output,
            )
            first += primary_count
            chunks += 1
        return chunks

    chunks = traverse()
    seconds = median_seconds(traverse, repeats)
    expected = np.empty_like(output)
    all_ids = np.arange(faces.shape[0], dtype=np.int64)
    place_level1_blocks(
        backing,
        zero,
        block,
        all_ids,
        domain,
        block,
        coord_to_rank,
        rank_to_coord,
        expected,
    )
    values = output.size
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    traverse()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "capacity": capacity,
        "chunks": chunks,
        "seconds": seconds,
        "million_values_per_second": values / seconds / 1e6,
        "managed_workspace_bytes": workspace_nbytes(
            capacity,
            field_count,
            block,
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "bitwise_equal_to_full_placement": bool(
            np.array_equal(output.view(np.uint64), expected.view(np.uint64))
        ),
    }


def current_comparison(repeats: int) -> dict:
    root_shape = (8, 6, 4)
    block_shape = (8, 8, 8)
    field_count = 2
    root = i3(*root_shape)
    block = i3(*block_shape)
    domain = root * block
    coord_to_rank, rank_to_coord = level1_morton(root)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    data = backing_data(rank_to_coord.shape[0], field_count, block_shape)
    rewritten = np.empty(
        (field_count, *(int(value) for value in domain)),
        dtype=np.float64,
    )
    zero = i3(0, 0, 0)

    def rewrite_place() -> None:
        place_level1_blocks(
            data,
            zero,
            block,
            block_ids,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            rewritten,
        )

    forest = AMRForest(
        3,
        *root_shape,
        np.ones(rank_to_coord.shape[0], dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block.astype(np.uint32),
        domain.astype(np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(0),
        np.uint32(field_count),
        forest,
    )
    current = np.empty_like(rewritten)

    def current_place() -> None:
        mesh.uniform_full_level1(data, current)

    rewrite_place()
    current_place()
    return {
        "root_shape": root_shape,
        "block_shape": block_shape,
        "field_count": field_count,
        "rewrite_seconds": median_seconds(rewrite_place, repeats),
        "current_seconds": median_seconds(current_place, repeats),
        "bitwise_equal": bool(
            np.array_equal(rewritten.view(np.uint64), current.view(np.uint64))
        ),
    }


def reference_comparison(repeats: int) -> dict:
    root = i3(3, 2, 4)
    block = i3(2, 4, 2)
    domain = root * block
    coord_to_rank, rank_to_coord = level1_morton(root)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)[::-1].copy()
    payload = np.ascontiguousarray(
        backing_data(rank_to_coord.shape[0], 2, (2, 4, 2))[block_ids]
    )
    zero = i3(0, 0, 0)
    expected = np.empty((2, *(int(value) for value in domain)))
    actual = np.empty_like(expected)
    started = time.perf_counter()
    place_level1_blocks_reference(
        payload,
        zero,
        block_ids,
        block,
        rank_to_coord,
        expected,
    )
    reference_seconds = time.perf_counter() - started

    def compiled() -> None:
        place_level1_blocks(
            payload,
            zero,
            block,
            block_ids,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            actual,
        )

    compiled_seconds = median_seconds(compiled, repeats)
    return {
        "reference_seconds": reference_seconds,
        "compiled_seconds": compiled_seconds,
        "speedup": reference_seconds / compiled_seconds,
        "bitwise_equal": bool(
            np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-shape", default="16,12,8")
    parser.add_argument("--block-shape", default="8,8,8")
    parser.add_argument("--fields", default="1,4")
    parser.add_argument("--selected", default="64,512")
    parser.add_argument("--capacities", default="64,256")
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()

    root_shape = tuple(int(value) for value in args.root_shape.split(","))
    block_shape = tuple(int(value) for value in args.block_shape.split(","))
    cases = []
    for field_count in (int(value) for value in args.fields.split(",")):
        for selected_count in (int(value) for value in args.selected.split(",")):
            for shuffled in (False, True):
                cases.append(
                    placement_case(
                        root_shape,
                        block_shape,
                        field_count,
                        selected_count,
                        shuffled,
                        args.repeats,
                    )
                )
    composed = [
        bounded_composition(
            root_shape,
            block_shape,
            max(int(value) for value in args.fields.split(",")),
            capacity,
            args.repeats,
        )
        for capacity in (int(value) for value in args.capacities.split(","))
    ]
    print(
        json.dumps(
            {
                "capability": "SAM-001",
                "root_shape": root_shape,
                "block_shape": block_shape,
                "placement_cases": cases,
                "bounded_composition": composed,
                "current": current_comparison(args.repeats),
                "reference": reference_comparison(args.repeats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
