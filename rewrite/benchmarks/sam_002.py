"""General and bounded-path measurements for SAM-002."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_chunk, workspace_nbytes
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.sampling import (
    place_level1_blocks,
    sample_level1_zero_order,
)
from simesh_rewrite.sampling_reference import sample_level1_zero_order_reference
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
    result = np.empty((block_count, field_count, *block_shape), dtype=np.float64)
    x, y, z = np.indices(block_shape, dtype=np.float64)
    local = 100.0 * x + 10.0 * y + z + 1.0
    for block_id in range(block_count):
        for field in range(field_count):
            result[block_id, field] = (
                100000.0 * block_id + 10000.0 * field + local
            )
    return result


def current_mesh(
    root_shape: tuple[int, int, int],
    block: np.ndarray,
    domain: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    field_count: int,
) -> AMRMesh:
    forest = AMRForest(
        3,
        *root_shape,
        np.ones(int(np.prod(root_shape)), dtype=np.int32),
    )
    return AMRMesh(
        3,
        block.astype(np.uint32),
        domain.astype(np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(0),
        np.uint32(field_count),
        forest,
    )


def sampling_case(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    field_count: int,
    output_shape: tuple[int, int, int],
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    shuffled: bool,
    repeats: int,
) -> dict:
    root = i3(*root_shape)
    block = i3(*block_shape)
    domain = root * block
    domain_lower = np.array([-2.0, 1.0, 10.0])
    domain_upper = np.array([6.0, 7.0, 14.0])
    coord_to_rank, rank_to_coord = level1_morton(root)
    backing = backing_data(rank_to_coord.shape[0], field_count, block_shape)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    payload = backing
    if shuffled:
        np.random.default_rng(1701).shuffle(block_ids)
        payload = np.ascontiguousarray(backing[block_ids])
    output = np.empty((field_count, *output_shape), dtype=np.float64)
    zero = i3(0, 0, 0)

    def sample() -> None:
        sample_level1_zero_order(
            payload,
            zero,
            block,
            block_ids,
            domain_lower,
            domain_upper,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )

    sample()
    seconds = median_seconds(sample, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    sample()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = output.size
    result = {
        "field_count": field_count,
        "output_shape": output_shape,
        "shuffled": shuffled,
        "seconds": seconds,
        "million_values_per_second": values / seconds / 1e6,
        "read_write_gb_per_second": 16 * values / seconds / 1e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }

    if not shuffled:
        mesh = current_mesh(
            root_shape,
            block,
            domain,
            domain_lower,
            domain_upper,
            field_count,
        )
        current = np.empty_like(output)

        def current_sample() -> None:
            mesh.uniform_grid_zero_order(
                backing,
                current,
                np.asarray(output_shape, dtype=np.uint32),
                sample_lower,
                sample_upper,
            )

        current_sample()
        mismatch_values = int(
            np.count_nonzero(output.view(np.uint64) != current.view(np.uint64))
        )
        result.update(
            {
                "current_seconds": median_seconds(current_sample, repeats),
                "current_bitwise_equal": mismatch_values == 0,
                "current_mismatch_values": mismatch_values,
            }
        )

        if np.array_equal(sample_lower, domain_lower) and np.array_equal(
            sample_upper, domain_upper
        ) and output_shape == tuple(int(value) for value in domain):
            placed = np.empty_like(output)

            def place() -> None:
                place_level1_blocks(
                    backing,
                    zero,
                    block,
                    block_ids,
                    domain,
                    block,
                    coord_to_rank,
                    rank_to_coord,
                    placed,
                )

            place()
            result.update(
                {
                    "sam_001_seconds": median_seconds(place, repeats),
                    "sam_001_bitwise_equal": bool(
                        np.array_equal(
                            output.view(np.uint64),
                            placed.view(np.uint64),
                        )
                    ),
                }
            )
    return result


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
    domain_lower = np.array([-2.0, 1.0, 10.0])
    domain_upper = np.array([6.0, 7.0, 14.0])
    sample_lower = np.array([-1.5, 1.5, 10.25])
    sample_upper = np.array([5.5, 6.5, 13.75])
    output_shape = (96, 72, 48)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    backing = backing_data(rank_to_coord.shape[0], field_count, block_shape)
    ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty((capacity, field_count, *block_shape), dtype=np.float64)
    output = np.empty((field_count, *output_shape), dtype=np.float64)
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
            sample_level1_zero_order(
                payload[:primary_count],
                zero,
                block,
                ids[:primary_count],
                domain_lower,
                domain_upper,
                domain,
                block,
                coord_to_rank,
                rank_to_coord,
                sample_lower,
                sample_upper,
                output,
            )
            first += primary_count
            chunks += 1
        return chunks

    chunks = traverse()
    seconds = median_seconds(traverse, repeats)
    expected = np.empty_like(output)
    all_ids = np.arange(faces.shape[0], dtype=np.int64)
    sample_level1_zero_order(
        backing,
        zero,
        block,
        all_ids,
        domain_lower,
        domain_upper,
        domain,
        block,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        expected,
    )
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    traverse()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "capacity": capacity,
        "chunks": chunks,
        "seconds": seconds,
        "million_values_per_second": output.size / seconds / 1e6,
        "managed_workspace_bytes": workspace_nbytes(
            capacity,
            field_count,
            block,
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "bitwise_equal_to_full_sampling": bool(
            np.array_equal(output.view(np.uint64), expected.view(np.uint64))
        ),
    }


def reference_comparison(repeats: int) -> dict:
    root = i3(2, 2, 2)
    block = i3(2, 2, 2)
    domain = root * block
    domain_lower = np.array([-1.0, 2.0, 10.0])
    domain_upper = np.array([3.0, 6.0, 14.0])
    sample_lower = np.array([-0.75, 2.25, 10.5])
    sample_upper = np.array([2.75, 5.75, 13.5])
    coord_to_rank, rank_to_coord = level1_morton(root)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)[::-1].copy()
    payload = np.ascontiguousarray(
        backing_data(rank_to_coord.shape[0], 2, (2, 2, 2))[block_ids]
    )
    expected = np.full((2, 7, 5, 3), -1.0)
    actual = expected.copy()
    zero = i3(0, 0, 0)
    started = time.perf_counter()
    sample_level1_zero_order_reference(
        payload,
        zero,
        block_ids,
        domain_lower,
        domain_upper,
        domain,
        block,
        coord_to_rank,
        sample_lower,
        sample_upper,
        expected,
    )
    reference_seconds = time.perf_counter() - started

    def compiled() -> None:
        sample_level1_zero_order(
            payload,
            zero,
            block,
            block_ids,
            domain_lower,
            domain_upper,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
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
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    root_shape = (8, 6, 4)
    block_shape = (8, 8, 8)
    domain_lower = np.array([-2.0, 1.0, 10.0])
    domain_upper = np.array([6.0, 7.0, 14.0])
    scenarios = [
        ("coarse", (32, 24, 16), domain_lower, domain_upper),
        ("native", (64, 48, 32), domain_lower, domain_upper),
        ("fine", (96, 72, 48), domain_lower, domain_upper),
        (
            "subdomain",
            (40, 30, 20),
            np.array([-1.5, 1.5, 10.25]),
            np.array([5.5, 6.5, 13.75]),
        ),
    ]
    cases = []
    for field_count in (1, 4):
        for name, output_shape, sample_lower, sample_upper in scenarios:
            result = sampling_case(
                root_shape,
                block_shape,
                field_count,
                output_shape,
                sample_lower,
                sample_upper,
                False,
                args.repeats,
            )
            result["scenario"] = name
            cases.append(result)
        shuffled = sampling_case(
            root_shape,
            block_shape,
            field_count,
            scenarios[2][1],
            domain_lower,
            domain_upper,
            True,
            args.repeats,
        )
        shuffled["scenario"] = "fine"
        cases.append(shuffled)

    print(
        json.dumps(
            {
                "capability": "SAM-002",
                "root_shape": root_shape,
                "block_shape": block_shape,
                "sampling_cases": cases,
                "bounded_composition": [
                    bounded_composition(
                        root_shape,
                        block_shape,
                        4,
                        capacity,
                        args.repeats,
                    )
                    for capacity in (32, 128)
                ],
                "reference": reference_comparison(args.repeats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
