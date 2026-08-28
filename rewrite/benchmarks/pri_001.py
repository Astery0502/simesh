"""PRI-001 dense-prefix throughput, compatibility, and WSP composition."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._primary import fill_ascending_primary_prefix_unchecked
from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.primary import fill_ascending_primary_prefix
from simesh_rewrite.workspace import workspace_slot_capacity


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def traverse_canonical(block_count: int, primary_ids: np.ndarray) -> int:
    first = 0
    calls = 0
    while first < block_count:
        count = fill_ascending_primary_prefix(first, block_count, primary_ids)
        first += count
        calls += 1
    return calls


def traverse_compatibility(
    face_neighbor_ids: np.ndarray,
    primary_ids: np.ndarray,
) -> int:
    first = 0
    calls = 0
    while first < face_neighbor_ids.shape[0]:
        count, selected = plan_level1_chunk(
            first,
            face_neighbor_ids,
            False,
            primary_ids,
        )
        if count != selected:
            raise AssertionError("no-closure compatibility counts diverged")
        first += count
        calls += 1
    return calls


def measure_capacity(
    block_count: int,
    capacity: int,
    repeats: int,
) -> dict:
    primary_ids = np.full(capacity, -1, dtype=np.int64)
    faces = np.full((block_count, 6), np.iinfo(np.int64).max, dtype=np.int64)
    expected_calls = (block_count + capacity - 1) // capacity
    if traverse_canonical(block_count, primary_ids) != expected_calls:
        raise AssertionError("canonical traversal call count is not maximal")
    canonical_seconds = median_seconds(
        lambda: traverse_canonical(block_count, primary_ids), repeats
    )
    compatibility_seconds = median_seconds(
        lambda: traverse_compatibility(faces, primary_ids), repeats
    )
    full_count = fill_ascending_primary_prefix(0, block_count, primary_ids)
    unchecked_seconds = median_seconds(
        lambda: fill_ascending_primary_prefix_unchecked(
            0, block_count, primary_ids
        ),
        repeats,
    )
    final_first = max(0, block_count - max(1, capacity // 3))
    suffix_before = primary_ids.copy()
    final_count = fill_ascending_primary_prefix(
        final_first, block_count, primary_ids
    )
    if not np.array_equal(primary_ids[final_count:], suffix_before[final_count:]):
        raise AssertionError("final prefix changed reusable suffix")

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_ascending_primary_prefix(0, block_count, primary_ids)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "capacity": capacity,
        "calls": expected_calls,
        "canonical_seconds": canonical_seconds,
        "compatibility_seconds": compatibility_seconds,
        "million_ids_per_second": block_count / canonical_seconds / 1.0e6,
        "compatibility_million_ids_per_second": (
            block_count / compatibility_seconds / 1.0e6
        ),
        "unchecked_full_prefix_seconds": unchecked_seconds,
        "full_prefix_count": int(full_count),
        "final_prefix_count": int(final_count),
        "full_prefix_bytes_written": int(8 * full_count),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-count", type=int, default=32768)
    parser.add_argument("--capacities", default="1,8,64,256,1024,32768")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    capacities = [int(value) for value in args.capacities.split(",")]
    wsp_shape = np.asarray([16, 16, 16], dtype=np.int64)
    wsp_capacity = workspace_slot_capacity(
        2 * 1024 * 1024,
        args.block_count,
        2,
        wsp_shape,
    )
    report = {
        "capability": "PRI-001",
        "block_count": args.block_count,
        "repeats": args.repeats,
        "results": [
            measure_capacity(args.block_count, capacity, args.repeats)
            for capacity in capacities
        ],
        "wsp_composition": {
            "budget_bytes": 2 * 1024 * 1024,
            "field_count": 2,
            "workspace_shape": wsp_shape.tolist(),
            "capacity": wsp_capacity,
            "traversal_calls": (
                (args.block_count + wsp_capacity - 1) // wsp_capacity
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
