"""Retained gather/scatter throughput and allocation probe for STO-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._storage import (
    gather_blocks_into_unchecked,
    scatter_blocks_from_unchecked,
)
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_call(function, arguments, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*arguments)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure(
    block_size: int,
    selected_blocks: int,
    repeats: int,
    pattern: str,
    mode: str = "full",
) -> dict:
    nblock, nfield = max(128, selected_blocks), 4
    selected_fields = 3
    if mode == "full":
        spatial_shape = (block_size, block_size, block_size)
        lower = i3(0, 0, 0)
    elif mode == "padded_row":
        spatial_shape = (block_size + 4, block_size + 4, block_size + 4)
        lower = i3(2, 2, 2)
    elif mode == "padded_plane":
        spatial_shape = (block_size + 4, block_size + 4, block_size)
        lower = i3(2, 2, 0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    backing = np.ones(
        (nblock, nfield, *spatial_shape),
        dtype=np.float64,
    )
    block_ids = np.arange(selected_blocks, dtype=np.int64)
    if pattern == "permuted":
        block_ids = np.ascontiguousarray((block_ids * 37) % nblock)
    elif pattern == "duplicate":
        block_ids[1::2] = block_ids[::2]
    field_ids = np.array([2, 0, 3], dtype=np.int64)
    destination = np.empty(
        (selected_blocks, selected_fields, *spatial_shape),
        dtype=np.float64,
    )
    upper = lower + i3(block_size, block_size, block_size)
    extent = upper - lower
    gather_arguments = (
        backing,
        lower,
        upper,
        block_ids,
        field_ids,
        destination,
        lower,
    )
    scatter_arguments = (
        destination,
        lower,
        upper,
        block_ids,
        field_ids,
        backing,
        lower,
    )
    gather_seconds = median_call(gather_blocks_into, gather_arguments, repeats)
    scatter_seconds = median_call(scatter_blocks_from, scatter_arguments, repeats)
    unchecked_gather = median_call(
        gather_blocks_into_unchecked,
        (backing, lower, extent, block_ids, field_ids, destination, lower),
        repeats,
    )
    unchecked_scatter = median_call(
        scatter_blocks_from_unchecked,
        (destination, lower, extent, block_ids, field_ids, backing, lower),
        repeats,
    )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    gather_blocks_into(*gather_arguments)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    copied_bytes = (
        selected_blocks
        * selected_fields
        * block_size**3
        * np.dtype(np.float64).itemsize
    )
    return {
        "block_size": block_size,
        "selected_blocks": selected_blocks,
        "pattern": pattern,
        "mode": mode,
        "selected_pairs": selected_blocks * selected_fields,
        "copied_bytes": copied_bytes,
        "gather_seconds": gather_seconds,
        "scatter_seconds": scatter_seconds,
        "unchecked_gather_seconds": unchecked_gather,
        "unchecked_scatter_seconds": unchecked_scatter,
        "gather_payload_gb_per_second": copied_bytes / gather_seconds / 1.0e9,
        "scatter_payload_gb_per_second": copied_bytes / scatter_seconds / 1.0e9,
        "gather_traffic_gb_per_second": 2 * copied_bytes / gather_seconds / 1.0e9,
        "scatter_traffic_gb_per_second": 2 * copied_bytes / scatter_seconds / 1.0e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="8,16,32")
    parser.add_argument("--patterns", default="ordered,permuted,duplicate")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    patterns = args.patterns.split(",")
    base_cases = [
        (size, 32, pattern, "full")
        for size in sizes
        for pattern in patterns
    ]
    scaling_cases = [
        (8, 1, "ordered", "full"),
        (8, 4096, "ordered", "full"),
        (16, 1, "ordered", "full"),
        (16, 512, "ordered", "full"),
        (32, 1, "ordered", "full"),
        (32, 64, "ordered", "full"),
        (8, 32, "ordered", "padded_row"),
        (8, 32, "ordered", "padded_plane"),
    ]
    report = {
        "capability": "STO-001",
        "repeats": args.repeats,
        "results": [
            measure(size, selected_blocks, args.repeats, pattern, mode)
            for size, selected_blocks, pattern, mode in [
                *base_cases,
                *scaling_cases,
            ]
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
