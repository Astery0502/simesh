"""Functional block-adapter overhead probe for STO-003."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.blockio import array_block_reader, read_blocks_into
from simesh_rewrite.storage import gather_blocks_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_call(function, arguments, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*arguments)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure(block_size: int, selected_blocks: int, repeats: int) -> dict:
    nblock, nfield = max(128, selected_blocks), 4
    backing = np.ones(
        (nblock, nfield, block_size, block_size, block_size),
        dtype=np.float64,
    )
    block_ids = np.arange(selected_blocks, dtype=np.int64)
    field_ids = np.array([2, 0, 3], dtype=np.int64)
    destination = np.empty(
        (selected_blocks, 3, block_size, block_size, block_size),
        dtype=np.float64,
    )
    lower = i3(0, 0, 0)
    upper = i3(block_size, block_size, block_size)
    direct_arguments = (
        backing,
        lower,
        upper,
        block_ids,
        field_ids,
        destination,
        lower,
    )
    adapter_arguments = (
        array_block_reader(backing),
        lower,
        upper,
        block_ids,
        field_ids,
        destination,
        lower,
    )
    direct_seconds = median_call(gather_blocks_into, direct_arguments, repeats)
    adapter_seconds = median_call(read_blocks_into, adapter_arguments, repeats)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    read_blocks_into(*adapter_arguments)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    copied_bytes = selected_blocks * 3 * block_size**3 * 8
    return {
        "block_size": block_size,
        "selected_blocks": selected_blocks,
        "copied_bytes": copied_bytes,
        "direct_seconds": direct_seconds,
        "adapter_seconds": adapter_seconds,
        "adapter_minus_direct_seconds": adapter_seconds - direct_seconds,
        "adapter_over_direct": adapter_seconds / direct_seconds,
        "adapter_payload_gb_per_second": copied_bytes / adapter_seconds / 1.0e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    report = {
        "capability": "STO-003",
        "repeats": args.repeats,
        "results": [
            measure(8, 1, args.repeats),
            measure(8, 32, args.repeats),
            measure(16, 32, args.repeats),
            measure(32, 32, args.repeats),
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
