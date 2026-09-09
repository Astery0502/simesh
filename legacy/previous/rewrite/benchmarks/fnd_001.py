"""Retained runtime, throughput, and allocation probe for FND-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.foundation import copy_region_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def measure(size: int, repeats: int, warmups: int) -> dict[str, float | int]:
    nslot, nfield = 4, 3
    extent = i3(size, size, size)
    source_lower = i3(1, 2, 1)
    destination_lower = i3(2, 1, 3)
    source_shape = tuple(source_lower + extent + i3(2, 1, 2))
    destination_shape = tuple(destination_lower + extent + i3(1, 2, 1))
    source = np.ones((nslot, nfield, *source_shape), dtype=np.float64)
    destination = np.empty((nslot, nfield, *destination_shape), dtype=np.float64)

    for _ in range(warmups):
        copy_region_into(source, source_lower, destination, destination_lower, extent)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        copy_region_into(source, source_lower, destination, destination_lower, extent)
        samples.append(time.perf_counter() - started)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    copy_region_into(source, source_lower, destination, destination_lower, extent)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median_seconds = statistics.median(samples)
    payload_bytes = nslot * nfield * size**3 * np.dtype(np.float64).itemsize
    return {
        "size": size,
        "cells_copied": nslot * nfield * size**3,
        "median_seconds": median_seconds,
        "payload_gb_per_second": payload_bytes / median_seconds / 1.0e9,
        "memory_traffic_gb_per_second": 2 * payload_bytes / median_seconds / 1.0e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "source_bytes": source.nbytes,
        "destination_bytes": destination.nbytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="8,16,32,48")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    result = {
        "capability": "FND-001",
        "repeats": args.repeats,
        "warmups": args.warmups,
        "results": [measure(size, args.repeats, args.warmups) for size in sizes],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
