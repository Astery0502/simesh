"""Descriptive scalar-call and aggregate-gate probe for PBC-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.boundary_rules import (
    physical_halo_source_index,
    transform_physical_halo_value,
)


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    lower = np.array([3, 4, 5], dtype=np.int64)
    upper = np.array([11, 12, 13], dtype=np.int64)
    values = [1.5, -2.5, float("inf"), -0.0]

    def map_calls() -> None:
        for index in range(args.calls):
            face = index % 6
            axis = face // 2
            layer = index % 8 + 1
            target = (
                int(lower[axis]) - layer
                if face % 2 == 0
                else int(upper[axis]) + layer - 1
            )
            physical_halo_source_index(
                target,
                lower,
                upper,
                face,
                index % 4,
            )

    def value_calls() -> None:
        for index in range(args.calls):
            transform_physical_halo_value(
                values[index % len(values)],
                index % 3,
                1,
                index % 6,
                index % 4,
            )

    map_seconds = median_seconds(map_calls, args.repeats)
    value_seconds = median_seconds(value_calls, args.repeats)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    physical_halo_source_index(2, lower, upper, 0, 1)
    transform_physical_halo_value(-1.0, 1, 1, 1, 3)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "capability": "PBC-001",
        "calls_per_repeat": args.calls,
        "repeats": args.repeats,
        "mapping_seconds": map_seconds,
        "mapping_nanoseconds_per_call": map_seconds / args.calls * 1.0e9,
        "value_seconds": value_seconds,
        "value_nanoseconds_per_call": value_seconds / args.calls * 1.0e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "meaningful_composed_benchmarks": [
            "rewrite/benchmarks/hal_001.py",
            "rewrite/benchmarks/hal_002.py",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
