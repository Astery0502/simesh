"""Retained metadata runtime and allocation probe for FND-002."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.access import required_input_region, valid_output_region


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def time_calls(function, arguments, iterations: int, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            function(*arguments)
        samples.append((time.perf_counter() - started) / iterations)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    output_lower = i3(4, 5, 6)
    output_upper = i3(36, 53, 70)
    lower_reach = i3(2, 0, 3)
    upper_reach = i3(0, 4, 1)
    required = required_input_region(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )
    required_seconds = time_calls(
        required_input_region,
        (output_lower, output_upper, lower_reach, upper_reach),
        args.iterations,
        args.repeats,
    )
    valid_seconds = time_calls(
        valid_output_region,
        (*required, lower_reach, upper_reach),
        args.iterations,
        args.repeats,
    )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    result = required_input_region(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output_extent = output_upper - output_lower
    exact_extent = output_extent + lower_reach + upper_reach
    symmetric_reach = np.maximum(lower_reach, upper_reach)
    symmetric_extent = output_extent + 2 * symmetric_reach
    report = {
        "capability": "FND-002",
        "iterations": args.iterations,
        "repeats": args.repeats,
        "required_input_call_ns": required_seconds * 1.0e9,
        "valid_output_call_ns": valid_seconds * 1.0e9,
        "returned_index_bytes": result[0].nbytes + result[1].nbytes,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "asymmetric_required_cells": int(np.prod(exact_extent)),
        "symmetric_required_cells": int(np.prod(symmetric_extent)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
