"""LIM-001 scalar-boundary and small RST composition measurements."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from simesh_rewrite._limiter import three_point_limited_slope_unchecked
from simesh_rewrite.limiter import three_point_limited_slope
from simesh_rewrite.limiter_reference import (
    three_point_limited_slope_reference,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def value_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def arrays_bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def fixed_branch_cases() -> tuple[np.ndarray, list[str]]:
    maximum = np.float64(np.finfo(np.float64).max)
    half_maximum = np.float64(maximum * np.float64(0.5))
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    cases = np.asarray(
        [
            (0.0, 1.0, 3.0),
            (3.0, 2.0, 0.0),
            (0.0, 2.0, 1.0),
            (5.0, 5.0, 5.0),
            (np.nan, 1.0, 2.0),
            (-np.inf, 0.0, np.inf),
            (-maximum, 0.0, half_maximum),
            (0.0, smallest, np.float64(3.0) * smallest),
        ],
        dtype=np.float64,
    )
    labels = [
        "positive_monotone",
        "negative_monotone",
        "extremum",
        "plateau",
        "nan",
        "infinity",
        "centered_overflow",
        "subnormal",
    ]
    return cases, labels


def build_call_arrays(call_count: int):
    cases, labels = fixed_branch_cases()
    left = np.ascontiguousarray(np.resize(cases[:, 0], call_count))
    center = np.ascontiguousarray(np.resize(cases[:, 1], call_count))
    right = np.ascontiguousarray(np.resize(cases[:, 2], call_count))
    quotient, remainder = divmod(call_count, len(labels))
    case_counts = [quotient + (index < remainder) for index in range(len(labels))]
    return left, center, right, labels, case_counts


def evaluate_calls(
    operation,
    left: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    output: np.ndarray,
) -> None:
    for index in range(output.size):
        output[index] = operation(left[index], center[index], right[index])


def classify_outputs(output: np.ndarray) -> dict:
    bits = output.view(np.uint64)
    finite = np.isfinite(output)
    return {
        "positive_zero": int(np.count_nonzero(bits == np.uint64(0))),
        "negative_zero": int(
            np.count_nonzero(bits == np.uint64(0x8000000000000000))
        ),
        "finite_positive_nonzero": int(
            np.count_nonzero(finite & (output > 0.0))
        ),
        "finite_negative_nonzero": int(
            np.count_nonzero(finite & (output < 0.0))
        ),
        "positive_infinity": int(np.count_nonzero(np.isposinf(output))),
        "negative_infinity": int(np.count_nonzero(np.isneginf(output))),
        "nan": int(np.count_nonzero(np.isnan(output))),
    }


def branch_counts(
    cases: np.ndarray,
    labels: list[str],
    case_counts: list[int],
) -> tuple[dict, dict]:
    label_counts = {
        label: int(case_counts[index]) for index, label in enumerate(labels)
    }
    centered_slope_branch_counts = {
        "positive": 0,
        "negative": 0,
        "fallback": 0,
    }
    for index, (left, center, right) in enumerate(cases):
        count = int(case_counts[index])
        with np.errstate(all="ignore"):
            slope_l = np.float64(center - left)
            slope_r = np.float64(right - center)
            sum_lr = np.float64(slope_l + slope_r)
            slope_c = np.float64(np.float64(0.5) * sum_lr)
        if slope_c > 0.0:
            centered_slope_branch_counts["positive"] += count
        elif slope_c < 0.0:
            centered_slope_branch_counts["negative"] += count
        else:
            centered_slope_branch_counts["fallback"] += count
    return label_counts, centered_slope_branch_counts


def rotating_scalar_timing(call_count: int, repeats: int) -> dict:
    left, center, right, labels, case_counts = build_call_arrays(call_count)
    cases, _ = fixed_branch_cases()
    operations = (
        ("validated", three_point_limited_slope),
        ("unchecked", three_point_limited_slope_unchecked),
        ("manual_reference", three_point_limited_slope_reference),
    )
    outputs = {
        name: np.empty(call_count, dtype=np.float64) for name, _ in operations
    }

    # Exact full per-call arrays are produced and compared outside timing.
    for name, operation in operations:
        evaluate_calls(operation, left, center, right, outputs[name])
    exact_validated_unchecked = arrays_bits_equal(
        outputs["validated"], outputs["unchecked"]
    )
    exact_validated_reference = arrays_bits_equal(
        outputs["validated"], outputs["manual_reference"]
    )
    if not exact_validated_unchecked or not exact_validated_reference:
        raise AssertionError("LIM full per-call result arrays disagree")

    samples = {name: [] for name, _ in operations}
    rotation_orders = []
    operation_count = len(operations)
    for repeat in range(repeats):
        order = [
            operations[(repeat + offset) % operation_count]
            for offset in range(operation_count)
        ]
        rotation_orders.append([name for name, _ in order])
        for name, operation in order:
            started = time.perf_counter()
            evaluate_calls(
                operation,
                left,
                center,
                right,
                outputs[name],
            )
            samples[name].append(time.perf_counter() - started)

    # Timed calls also wrote complete arrays; verify them again outside timing.
    if not arrays_bits_equal(outputs["validated"], outputs["unchecked"]):
        raise AssertionError("timed validated/unchecked LIM arrays disagree")
    if not arrays_bits_equal(outputs["validated"], outputs["manual_reference"]):
        raise AssertionError("timed validated/reference LIM arrays disagree")

    medians = {
        name: statistics.median(values) for name, values in samples.items()
    }
    label_counts, centered_branch_counts = branch_counts(
        cases, labels, case_counts
    )
    return {
        "calls_per_path": call_count,
        "input_scalar_type": "numpy.float64",
        "full_output_array_bytes_per_path": outputs["validated"].nbytes,
        "exact_full_validated_unchecked_arrays": exact_validated_unchecked,
        "exact_full_validated_reference_arrays": exact_validated_reference,
        "rotation_orders": rotation_orders,
        "median_seconds": medians,
        "million_calls_per_second": {
            name: call_count / seconds / 1.0e6
            for name, seconds in medians.items()
        },
        "validated_over_unchecked": medians["validated"] / medians["unchecked"],
        "validated_over_manual_reference": (
            medians["validated"] / medians["manual_reference"]
        ),
        "input_case_counts": label_counts,
        "descriptive_centered_slope_branch_counts": centered_branch_counts,
        "descriptive_output_type_counts": classify_outputs(outputs["validated"]),
    }


def traced_validated_scalar_call() -> dict:
    three_point_limited_slope(0.0, 1.0, 3.0)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    result = three_point_limited_slope(0.0, 1.0, 3.0)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "retained_bytes": after_current - before_current,
        "peak_bytes": peak - before_current,
        "result": result,
        "result_bits": value_bits(result),
    }


def rst_coarse_profile_composition() -> dict:
    """Small producer/consumer evidence; PRL owns meaningful inline throughput."""
    x = np.arange(8, dtype=np.float64)[:, None, None]
    fine = np.empty((1, 4, 8, 4, 4), dtype=np.float64)
    fine[0, 0].fill(5.0)
    fine[0, 1] = np.broadcast_to(x + 0.5, (8, 4, 4))
    fine[0, 2] = np.broadcast_to(
        np.where(x < 4, 0.0, 10.0), (8, 4, 4)
    )
    fine[0, 3] = np.broadcast_to((x + 0.5) ** 2, (8, 4, 4))
    coarse = np.empty((1, 4, 4, 2, 2), dtype=np.float64)
    started = time.perf_counter()
    restrict_cartesian_2to1_into(
        fine,
        i3(0, 0, 0),
        i3(8, 4, 4),
        coarse,
        i3(0, 0, 0),
    )
    rst_seconds = time.perf_counter() - started

    triples = np.empty((4, 3), dtype=np.float64)
    for field in range(4):
        triples[field] = coarse[0, field, 0:3, 0, 0]
    validated = np.empty(4, dtype=np.float64)
    unchecked = np.empty(4, dtype=np.float64)
    reference = np.empty(4, dtype=np.float64)
    for row in range(triples.shape[0]):
        left, center, right = triples[row]
        validated[row] = three_point_limited_slope(left, center, right)
        unchecked[row] = three_point_limited_slope_unchecked(
            left, center, right
        )
        reference[row] = three_point_limited_slope_reference(
            left, center, right
        )
    exact_unchecked = arrays_bits_equal(validated, unchecked)
    exact_reference = arrays_bits_equal(validated, reference)
    if not exact_unchecked or not exact_reference:
        raise AssertionError("RST-produced explicit LIM triples disagree")
    return {
        "description": (
            "RST-produced coarse profile with four explicit x triples; "
            "no slope array/cache API is introduced"
        ),
        "rst_seconds_descriptive": rst_seconds,
        "coarse_profile_shape": list(coarse.shape),
        "triples": triples.tolist(),
        "validated_slopes": validated.tolist(),
        "validated_slope_bits": validated.view(np.uint64).astype(object).tolist(),
        "exact_validated_unchecked_array": exact_unchecked,
        "exact_validated_reference_array": exact_reference,
        "meaningful_inline_prl_throughput": "deferred to PRL-001",
        "not_an_array_slope_or_cache_benchmark": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.calls < 1 or args.repeats < 1:
        raise ValueError("calls and repeats must be positive")

    report = {
        "capability": "LIM-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "scope": (
            "descriptive scalar Python/Cython boundary timing; meaningful inline "
            "region throughput and inline-versus-cache decisions are deferred to PRL-001"
        ),
        "scalar_timing": rotating_scalar_timing(args.calls, args.repeats),
        "validated_scalar_tracemalloc": traced_validated_scalar_call(),
        "rst_coarse_profile_composition": rst_coarse_profile_composition(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
