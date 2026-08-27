"""Kernel, accuracy, and memmap streaming measurements for RED-001."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import tempfile
import time
import tracemalloc

from numpy.lib.format import open_memmap
import numpy as np

from simesh_rewrite.chunking import plan_level1_chunk, workspace_nbytes
from simesh_rewrite.reductions import (
    accumulate_field_sum,
    finalize_field_sum,
    merge_field_sums,
)
from simesh_rewrite.storage import gather_blocks_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def finite_nonzero_ulp_distance(left: float, right: float) -> int | None:
    if not (np.isfinite(left) and np.isfinite(right) and left != 0.0 and right != 0.0):
        return None
    bits = np.asarray([left, right], dtype=np.float64).view(np.uint64)
    sign_mask = np.uint64(1 << 63)
    ordered = np.where(bits & sign_mask, ~bits, bits | sign_mask)
    return int(abs(int(ordered[0]) - int(ordered[1])))


def kernel_case(slot_count: int, spatial_shape: tuple[int, int, int], repeats: int) -> dict:
    rng = np.random.default_rng(1701)
    payload = rng.normal(size=(slot_count, 2, *spatial_shape))
    lower = i3(0, 0, 0)
    upper = i3(*spatial_shape)
    accumulator = np.array([0.0])
    numpy_accumulator = np.array([0.0])

    def compiled() -> None:
        accumulator[0] = 0.0
        accumulate_field_sum(payload, lower, upper, 1, accumulator)

    compiled()
    seconds = median_seconds(compiled, repeats)
    def numpy_reduce() -> None:
        numpy_accumulator[0] = np.sum(payload[:, 1])

    numpy_seconds = median_seconds(numpy_reduce, repeats)
    serial = float(accumulator[0])
    numpy_value = float(np.sum(payload[:, 1]))
    exact = math.fsum(float(value) for value in payload[:, 1].ravel())
    absolute_mass = math.fsum(abs(float(value)) for value in payload[:, 1].ravel())
    relative = (
        abs(serial - exact) / abs(exact)
        if abs(exact) >= np.sqrt(np.finfo(np.float64).eps) * max(1.0, absolute_mass)
        else None
    )
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    compiled()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = slot_count * int(np.prod(spatial_shape))
    return {
        "slot_count": slot_count,
        "spatial_shape": spatial_shape,
        "seconds": seconds,
        "numpy_seconds": numpy_seconds,
        "million_values_per_second": values / seconds / 1e6,
        "read_gb_per_second": 8 * values / seconds / 1e9,
        "serial": serial,
        "numpy": numpy_value,
        "math_fsum": exact,
        "absolute_error_vs_fsum": abs(serial - exact),
        "relative_error_vs_fsum": relative,
        "finite_nonzero_ulp_error_vs_fsum": finite_nonzero_ulp_distance(
            serial,
            exact,
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def stream_memmap(backing: np.memmap, capacity: int, repeats: int) -> dict:
    block_count = backing.shape[0]
    block_shape = np.asarray(backing.shape[2:], dtype=np.int64)
    faces = np.full((block_count, 6), -1, dtype=np.int64)
    ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty((capacity, 1, *backing.shape[2:]), dtype=np.float64)
    fields = np.array([0], dtype=np.int64)
    zero = i3(0, 0, 0)
    accumulator = np.array([0.0])

    def traverse() -> int:
        accumulator[0] = 0.0
        first = 0
        chunks = 0
        while first < block_count:
            primary_count, selected_count = plan_level1_chunk(
                first,
                faces,
                False,
                ids,
            )
            gather_blocks_into(
                backing,
                zero,
                block_shape,
                ids[:selected_count],
                fields,
                payload[:selected_count],
                zero,
            )
            accumulate_field_sum(
                payload[:primary_count],
                zero,
                block_shape,
                0,
                accumulator,
            )
            first += primary_count
            chunks += 1
        return chunks

    chunks = traverse()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    seconds = median_seconds(traverse, repeats)
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    traverse()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    managed = workspace_nbytes(capacity, 1, block_shape) + accumulator.nbytes
    values = backing.size
    return {
        "capacity": capacity,
        "chunks": chunks,
        "managed_workspace_bytes": managed,
        "mapped_payload_bytes": backing.nbytes,
        "seconds": seconds,
        "million_values_per_second": values / seconds / 1e6,
        "read_gb_per_second": 8 * values / seconds / 1e9,
        "sum_bits": int(accumulator.view(np.uint64)[0]),
        "sum": finalize_field_sum(accumulator),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "minor_page_fault_delta": usage_after.ru_minflt - usage_before.ru_minflt,
        "major_page_fault_delta": usage_after.ru_majflt - usage_before.ru_majflt,
    }


def adversarial_memmap() -> dict:
    block_count = 257
    pattern = np.array([1.0e16, 1.0, -1.0e16, 1.0] * 2)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "adversarial.npy")
        writable = open_memmap(
            path,
            mode="w+",
            dtype=np.float64,
            shape=(block_count, 1, 2, 2, 2),
        )
        writable.reshape(block_count, -1)[:] = pattern
        writable.flush()
        del writable
        backing = np.load(path, mmap_mode="r")
        capacities = {}
        for capacity in (1, 7, 31, 257):
            persistent = stream_memmap(backing, capacity, 1)

            faces = np.full((block_count, 6), -1, dtype=np.int64)
            ids = np.empty(capacity, dtype=np.int64)
            payload = np.empty((capacity, 1, 2, 2, 2))
            merged = np.array([0.0])
            zero = i3(0, 0, 0)
            first = 0
            while first < block_count:
                primary_count, selected_count = plan_level1_chunk(
                    first,
                    faces,
                    False,
                    ids,
                )
                gather_blocks_into(
                    backing,
                    zero,
                    i3(2, 2, 2),
                    ids[:selected_count],
                    np.array([0], dtype=np.int64),
                    payload[:selected_count],
                    zero,
                )
                partial = np.array([0.0])
                accumulate_field_sum(
                    payload[:primary_count],
                    zero,
                    i3(2, 2, 2),
                    0,
                    partial,
                )
                merge_field_sums(merged, partial)
                first += primary_count
            capacities[str(capacity)] = {
                "persistent": persistent["sum"],
                "merged_partials": float(merged[0]),
            }
        numpy_sum = float(np.sum(backing))
        exact = math.fsum(float(value) for value in backing.ravel())
    return {
        "capacities": capacities,
        "numpy": numpy_sum,
        "math_fsum": exact,
        "persistent_absolute_error_vs_fsum": abs(
            capacities["257"]["persistent"] - exact
        ),
        "persistent_finite_nonzero_ulp_error_vs_fsum": (
            finite_nonzero_ulp_distance(
                capacities["257"]["persistent"],
                exact,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--blocks", type=int, default=4096)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "performance.npy")
        writable = open_memmap(
            path,
            mode="w+",
            dtype=np.float64,
            shape=(args.blocks, 1, 8, 8, 8),
        )
        for block_id in range(args.blocks):
            writable[block_id, 0] = block_id + np.arange(8**3).reshape(8, 8, 8)
        writable.flush()
        del writable
        backing = np.load(path, mmap_mode="r")
        streams = [
            stream_memmap(backing, capacity, args.repeats)
            for capacity in (64, 256, 1024)
        ]
        started = time.perf_counter()
        numpy_sum = float(np.sum(backing))
        numpy_seconds = time.perf_counter() - started

    print(
        json.dumps(
            {
                "capability": "RED-001",
                "kernel_cases": [
                    kernel_case(slots, shape, args.repeats)
                    for slots, shape in (
                        (16, (8, 8, 8)),
                        (128, (8, 8, 8)),
                        (16, (16, 16, 16)),
                    )
                ],
                "memmap_streams": streams,
                "memmap_numpy": {
                    "sum": numpy_sum,
                    "seconds": numpy_seconds,
                },
                "adversarial": adversarial_memmap(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
