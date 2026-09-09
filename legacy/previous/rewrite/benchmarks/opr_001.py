"""Kernel and bounded gather/compute/scatter measurements for OPR-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.operators import scaled_difference_into
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from
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


def finite_error_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    local_scale: np.ndarray,
) -> dict:
    finite = np.isfinite(actual) & np.isfinite(reference)
    difference = np.abs(actual - reference)
    max_absolute = float(np.max(difference[finite], initial=0.0))
    relative_mask = finite & (
        np.abs(reference) >= np.sqrt(np.finfo(np.float64).eps) * local_scale
    )
    max_relative = float(
        np.max(
            difference[relative_mask] / np.abs(reference[relative_mask]),
            initial=0.0,
        )
    )

    nonzero_finite = finite & (actual != 0.0) & (reference != 0.0)
    if np.any(nonzero_finite):
        actual_bits = actual[nonzero_finite].view(np.uint64)
        reference_bits = reference[nonzero_finite].view(np.uint64)
        sign_mask = np.uint64(1 << 63)
        actual_ordered = np.where(
            actual_bits & sign_mask,
            ~actual_bits,
            actual_bits | sign_mask,
        )
        reference_ordered = np.where(
            reference_bits & sign_mask,
            ~reference_bits,
            reference_bits | sign_mask,
        )
        ulp = np.where(
            actual_ordered >= reference_ordered,
            actual_ordered - reference_ordered,
            reference_ordered - actual_ordered,
        )
        max_ulp = int(np.max(ulp, initial=np.uint64(0)))
    else:
        max_ulp = 0
    return {
        "max_absolute_error": max_absolute,
        "max_relative_error_away_from_cancellation": max_relative,
        "max_finite_nonzero_ulp_error": max_ulp,
    }


def kernel_case(
    slot_count: int,
    spatial_shape: tuple[int, int, int],
    repeats: int,
) -> dict:
    rng = np.random.default_rng(1701)
    source = rng.normal(size=(slot_count, 3, *spatial_shape))
    destination = np.full((slot_count, 2, *spatial_shape), -7.0)
    expected = destination.copy()
    product = np.empty((slot_count, *spatial_shape), dtype=np.float64)
    zero = i3(0, 0, 0)
    upper = i3(*spatial_shape)

    def compiled() -> None:
        scaled_difference_into(
            source,
            zero,
            upper,
            2,
            0,
            0.5,
            destination,
            1,
            zero,
        )

    def numpy_reference() -> None:
        np.multiply(source[:, 0], np.float64(0.5), out=product)
        np.subtract(source[:, 2], product, out=expected[:, 1])

    compiled()
    numpy_reference()
    seconds = median_seconds(compiled, repeats)
    numpy_seconds = median_seconds(numpy_reference, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    compiled()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = slot_count * int(np.prod(spatial_shape))
    actual_values = destination[:, 1]
    reference_values = expected[:, 1]
    product_values = np.float64(0.5) * source[:, 0]
    local_scale = np.maximum.reduce(
        (
            np.ones_like(actual_values),
            np.abs(source[:, 2]),
            np.abs(product_values),
            np.abs(reference_values),
        )
    )
    result = {
        "slot_count": slot_count,
        "spatial_shape": spatial_shape,
        "seconds": seconds,
        "numpy_seconds": numpy_seconds,
        "million_values_per_second": values / seconds / 1e6,
        "effective_gb_per_second": 24 * values / seconds / 1e9,
        "bitwise_equal": bool(
            np.array_equal(destination.view(np.uint64), expected.view(np.uint64))
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }
    result.update(finite_error_metrics(actual_values, reference_values, local_scale))
    return result


def bounded_composition(capacity: int, repeats: int) -> dict:
    root = i3(32, 16, 8)
    block = i3(8, 8, 8)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    block_count = faces.shape[0]
    x, y, z = np.indices((8, 8, 8), dtype=np.float64)
    backing = np.empty((block_count, 3, 8, 8, 8), dtype=np.float64)
    for block_id in range(block_count):
        backing[block_id, 0] = 1.0 + block_id + x + 2.0 * y + 3.0 * z
        backing[block_id, 1] = 10.0 + backing[block_id, 0]
        backing[block_id, 2] = 30.0 + 2.0 * backing[block_id, 0]
    expected = backing[:, 2] - 0.5 * backing[:, 0]
    result = np.empty((block_count, 1, 8, 8, 8), dtype=np.float64)
    ids = np.empty(capacity, dtype=np.int64)
    source_workspace = np.empty((capacity, 2, 8, 8, 8), dtype=np.float64)
    destination_workspace = np.empty((capacity, 1, 8, 8, 8), dtype=np.float64)
    source_fields = np.array([2, 0], dtype=np.int64)
    destination_field = np.array([0], dtype=np.int64)
    zero = i3(0, 0, 0)

    def traverse() -> int:
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
                block,
                ids[:selected_count],
                source_fields,
                source_workspace[:selected_count],
                zero,
            )
            scaled_difference_into(
                source_workspace[:primary_count],
                zero,
                block,
                0,
                1,
                0.5,
                destination_workspace[:primary_count],
                0,
                zero,
            )
            scatter_blocks_from(
                destination_workspace[:primary_count],
                zero,
                block,
                ids[:primary_count],
                destination_field,
                result,
                zero,
            )
            first += primary_count
            chunks += 1
        return chunks

    chunks = traverse()
    seconds = median_seconds(traverse, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    traverse()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    managed_bytes = (
        ids.nbytes + source_workspace.nbytes + destination_workspace.nbytes
    )
    values = result[:, 0].size
    product_values = np.float64(0.5) * backing[:, 0]
    comparison_scale = np.maximum.reduce(
        (
            np.ones_like(expected),
            np.abs(backing[:, 2]),
            np.abs(product_values),
            np.abs(expected),
        )
    )
    comparison = finite_error_metrics(result[:, 0], expected, comparison_scale)
    report = {
        "capacity": capacity,
        "chunks": chunks,
        "managed_workspace_bytes": managed_bytes,
        "seconds": seconds,
        "million_values_per_second": values / seconds / 1e6,
        "bitwise_equal": bool(
            np.array_equal(result[:, 0].view(np.uint64), expected.view(np.uint64))
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }
    report.update(comparison)
    return report


def whole_array_baseline(repeats: int) -> dict:
    rng = np.random.default_rng(1701)
    source = rng.normal(size=(4096, 3, 8, 8, 8))
    product = np.empty((4096, 8, 8, 8), dtype=np.float64)
    output = np.empty_like(product)

    def compute() -> None:
        np.multiply(source[:, 0], np.float64(0.5), out=product)
        np.subtract(source[:, 2], product, out=output)

    compute()
    seconds = median_seconds(compute, repeats)
    return {
        "seconds": seconds,
        "million_values_per_second": output.size / seconds / 1e6,
        "temporary_bytes": product.nbytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    cases = [
        kernel_case(slots, shape, args.repeats)
        for slots, shape in (
            (16, (8, 8, 8)),
            (128, (8, 8, 8)),
            (16, (16, 16, 16)),
        )
    ]
    print(
        json.dumps(
            {
                "capability": "OPR-001",
                "kernel_cases": cases,
                "bounded_composition": [
                    bounded_composition(capacity, args.repeats)
                    for capacity in (64, 256, 1024)
                ],
                "whole_array_numpy": whole_array_baseline(args.repeats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
