"""Retained runtime, throughput, and memory probe for MOR-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.morton import fill_level1_morton
from simesh_rewrite.morton_reference import level1_morton_reference


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(part) for part in value.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"invalid shape {value!r}")
    return shape


def measure(shape: tuple[int, int, int], repeats: int, reference_limit: int) -> dict:
    root_shape = np.asarray(shape, dtype=np.int64)
    volume = int(np.prod(root_shape, dtype=np.int64))
    coord_to_rank = np.empty(shape, dtype=np.int64)
    rank_to_coord = np.empty((volume, 3), dtype=np.int64)
    fill_level1_morton(root_shape, coord_to_rank, rank_to_coord)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fill_level1_morton(root_shape, coord_to_rank, rank_to_coord)
        samples.append(time.perf_counter() - started)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_level1_morton(root_shape, coord_to_rank, rank_to_coord)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    reference_seconds = None
    if volume <= reference_limit:
        started = time.perf_counter()
        level1_morton_reference(root_shape)
        reference_seconds = time.perf_counter() - started

    median_seconds = statistics.median(samples)
    return {
        "shape": shape,
        "coordinates": volume,
        "median_seconds": median_seconds,
        "million_coordinates_per_second": volume / median_seconds / 1.0e6,
        "output_bytes": coord_to_rank.nbytes + rank_to_coord.nbytes,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "reference_seconds": reference_seconds,
    }


def measure_current(shape: tuple[int, int, int], repeats: int) -> dict:
    from simesh.utils.lib.amr.morton import fill_morton_mapping3D

    volume = int(np.prod(shape))
    coord_to_rank = np.empty(shape, dtype=np.uint32)
    rank_to_coord = np.empty((volume, 3), dtype=np.uint32)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fill_morton_mapping3D(coord_to_rank, rank_to_coord, *shape)
        samples.append(time.perf_counter() - started)
    return {
        "shape": shape,
        "coordinates": volume,
        "median_seconds": statistics.median(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default=(
            "16x16x16,32x32x32,33x33x33,64x64x64,"
            "31x29x27,127x113x97,65536x1x1"
        ),
    )
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--reference-limit", type=int, default=50_000)
    parser.add_argument("--current-shapes", default="")
    parser.add_argument("--current-repeats", type=int, default=3)
    args = parser.parse_args()
    shapes = [parse_shape(value) for value in args.shapes.split(",")]
    report = {
        "capability": "MOR-001",
        "repeats": args.repeats,
        "results": [
            measure(shape, args.repeats, args.reference_limit) for shape in shapes
        ],
        "current_results": [
            measure_current(shape, args.current_repeats)
            for shape in (
                parse_shape(value)
                for value in args.current_shapes.split(",")
                if value
            )
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
