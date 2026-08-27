"""Retained checked runtime and memory probe for TOP-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._topology import (
    fill_level1_face_neighbors_unchecked,
    validate_level1_maps_unchecked,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import fill_level1_face_neighbors
from simesh_rewrite.topology_reference import level1_face_neighbors_reference


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(part) for part in value.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"invalid shape {value!r}")
    return shape


def median_call(function, arguments, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*arguments)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure(shape: tuple[int, int, int], repeats: int, reference_limit: int) -> dict:
    root_shape = np.asarray(shape, dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = np.empty((rank_to_coord.shape[0], 6), dtype=np.int64)
    arguments = (root_shape, coord_to_rank, rank_to_coord)
    fill_level1_face_neighbors(*arguments, neighbors)

    checked_seconds = median_call(
        fill_level1_face_neighbors,
        (*arguments, neighbors),
        repeats,
    )
    validation_seconds = median_call(
        validate_level1_maps_unchecked,
        arguments,
        repeats,
    )
    fill_seconds = median_call(
        fill_level1_face_neighbors_unchecked,
        (*arguments, neighbors),
        repeats,
    )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_level1_face_neighbors(*arguments, neighbors)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    reference_seconds = None
    if rank_to_coord.shape[0] <= reference_limit:
        started = time.perf_counter()
        level1_face_neighbors_reference(*arguments)
        reference_seconds = time.perf_counter() - started

    volume = rank_to_coord.shape[0]
    return {
        "shape": shape,
        "blocks": volume,
        "checked_seconds": checked_seconds,
        "validation_seconds": validation_seconds,
        "fill_seconds": fill_seconds,
        "million_blocks_per_second": volume / checked_seconds / 1.0e6,
        "million_faces_per_second": 6 * volume / checked_seconds / 1.0e6,
        "output_bytes": neighbors.nbytes,
        "full_27_int64_bytes": volume * 27 * np.dtype(np.int64).itemsize,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "reference_seconds": reference_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="16x16x16,32x32x32,64x64x64,31x29x27,127x113x97,65536x1x1",
    )
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--reference-limit", type=int, default=50_000)
    args = parser.parse_args()
    shapes = [parse_shape(value) for value in args.shapes.split(",")]
    report = {
        "capability": "TOP-001",
        "repeats": args.repeats,
        "results": [
            measure(shape, args.repeats, args.reference_limit) for shape in shapes
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
