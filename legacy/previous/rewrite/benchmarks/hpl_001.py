"""Kernel, allocation, and bounded-composition probe for HPL-001."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halo_plans import fill_level1_halo_relation_plan
from simesh_rewrite.halo_plans_reference import (
    level1_halo_relation_plan_reference,
)
from simesh_rewrite.morton import level1_morton
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


def benchmark_case(
    root_shape: tuple[int, int, int],
    capacity: int,
    repeats: int,
) -> dict:
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    first_coord = tuple(int(value // 2) for value in root)
    first = int(coord_to_rank[first_coord])
    ids = np.empty(capacity, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_chunk(first, faces, ids)
    selected = ids[:selected_count]
    source_slots = np.empty((primary_count, 27), dtype=np.int64)
    masks = np.empty((primary_count, 27), dtype=np.uint8)

    def fill() -> None:
        fill_level1_halo_relation_plan(
            selected,
            primary_count,
            faces,
            source_slots,
            masks,
        )

    fill()
    expected = level1_halo_relation_plan_reference(
        selected,
        primary_count,
        faces,
    )
    exact = np.array_equal(source_slots, expected[0]) and np.array_equal(
        masks, expected[1]
    )
    fill_seconds = median_seconds(fill, repeats)

    started = time.perf_counter()
    level1_halo_relation_plan_reference(selected, primary_count, faces)
    reference_seconds = time.perf_counter() - started

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    directions = primary_count * 27
    return {
        "root_shape": root_shape,
        "capacity": capacity,
        "primary_count": primary_count,
        "selected_count": selected_count,
        "directions": directions,
        "fill_seconds": fill_seconds,
        "million_directions_per_second": directions / fill_seconds / 1.0e6,
        "reference_seconds": reference_seconds,
        "reference_speedup": reference_seconds / fill_seconds,
        "output_bytes": source_slots.nbytes + masks.nbytes,
        "formula_bytes": 243 * primary_count,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "exact_reference": bool(exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-shape", default="12,10,8")
    parser.add_argument("--capacities", default="27,64,128,256")
    parser.add_argument("--scaling-sides", default="8,16,32")
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    root_shape = tuple(int(value) for value in args.root_shape.split(","))
    report = {
        "capability": "HPL-001",
        "repeats": args.repeats,
        "cases": [
            benchmark_case(root_shape, int(capacity), args.repeats)
            for capacity in args.capacities.split(",")
        ],
        "fixed_primary_topology_scaling": [
            benchmark_case(
                (int(side), int(side), int(side)),
                27,
                args.repeats,
            )
            for side in args.scaling_sides.split(",")
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
