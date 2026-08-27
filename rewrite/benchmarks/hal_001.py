"""Retained HAL-001 physical-envelope throughput probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._halos import fill_physical_halos_unchecked
from simesh_rewrite.halos import (
    BoundaryMode,
    common_physical_valid_region,
    fill_physical_halos,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_call(function, arguments, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(*arguments)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure(block_size: int, halo: int, slots: int, all_physical: bool, repeats: int) -> dict:
    fields = 4
    spatial = block_size + 2 * halo
    lower = i3(halo, halo, halo)
    upper = lower + i3(block_size, block_size, block_size)
    payload = np.ones((slots, fields, spatial, spatial, spatial), dtype=np.float64)
    if all_physical:
        block_ids = np.zeros(slots, dtype=np.int64)
        faces = np.full((1, 6), -1, dtype=np.int64)
    else:
        block_ids = np.arange(slots, dtype=np.int64)
        root_shape = i3(slots, 1, 1)
        coord_to_rank, rank_to_coord = level1_morton(root_shape)
        faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    modes = np.array(
        [
            [0, 1, 2, 0, 1, 2],
            [3, 3, 1, 2, 0, 1],
            [1, 2, 3, 3, 2, 0],
            [2, 0, 1, 2, 3, 3],
        ],
        dtype=np.uint8,
    )
    normals = i3(1, 2, 3)
    arguments = (payload, lower, upper, block_ids, faces, modes, normals)
    checked_seconds = median_call(fill_physical_halos, arguments, repeats)
    unchecked_seconds = median_call(
        fill_physical_halos_unchecked,
        arguments,
        repeats,
    )

    produced = 0
    for block_id in block_ids:
        envelope = 1
        for axis in range(3):
            width = block_size
            if faces[block_id, 2 * axis] == -1:
                width += halo
            if faces[block_id, 2 * axis + 1] == -1:
                width += halo
            envelope *= width
        produced += fields * (envelope - block_size**3)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_physical_halos(*arguments)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "block_size": block_size,
        "halo": halo,
        "slots": slots,
        "all_physical": all_physical,
        "produced_values": produced,
        "checked_seconds": checked_seconds,
        "unchecked_seconds": unchecked_seconds,
        "million_produced_values_per_second": produced / checked_seconds / 1.0e6,
        "payload_bytes": payload.nbytes,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def metadata_scaling(slot_count: int) -> dict:
    payload = np.empty((slot_count, 1, 1, 1, 1), dtype=np.float64)
    block_ids = np.zeros(slot_count, dtype=np.int64)
    faces = np.full((1, 6), -1, dtype=np.int64)
    modes = np.zeros((1, 6), dtype=np.uint8)
    lower = i3(0, 0, 0)
    upper = i3(1, 1, 1)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    fill_physical_halos(
        payload,
        lower,
        upper,
        block_ids,
        faces,
        modes,
        i3(-1, -1, -1),
    )
    common = common_physical_valid_region(
        i3(1, 1, 1),
        lower,
        upper,
        block_ids,
        faces,
    )
    seconds = time.perf_counter() - started
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "slots": slot_count,
        "seconds": seconds,
        "common_lower": common[0].tolist(),
        "common_upper": common[1].tolist(),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--metadata-slots", type=int, default=1_000_000)
    args = parser.parse_args()
    cases = [
        (8, 1, 64),
        (8, 2, 64),
        (16, 2, 32),
        (16, 4, 32),
        (32, 2, 8),
    ]
    report = {
        "capability": "HAL-001",
        "repeats": args.repeats,
        "results": [
            measure(block, halo, slots, physical, args.repeats)
            for block, halo, slots in cases
            for physical in (True, False)
        ],
        "metadata_scaling": metadata_scaling(args.metadata_slots),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
