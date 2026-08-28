"""FCL-001 traversal, amplification, compatibility, and allocation probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.chunking import minimum_face_closed_slots, plan_level1_chunk
from simesh_rewrite.face_closure import (
    minimum_direct_face_closed_slots,
    plan_direct_face_closed_prefix,
)
from simesh_rewrite.face_closure_reference import (
    plan_direct_face_closed_prefix_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def traverse(
    face_neighbor_ids: np.ndarray,
    selected_ids: np.ndarray,
    compatibility: bool,
) -> tuple[int, int]:
    first = 0
    chunks = 0
    selected_total = 0
    while first < face_neighbor_ids.shape[0]:
        if compatibility:
            primary_count, selected_count = plan_level1_chunk(
                first,
                face_neighbor_ids,
                True,
                selected_ids,
            )
        else:
            primary_count, selected_count = plan_direct_face_closed_prefix(
                first,
                face_neighbor_ids,
                selected_ids,
            )
        first += primary_count
        selected_total += selected_count
        chunks += 1
    return chunks, selected_total


def measure_capacity(
    face_neighbor_ids: np.ndarray,
    capacity: int,
    repeats: int,
) -> dict:
    selected_ids = np.full(capacity, -1, dtype=np.int64)
    chunks, selected_total = traverse(
        face_neighbor_ids, selected_ids, False
    )
    compatibility_result = traverse(face_neighbor_ids, selected_ids, True)
    if compatibility_result != (chunks, selected_total):
        raise AssertionError("canonical and compatibility traversals diverged")
    canonical_seconds = median_seconds(
        lambda: traverse(face_neighbor_ids, selected_ids, False), repeats
    )
    compatibility_seconds = median_seconds(
        lambda: traverse(face_neighbor_ids, selected_ids, True), repeats
    )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    plan_direct_face_closed_prefix(0, face_neighbor_ids, selected_ids)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    block_count = face_neighbor_ids.shape[0]
    return {
        "capacity": capacity,
        "chunks": chunks,
        "canonical_seconds": canonical_seconds,
        "compatibility_seconds": compatibility_seconds,
        "million_primaries_per_second": block_count / canonical_seconds / 1.0e6,
        "compatibility_million_primaries_per_second": (
            block_count / compatibility_seconds / 1.0e6
        ),
        "selected_ids_per_second": selected_total / canonical_seconds,
        "load_amplification": selected_total / block_count,
        "average_capacity_utilization": selected_total / (chunks * capacity),
        "caller_buffer_bytes": int(selected_ids.nbytes),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def small_reference_probe(repeats: int) -> dict:
    root_shape = np.asarray([8, 6, 4], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    capacity = 64
    selected = np.empty(capacity, dtype=np.int64)
    expected = plan_direct_face_closed_prefix_reference(0, faces, capacity)
    actual = plan_direct_face_closed_prefix(0, faces, selected)
    if actual != expected[1:] or selected[: actual[1]].tolist() != expected[0]:
        raise AssertionError("small FCL reference disagreement")
    return {
        "blocks": int(faces.shape[0]),
        "capacity": capacity,
        "production_seconds": median_seconds(
            lambda: plan_direct_face_closed_prefix(0, faces, selected), repeats
        ),
        "reference_seconds": median_seconds(
            lambda: plan_direct_face_closed_prefix_reference(0, faces, capacity),
            repeats,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", default="64,256,1024")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    root_shape = np.asarray([32, 32, 32], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    minimum = minimum_direct_face_closed_slots(faces)
    if minimum_face_closed_slots(faces) != minimum:
        raise AssertionError("canonical and compatibility minimum disagree")
    minimum_seconds = median_seconds(
        lambda: minimum_direct_face_closed_slots(faces), args.repeats
    )
    compatibility_minimum_seconds = median_seconds(
        lambda: minimum_face_closed_slots(faces), args.repeats
    )
    report = {
        "capability": "FCL-001",
        "root_shape": root_shape.tolist(),
        "blocks": int(faces.shape[0]),
        "minimum_slots": minimum,
        "minimum_seconds": minimum_seconds,
        "minimum_million_blocks_per_second": (
            faces.shape[0] / minimum_seconds / 1.0e6
        ),
        "compatibility_minimum_seconds": compatibility_minimum_seconds,
        "repeats": args.repeats,
        "results": [
            measure_capacity(faces, int(capacity), args.repeats)
            for capacity in args.capacities.split(",")
        ],
        "small_reference": small_reference_probe(args.repeats),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
