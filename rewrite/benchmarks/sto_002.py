"""Retained planner scaling and memmap streaming probe for STO-002."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import tempfile
import time
import tracemalloc

from numpy.lib.format import open_memmap
import numpy as np

from simesh_rewrite.chunking import (
    plan_level1_chunk,
    plan_level1_halo_chunk,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.face_closure import plan_direct_face_closed_prefix
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors
from simesh_rewrite.workspace import workspace_nbytes, workspace_slot_capacity


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def plan_full_traversal(
    neighbors: np.ndarray,
    capacity: int,
    closure: str,
) -> dict:
    ids = np.empty(capacity, dtype=np.int64)
    first = 0
    chunks = 0
    selected_total = 0
    started = time.perf_counter()
    while first < neighbors.shape[0]:
        if closure == "halo":
            primary_count, selected_count = plan_level1_halo_chunk(
                first,
                neighbors,
                ids,
            )
        elif closure == "faces":
            primary_count, selected_count = plan_direct_face_closed_prefix(
                first,
                neighbors,
                ids,
            )
        else:
            primary_count, selected_count = plan_level1_chunk(
                first,
                neighbors,
                False,
                ids,
            )
        first += primary_count
        selected_total += selected_count
        chunks += 1
    seconds = time.perf_counter() - started
    return {
        "capacity": capacity,
        "closure": closure,
        "chunks": chunks,
        "seconds": seconds,
        "million_primaries_per_second": neighbors.shape[0] / seconds / 1.0e6,
        "load_amplification": selected_total / neighbors.shape[0],
        "average_capacity_utilization": selected_total / (chunks * capacity),
    }


def memmap_stream(block_count: int, budget_bytes: int) -> dict:
    field_count = 2
    block_shape = i3(16, 16, 16)
    capacity = workspace_slot_capacity(
        budget_bytes,
        block_count,
        field_count,
        block_shape,
    )
    managed_bytes = workspace_nbytes(capacity, field_count, block_shape)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "blocks.npy")
        writable = open_memmap(
            path,
            mode="w+",
            dtype=np.float64,
            shape=(
                block_count,
                field_count,
                *(int(value) for value in block_shape),
            ),
        )
        for block_id in range(block_count):
            writable[block_id].fill(float(block_id))
        writable.flush()
        del writable
        backing = np.load(path, mmap_mode="r")
        ids = np.empty(capacity, dtype=np.int64)
        payload = np.empty(
            (capacity, field_count, *(int(value) for value in block_shape))
        )
        faces = np.full((block_count, 6), -1, dtype=np.int64)
        fields = np.arange(field_count, dtype=np.int64)
        lower = i3(0, 0, 0)
        first = 0
        total = 0.0
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        started = time.perf_counter()
        while first < block_count:
            primary_count, selected_count = plan_level1_chunk(
                first,
                faces,
                False,
                ids,
            )
            gather_blocks_into(
                backing,
                lower,
                block_shape,
                ids[:selected_count],
                fields,
                payload[:selected_count],
                lower,
            )
            total += float(np.sum(payload[:primary_count]))
            first += primary_count
        seconds = time.perf_counter() - started
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        file_payload_bytes = backing.nbytes
    return {
        "block_count": block_count,
        "budget_bytes": budget_bytes,
        "capacity": capacity,
        "managed_bytes": managed_bytes,
        "file_payload_bytes": file_payload_bytes,
        "seconds": seconds,
        "payload_gb_per_second": file_payload_bytes / seconds / 1.0e9,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "minor_page_fault_delta": usage_after.ru_minflt - usage_before.ru_minflt,
        "major_page_fault_delta": usage_after.ru_majflt - usage_before.ru_majflt,
        "process_max_rss_raw": usage_after.ru_maxrss,
        "process_max_rss_platform": platform.system(),
        "cache_state": "warm_after_file_creation",
        "sum": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", default="64,256,1024")
    parser.add_argument("--memmap-blocks", type=int, default=2048)
    parser.add_argument("--budget-mib", type=int, default=2)
    args = parser.parse_args()

    root_shape = i3(32, 32, 32)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    capacities = [int(value) for value in args.capacities.split(",")]
    planner_results = []
    for capacity in capacities:
        for closure in ("none", "faces", "halo"):
            samples = [
                plan_full_traversal(neighbors, capacity, closure)
                for _ in range(5)
            ]
            median = statistics.median(sample["seconds"] for sample in samples)
            chosen = min(samples, key=lambda sample: abs(sample["seconds"] - median))
            planner_results.append(chosen)

    report = {
        "capability": "STO-002",
        "root_shape": root_shape.tolist(),
        "planner_results": planner_results,
        "memmap": memmap_stream(
            args.memmap_blocks,
            args.budget_mib * 1024 * 1024,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
