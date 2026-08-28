"""End-to-end bounded M0 executor measurements for INT-001."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import tempfile
import time
import tracemalloc

from numpy.lib.format import open_memmap
import numpy as np

import simesh_rewrite.pipeline as pipeline_module
from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.pipeline import execute_level1_m0
from simesh_rewrite.primary import fill_ascending_primary_prefix
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


def per_slot_bytes(field_count: int, block: np.ndarray) -> int:
    return 8 * (
        field_count * int(np.prod(block + 2))
        + int(np.prod(block))
        + 1
    )


def make_outputs(
    block_count: int,
    field_count: int,
    block: np.ndarray,
    domain: np.ndarray,
    sample_shape: tuple[int, int, int],
):
    sentinel = np.asarray([0x7FF8000000004321], dtype=np.uint64).view(np.float64)[0]
    return (
        np.full((block_count, 1, *(int(value) for value in block)), sentinel),
        np.full((block_count, 1, *(int(value) for value in block)), sentinel),
        np.full((field_count, *(int(value) for value in domain)), sentinel),
        np.full((field_count, *sample_shape), sentinel),
        np.full((field_count, *sample_shape), sentinel),
    )


def planner_stats(capacity: int, faces: np.ndarray) -> dict:
    ids = np.empty(capacity, dtype=np.int64)
    first = 0
    no_chunks = 0
    while first < faces.shape[0]:
        primary = fill_ascending_primary_prefix(first, faces.shape[0], ids)
        first += primary
        no_chunks += 1

    first = 0
    halo_chunks = 0
    selected_total = 0
    while first < faces.shape[0]:
        primary, selected = plan_level1_halo_chunk(first, faces, ids)
        first += primary
        selected_total += selected
        halo_chunks += 1
    return {
        "no_closure_chunks": no_chunks,
        "full_halo_chunks": halo_chunks,
        "full_halo_load_amplification": selected_total / faces.shape[0],
    }


def profile_call(call) -> tuple[tuple[float, int], float, dict]:
    names = (
        "fill_ascending_primary_prefix",
        "plan_level1_halo_chunk",
        "read_blocks_into",
        "scaled_difference_into",
        "write_blocks_from",
        "place_level1_blocks",
        "sample_level1_zero_order",
        "accumulate_field_sum",
        "fill_physical_halos",
        "fill_same_level_halos",
        "central_difference_into",
        "sample_level1_trilinear",
    )
    originals = {name: getattr(pipeline_module, name) for name in names}
    phases = ("preflight", "pass_one", "pass_two")
    totals = {
        phase: {name: 0.0 for name in names}
        for phase in phases
    }
    state = {
        "phase": "preflight",
        "dry_halo_complete": False,
        "pass_one_start": None,
        "pass_two_start": None,
    }

    def wrapped(name):
        original = originals[name]

        def operation(*args, **kwargs):
            if (
                name == "fill_ascending_primary_prefix"
                and state["phase"] == "preflight"
                and state["dry_halo_complete"]
                and int(args[0]) == 0
            ):
                state["phase"] = "pass_one"
                state["pass_one_start"] = time.perf_counter()
            elif (
                name == "plan_level1_halo_chunk"
                and state["phase"] == "pass_one"
                and int(args[0]) == 0
            ):
                state["phase"] = "pass_two"
                state["pass_two_start"] = time.perf_counter()
            active_phase = state["phase"]
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                totals[active_phase][name] += time.perf_counter() - started
            if (
                name == "plan_level1_halo_chunk"
                and active_phase == "preflight"
                and int(args[0]) + int(result[0]) >= args[1].shape[0]
            ):
                state["dry_halo_complete"] = True
            return result

        return operation

    for name in names:
        setattr(pipeline_module, name, wrapped(name))
    call_start = time.perf_counter()
    try:
        result = call()
    finally:
        call_end = time.perf_counter()
        elapsed = call_end - call_start
        for name, original in originals.items():
            setattr(pipeline_module, name, original)
    pass_one_start = state["pass_one_start"]
    pass_two_start = state["pass_two_start"]
    if pass_one_start is None or pass_two_start is None:
        raise RuntimeError("pipeline profiling did not observe both execution passes")
    phase_seconds = {
        "preflight": pass_one_start - call_start,
        "pass_one": pass_two_start - pass_one_start,
        "pass_two": call_end - pass_two_start,
    }
    for phase in phases:
        totals[phase]["unattributed_python"] = max(
            0.0,
            phase_seconds[phase] - sum(totals[phase].values()),
        )
    return result, elapsed, {
        "phase_seconds": phase_seconds,
        "stage_seconds_by_phase": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    root = i3(8, 6, 4)
    block = i3(8, 8, 8)
    domain = root * block
    block_count = int(np.prod(root))
    backing_fields = 6
    field_ids = np.array([4, 1, 5], dtype=np.int64)
    domain_lower = np.zeros(3)
    domain_upper = np.ones(3)
    sample_lower = np.array([0.11, 0.17, 0.19])
    sample_upper = np.array([0.89, 0.83, 0.81])
    sample_shape = (48, 36, 24)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    modes = np.zeros((len(field_ids), 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    bytes_per_slot = per_slot_bytes(len(field_ids), block)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "m0_source.npy")
        writable = open_memmap(
            path,
            mode="w+",
            dtype=np.float64,
            shape=(block_count, backing_fields, 8, 8, 8),
        )
        local = np.indices((8, 8, 8), dtype=np.float64)
        spacing = 1.0 / domain
        for block_id, coordinate in enumerate(rank_to_coord):
            global_index = coordinate[:, None, None, None] * block[:, None, None, None] + local
            centers = (global_index + 0.5) * spacing[:, None, None, None]
            base = centers[0] + 2.0 * centers[1] + 3.0 * centers[2]
            for field in range(backing_fields):
                writable[block_id, field] = 10.0 * field + (field + 1.0) * base
        writable.flush()
        del writable
        backing = np.load(path, mmap_mode="r")
        source_edges = (
            backing[0, :, 0, 0, 0].copy(),
            backing[-1, :, -1, -1, -1].copy(),
        )

        capacities = (block_count, 27, 64, 128)
        reference_outputs = None
        reference_reduction = None
        reports = []
        for capacity in capacities:
            outputs = make_outputs(
                block_count,
                len(field_ids),
                block,
                domain,
                sample_shape,
            )
            budget = 8 + capacity * bytes_per_slot

            def call():
                return execute_level1_m0(
                    backing,
                    field_ids,
                    domain_lower,
                    domain_upper,
                    domain,
                    block,
                    coord_to_rank,
                    rank_to_coord,
                    faces,
                    modes,
                    normals,
                    0,
                    1,
                    0.5,
                    2,
                    0,
                    1,
                    sample_lower,
                    sample_upper,
                    budget,
                    *outputs,
                )

            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.perf_counter()
            cold_result = call()
            cold_seconds = time.perf_counter() - started
            warm_seconds = median_seconds(call, args.repeats)
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            tracemalloc.start()
            before_current, _ = tracemalloc.get_traced_memory()
            result = call()
            after_current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _, profiled_seconds, stage_seconds = profile_call(call)

            if reference_outputs is None:
                reference_outputs = tuple(value.copy() for value in outputs)
                reference_reduction = result[0]
            comparisons = [
                bool(
                    np.array_equal(
                        value.view(np.uint64),
                        reference.view(np.uint64),
                    )
                )
                for value, reference in zip(
                    outputs,
                    reference_outputs,
                    strict=True,
                )
            ]
            managed = 8 + result[1] * bytes_per_slot
            report = {
                "requested_capacity": capacity,
                "returned_capacity": result[1],
                "budget_bytes": budget,
                "managed_workspace_bytes": managed,
                "mapped_source_bytes": backing.nbytes,
                "final_output_bytes": sum(value.nbytes for value in outputs),
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "profiled_seconds": profiled_seconds,
                "stage_seconds": stage_seconds,
                "result_arrays_bitwise_equal": comparisons,
                "reduction_bitwise_equal": bool(
                    np.asarray([result[0]]).view(np.uint64)[0]
                    == np.asarray([reference_reduction]).view(np.uint64)[0]
                ),
                "reduction": result[0],
                "traced_current_delta_bytes": after_current - before_current,
                "traced_peak_delta_bytes": peak - before_current,
                "minor_page_fault_delta": usage_after.ru_minflt - usage_before.ru_minflt,
                "major_page_fault_delta": usage_after.ru_majflt - usage_before.ru_majflt,
                "process_max_rss_raw": usage_after.ru_maxrss,
            }
            report.update(planner_stats(result[1], faces))
            reports.append(report)

        source_preserved = bool(
            np.array_equal(backing[0, :, 0, 0, 0].view(np.uint64), source_edges[0].view(np.uint64))
            and np.array_equal(
                backing[-1, :, -1, -1, -1].view(np.uint64),
                source_edges[1].view(np.uint64),
            )
        )

    print(
        json.dumps(
            {
                "capability": "INT-001",
                "root_shape": root.tolist(),
                "block_shape": block.tolist(),
                "field_ids": field_ids.tolist(),
                "bytes_per_slot": bytes_per_slot,
                "source_preserved": source_preserved,
                "runs": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
