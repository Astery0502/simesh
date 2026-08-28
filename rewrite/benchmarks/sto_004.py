"""STO-004 refined support planning and bounded REL-window composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._refined_support import (
    plan_balanced_refined_support_prefix_unchecked,
)
from simesh_rewrite._primary import fill_ascending_primary_prefix_unchecked
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_support import (
    maximum_balanced_refined_support_slots,
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relations import (
    balanced_refined_relations,
    fill_balanced_refined_relations,
)
from simesh_rewrite.storage import gather_blocks_into


ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def conformance_args(root_shape, coord_to_rank, rank_to_coord, forest):
    return (
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def relation_forest_args(root_shape, coord_to_rank, forest):
    return (
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def relation_arrays(capacity: int, direction_count: int):
    shape = (capacity, direction_count)
    return (
        np.empty(shape, dtype=np.uint8),
        np.empty(shape, dtype=np.uint8),
        np.empty(shape, dtype=np.uint8),
        np.empty((*shape, 4), dtype=np.int64),
    )


def traverse_precomputed(
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    capacity: int,
    checked: bool,
) -> tuple[int, int, int]:
    leaf_count = source_counts.shape[0]
    selected = np.empty(capacity, dtype=np.int64)
    first = 0
    chunks = 0
    selected_total = 0
    while first < leaf_count:
        candidate_count = min(capacity, leaf_count - first)
        counts = source_counts[first : first + candidate_count]
        sources = source_leaf_ids[first : first + candidate_count]
        if checked:
            primary_count, selected_count = plan_balanced_refined_support_prefix(
                first,
                leaf_count,
                counts,
                sources,
                selected,
            )
        else:
            primary_count, selected_count = (
                plan_balanced_refined_support_prefix_unchecked(
                    first,
                    counts,
                    sources,
                    selected,
                )
            )
        first += primary_count
        chunks += 1
        selected_total += selected_count
    return chunks, selected_total, leaf_count


def measure_precomputed(
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    capacity: int,
    repeats: int,
) -> dict:
    checked_result = traverse_precomputed(
        source_counts, source_leaf_ids, capacity, True
    )
    unchecked_result = traverse_precomputed(
        source_counts, source_leaf_ids, capacity, False
    )
    if checked_result != unchecked_result:
        raise AssertionError("checked and unchecked refined support plans diverged")
    checked_seconds = median_seconds(
        lambda: traverse_precomputed(
            source_counts, source_leaf_ids, capacity, True
        ),
        repeats,
    )
    unchecked_seconds = median_seconds(
        lambda: traverse_precomputed(
            source_counts, source_leaf_ids, capacity, False
        ),
        repeats,
    )
    chunks, selected_total, leaf_count = checked_result
    selected = np.empty(capacity, dtype=np.int64)
    first_count = min(capacity, leaf_count)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    plan_balanced_refined_support_prefix(
        0,
        leaf_count,
        source_counts[:first_count],
        source_leaf_ids[:first_count],
        selected,
    )
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "capacity": capacity,
        "chunks": chunks,
        "checked_seconds": checked_seconds,
        "unchecked_seconds": unchecked_seconds,
        "million_primaries_per_second": leaf_count / checked_seconds / 1.0e6,
        "load_amplification": selected_total / leaf_count,
        "average_capacity_utilization": selected_total / (chunks * capacity),
        "selected_id_bytes": int(selected.nbytes),
        "planner_relation_input_bytes": int(
            source_counts.nbytes + source_leaf_ids.nbytes
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }
    return result


def fill_relation_rows(
    relation_base,
    directions,
    first_leaf: int,
    count: int,
    candidate_ids: np.ndarray,
    outputs,
    offset: int,
) -> None:
    if count == 0:
        return
    filled = fill_ascending_primary_prefix_unchecked(
        first_leaf,
        first_leaf + count,
        candidate_ids[:count],
    )
    if filled != count:
        raise AssertionError("PRI did not fill the requested relation rows")
    fill_balanced_refined_relations(
        *relation_base,
        candidate_ids[:count],
        directions,
        *(output[offset : offset + count] for output in outputs),
    )


def append_plan_trace(
    trace: list,
    first: int,
    primary_count: int,
    selected_count: int,
    selected: np.ndarray,
    outputs,
) -> None:
    trace.append(
        (
            first,
            primary_count,
            selected_count,
            selected[:selected_count].copy(),
            tuple(output[:primary_count].copy() for output in outputs),
        )
    )


def assert_exact_trace_equal(left: list, right: list) -> None:
    if len(left) != len(right):
        raise AssertionError("naive and sliding trace lengths diverged")
    for left_chunk, right_chunk in zip(left, right):
        if left_chunk[:3] != right_chunk[:3]:
            raise AssertionError("naive and sliding chunk counts diverged")
        if not np.array_equal(left_chunk[3], right_chunk[3]):
            raise AssertionError("naive and sliding selected plans diverged")
        if any(
            not np.array_equal(left_output, right_output)
            for left_output, right_output in zip(
                left_chunk[4], right_chunk[4]
            )
        ):
            raise AssertionError("naive and sliding accepted relations diverged")


def sliding_composition(
    relation_base,
    leaf_count: int,
    capacity: int,
    directions: np.ndarray,
    with_gather: bool,
    record_trace: bool = False,
) -> dict:
    selected = np.empty(capacity, dtype=np.int64)
    outputs = relation_arrays(capacity, directions.shape[0])
    backing = None
    payload = None
    if with_gather:
        backing = np.arange(leaf_count, dtype=np.float64).reshape(
            leaf_count, 1, 1, 1, 1
        )
        payload = np.empty((capacity, 1, 1, 1, 1), dtype=np.float64)
    zero = np.asarray([0, 0, 0], dtype=np.int64)
    one = np.asarray([1, 1, 1], dtype=np.int64)
    field = np.asarray([0], dtype=np.int64)

    first = 0
    next_leaf = 0
    window_count = min(capacity, leaf_count)
    relation_seconds = 0.0
    planner_seconds = 0.0
    compaction_seconds = 0.0
    gather_seconds = 0.0
    compaction_bytes = 0
    selected_total = 0
    chunks = 0
    primary_sum = 0.0
    trace = []

    wall_started = time.perf_counter()
    started = wall_started
    fill_relation_rows(
        relation_base,
        directions,
        next_leaf,
        window_count,
        selected,
        outputs,
        0,
    )
    relation_seconds += time.perf_counter() - started
    next_leaf += window_count

    while first < leaf_count:
        started = time.perf_counter()
        primary_count, selected_count = (
            plan_balanced_refined_support_prefix_unchecked(
                first,
                outputs[2][:window_count],
                outputs[3][:window_count],
                selected,
            )
        )
        planner_seconds += time.perf_counter() - started
        if record_trace:
            append_plan_trace(
                trace,
                first,
                primary_count,
                selected_count,
                selected,
                outputs,
            )

        if with_gather:
            started = time.perf_counter()
            gather_blocks_into(
                backing,
                zero,
                one,
                selected[:selected_count],
                field,
                payload[:selected_count],
                zero,
            )
            primary_sum += float(np.sum(payload[:primary_count]))
            gather_seconds += time.perf_counter() - started

        first += primary_count
        selected_total += selected_count
        chunks += 1
        remaining = window_count - primary_count

        started = time.perf_counter()
        if remaining:
            for output in outputs:
                output[:remaining] = output[
                    primary_count:window_count
                ]
            compaction_bytes += remaining * 35 * directions.shape[0]
        compaction_seconds += time.perf_counter() - started

        refill = min(primary_count, leaf_count - next_leaf)
        started = time.perf_counter()
        fill_relation_rows(
            relation_base,
            directions,
            next_leaf,
            refill,
            selected,
            outputs,
            remaining,
        )
        relation_seconds += time.perf_counter() - started
        next_leaf += refill
        window_count = remaining + refill

    if next_leaf != leaf_count:
        raise AssertionError("sliding relation rows were not generated exactly once")
    if with_gather and primary_sum != leaf_count * (leaf_count - 1) / 2:
        raise AssertionError("sliding gather did not consume every primary once")
    measured_stage_seconds = (
        relation_seconds
        + planner_seconds
        + compaction_seconds
        + gather_seconds
    )
    result = {
        "chunks": chunks,
        "selected_total": selected_total,
        "load_amplification": selected_total / leaf_count,
        "rows_generated": next_leaf,
        "row_amplification": next_leaf / leaf_count,
        "relation_seconds": relation_seconds,
        "planner_seconds": planner_seconds,
        "compaction_seconds": compaction_seconds,
        "gather_seconds": gather_seconds,
        "measured_stage_seconds": measured_stage_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
        "compaction_bytes": compaction_bytes,
        "metadata_bytes": int(
            sum(output.nbytes for output in outputs) + selected.nbytes
        ),
        "gather_workspace_bytes": int(payload.nbytes) if with_gather else 0,
    }
    if record_trace:
        result["trace"] = trace
    return result


def naive_composition(
    relation_base,
    leaf_count: int,
    capacity: int,
    directions: np.ndarray,
    with_gather: bool,
    record_trace: bool = False,
) -> dict:
    selected = np.empty(capacity, dtype=np.int64)
    outputs = relation_arrays(capacity, directions.shape[0])
    backing = None
    payload = None
    if with_gather:
        backing = np.arange(leaf_count, dtype=np.float64).reshape(
            leaf_count, 1, 1, 1, 1
        )
        payload = np.empty((capacity, 1, 1, 1, 1), dtype=np.float64)
    zero = np.asarray([0, 0, 0], dtype=np.int64)
    one = np.asarray([1, 1, 1], dtype=np.int64)
    field = np.asarray([0], dtype=np.int64)
    first = 0
    rows_generated = 0
    chunks = 0
    selected_total = 0
    relation_seconds = 0.0
    planner_seconds = 0.0
    gather_seconds = 0.0
    primary_sum = 0.0
    trace = []
    wall_started = time.perf_counter()
    while first < leaf_count:
        candidate_count = min(capacity, leaf_count - first)
        started = time.perf_counter()
        fill_relation_rows(
            relation_base,
            directions,
            first,
            candidate_count,
            selected,
            outputs,
            0,
        )
        relation_seconds += time.perf_counter() - started
        started = time.perf_counter()
        primary_count, selected_count = (
            plan_balanced_refined_support_prefix_unchecked(
                first,
                outputs[2][:candidate_count],
                outputs[3][:candidate_count],
                selected,
            )
        )
        planner_seconds += time.perf_counter() - started
        if record_trace:
            append_plan_trace(
                trace,
                first,
                primary_count,
                selected_count,
                selected,
                outputs,
            )
        if with_gather:
            started = time.perf_counter()
            gather_blocks_into(
                backing,
                zero,
                one,
                selected[:selected_count],
                field,
                payload[:selected_count],
                zero,
            )
            primary_sum += float(np.sum(payload[:primary_count]))
            gather_seconds += time.perf_counter() - started
        first += primary_count
        rows_generated += candidate_count
        chunks += 1
        selected_total += selected_count
    if with_gather and primary_sum != leaf_count * (leaf_count - 1) / 2:
        raise AssertionError("naive gather did not consume every primary once")
    result = {
        "wall_seconds": time.perf_counter() - wall_started,
        "chunks": chunks,
        "selected_total": selected_total,
        "load_amplification": selected_total / leaf_count,
        "rows_generated": rows_generated,
        "row_amplification": rows_generated / leaf_count,
        "relation_seconds": relation_seconds,
        "planner_seconds": planner_seconds,
        "gather_seconds": gather_seconds,
        "metadata_bytes": int(
            sum(output.nbytes for output in outputs) + selected.nbytes
        ),
        "gather_workspace_bytes": int(payload.nbytes) if with_gather else 0,
    }
    if record_trace:
        result["trace"] = trace
    return result


def measure_dat(
    path: Path,
    capacities: list[int],
    repeats: int,
) -> dict:
    from simesh.amrvac.datio import get_metadata
    from simesh.utils.lib.amr.forest import AMRForest

    header, flags_input, _ = get_metadata(str(path))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags_input, dtype=np.bool_),
    )
    validate_refined_forest_arrays(
        *conformance_args(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    relation_base = relation_forest_args(root_shape, coord_to_rank, forest)
    validate_refined_all_touch_2to1(*relation_base)
    leaf_count = forest.leaf_node_ids.size
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    full_relation_seconds = median_seconds(
        lambda: balanced_refined_relations(
            *relation_base, leaf_ids, ALL_DIRECTIONS
        ),
        repeats,
    )
    full_outputs = balanced_refined_relations(
        *relation_base, leaf_ids, ALL_DIRECTIONS
    )
    exact_maximum = maximum_balanced_refined_support_slots(
        0,
        leaf_count,
        full_outputs[2],
        full_outputs[3],
    )
    maximum_seconds = median_seconds(
        lambda: maximum_balanced_refined_support_slots(
            0,
            leaf_count,
            full_outputs[2],
            full_outputs[3],
        ),
        repeats,
    )
    planner_results = [
        measure_precomputed(
            full_outputs[2], full_outputs[3], capacity, repeats
        )
        for capacity in capacities
    ]
    full_relation_bytes = int(sum(output.nbytes for output in full_outputs))
    del full_outputs
    del leaf_ids

    exact_trace_leaf_count = min(512, leaf_count)
    sliding_results = []
    naive_results = []
    for capacity in capacities:
        sliding_trace = sliding_composition(
            relation_base,
            exact_trace_leaf_count,
            capacity,
            ALL_DIRECTIONS,
            False,
            True,
        )
        naive_trace = naive_composition(
            relation_base,
            exact_trace_leaf_count,
            capacity,
            ALL_DIRECTIONS,
            False,
            True,
        )
        assert_exact_trace_equal(
            naive_trace.pop("trace"), sliding_trace.pop("trace")
        )
        sliding_samples = [
            sliding_composition(
                relation_base,
                leaf_count,
                capacity,
                ALL_DIRECTIONS,
                True,
            )
            for _ in range(repeats)
        ]
        median_wall = statistics.median(
            sample["wall_seconds"] for sample in sliding_samples
        )
        sliding = min(
            sliding_samples,
            key=lambda sample: abs(sample["wall_seconds"] - median_wall),
        )
        naive_samples = [
            naive_composition(
                relation_base,
                leaf_count,
                capacity,
                ALL_DIRECTIONS,
                True,
            )
            for _ in range(repeats)
        ]
        median_naive = statistics.median(
            sample["wall_seconds"] for sample in naive_samples
        )
        naive = min(
            naive_samples,
            key=lambda sample: abs(sample["wall_seconds"] - median_naive),
        )
        comparable_fields = ("chunks", "selected_total")
        if any(sliding[field] != naive[field] for field in comparable_fields):
            raise AssertionError("naive and sliding compositions diverged")
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        sliding_composition(
            relation_base,
            leaf_count,
            capacity,
            ALL_DIRECTIONS,
            True,
        )
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sliding["traced_current_delta_bytes"] = (
            after_current - before_current
        )
        sliding["traced_peak_delta_bytes"] = peak - before_current
        sliding_results.append(sliding | {"capacity": capacity})
        naive_results.append(naive | {"capacity": capacity})

    current = AMRForest(
        3,
        *tuple(int(value) for value in root_shape),
        np.ascontiguousarray(flags_input, dtype=np.int32),
    )
    current_retained_bytes = int(
        np.asarray(current.neighbor_type).nbytes
        + np.asarray(current.neighbor_index).nbytes
        + np.asarray(current.neighbor_children).nbytes
    )
    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "nodes": int(forest.node_levels.size),
        "leaves": int(leaf_count),
        "max_level": int(forest.max_level),
        "directions": int(ALL_DIRECTIONS.shape[0]),
        "exact_maximum_slots": exact_maximum,
        "universal_maximum_slots": 57,
        "maximum_seconds": maximum_seconds,
        "full_relation_seconds": full_relation_seconds,
        "full_relation_bytes": full_relation_bytes,
        "current_retained_connectivity_bytes": current_retained_bytes,
        "exact_trace_leaf_count": exact_trace_leaf_count,
        "exact_trace_capacities": capacities,
        "planner_results": planner_results,
        "sliding_results": sliding_results,
        "naive_results": naive_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", default="57,64,128,256,1024")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dat", type=Path)
    args = parser.parse_args()
    capacities = [int(value) for value in args.capacities.split(",")]
    report = {
        "capability": "STO-004",
        "repeats": args.repeats,
        "real_dat": (
            measure_dat(args.dat, capacities, args.repeats)
            if args.dat
            else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
