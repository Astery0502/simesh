"""RSL-001 synthetic scaling, WENO bounded, and repeated-consumer evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from simesh_rewrite._relation_slots import (
    count_relation_slot_comparisons_unchecked,
    resolve_refined_relation_source_slots_unchecked,
)
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_support import (
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relation_slots import (
    resolve_refined_relation_source_slots,
)
from simesh_rewrite.relation_slots_reference import (
    resolve_refined_relation_source_slots_reference,
)
from simesh_rewrite.relations import fill_balanced_refined_relations


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


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right))


def interleaved_timings(operations, repeats: int):
    samples = {name: [] for name, _ in operations}
    orders = []
    for repeat in range(repeats):
        order = [
            operations[(repeat + offset) % len(operations)]
            for offset in range(len(operations))
        ]
        orders.append([name for name, _ in order])
        for name, operation in order:
            started = time.perf_counter()
            operation()
            samples[name].append(time.perf_counter() - started)
    return (
        {name: statistics.median(values) for name, values in samples.items()},
        orders,
    )


def position_for(
    mode: str,
    selected_count: int,
    primary: int,
    direction: int,
    source: int,
) -> int:
    if mode == "first":
        return min(source, selected_count - 1)
    if mode == "middle":
        return min(selected_count - 1, selected_count // 2 + source)
    if mode == "last":
        return max(0, selected_count - 1 - source)
    if mode == "mixed":
        return (
            104729 * primary + 1009 * direction + 17 * source + 13
        ) % selected_count
    raise ValueError(f"unknown lookup position mode {mode}")


def synthetic_inputs(
    selected_count: int,
    primary_count: int,
    direction_count: int,
    count_mode: str | int,
    position_mode: str,
):
    if primary_count > selected_count:
        raise ValueError("primary_count exceeds selected_count")
    selected = np.arange(selected_count - 1, -1, -1, dtype=np.int64)
    counts = np.empty((primary_count, direction_count), dtype=np.uint8)
    sources = np.full(
        (primary_count, direction_count, 4), -1, dtype=np.int64
    )
    for primary in range(primary_count):
        for direction in range(direction_count):
            count = (
                (primary + direction) % 5
                if count_mode == "mixed"
                else int(count_mode)
            )
            counts[primary, direction] = count
            for source in range(count):
                position = position_for(
                    position_mode,
                    selected_count,
                    primary,
                    direction,
                    source,
                )
                sources[primary, direction, source] = selected[position]
    return selected, counts, sources


def synthetic_case(
    label: str,
    selected_count: int,
    primary_count: int,
    direction_count: int,
    count_mode: str | int,
    position_mode: str,
    repeats: int,
    *,
    trace_checked: bool = False,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    selected, counts, sources = synthetic_inputs(
        selected_count,
        primary_count,
        direction_count,
        count_mode,
        position_mode,
    )
    checked_output = np.empty_like(sources)
    unchecked_output = np.empty_like(sources)
    reference_output = np.empty_like(sources)
    leaf_count = selected_count

    def checked() -> None:
        resolve_refined_relation_source_slots(
            leaf_count,
            selected,
            counts,
            sources,
            checked_output,
        )

    def unchecked() -> None:
        resolve_refined_relation_source_slots_unchecked(
            selected,
            counts,
            sources,
            unchecked_output,
        )

    def reference() -> None:
        resolve_refined_relation_source_slots_reference(
            leaf_count,
            selected,
            counts,
            sources,
            reference_output,
        )

    checked()
    unchecked()
    reference()
    exact_unchecked = arrays_equal(checked_output, unchecked_output)
    exact_reference = arrays_equal(checked_output, reference_output)
    if not exact_unchecked or not exact_reference:
        raise AssertionError("synthetic RSL output arrays disagree")

    operations = [
        ("checked", checked),
        ("unchecked", unchecked),
        ("list_reference", reference),
    ]
    timings, timing_orders = interleaved_timings(operations, repeats)
    if not arrays_equal(checked_output, unchecked_output) or not arrays_equal(
        checked_output, reference_output
    ):
        raise AssertionError("timed synthetic RSL arrays disagree")

    active_lookups = int(np.sum(counts, dtype=np.int64))
    tensor_entries = int(primary_count * direction_count * 4)
    one_pass_comparisons = int(
        count_relation_slot_comparisons_unchecked(selected, counts, sources)
    )
    uniqueness_comparisons = selected_count * (selected_count - 1) // 2
    traced_retained = None
    traced_peak = None
    if trace_checked:
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        checked()
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_retained = after_current - before_current
        traced_peak = peak - before_current

    result = {
        "label": label,
        "selected_count": selected_count,
        "accepted_primary_count": primary_count,
        "direction_count": direction_count,
        "count_mode": count_mode,
        "lookup_position_mode": position_mode,
        "active_lookups": active_lookups,
        "tensor_entries": tensor_entries,
        "one_lookup_pass_comparisons": one_pass_comparisons,
        "comparisons_per_active_per_pass": (
            0.0
            if active_lookups == 0
            else one_pass_comparisons / active_lookups
        ),
        "checked_selected_uniqueness_comparisons": uniqueness_comparisons,
        "checked_two_lookup_pass_comparisons": 2 * one_pass_comparisons,
        "checked_total_identity_comparisons": (
            uniqueness_comparisons + 2 * one_pass_comparisons
        ),
        "source_slot_output_bytes": checked_output.nbytes,
        "exact_checked_unchecked_arrays": exact_unchecked,
        "exact_checked_list_reference_arrays": exact_reference,
        "timing_orders": timing_orders,
        "seconds": timings,
        "checked_over_unchecked": timings["checked"] / timings["unchecked"],
        "checked_over_list_reference": (
            timings["checked"] / timings["list_reference"]
        ),
        "active_lookups_per_second": {
            name: (
                0.0 if active_lookups == 0 else active_lookups / seconds
            )
            for name, seconds in timings.items()
        },
        "tensor_entries_per_second": {
            name: (
                0.0 if tensor_entries == 0 else tensor_entries / seconds
            )
            for name, seconds in timings.items()
        },
        "checked_traced_retained_bytes": traced_retained,
        "checked_traced_peak_bytes": traced_peak,
    }
    return result, (selected, counts, sources)


def repeated_consumer_comparison(
    selected: np.ndarray,
    counts: np.ndarray,
    sources: np.ndarray,
    repeats: int,
) -> dict:
    slots = np.empty_like(sources)
    resolve_refined_relation_source_slots_unchecked(
        selected, counts, sources, slots
    )
    materialized_actions = np.empty_like(sources)
    repeated_lookup_actions = np.empty_like(sources)

    def consume_materialized() -> None:
        for _ in range(repeats):
            np.copyto(materialized_actions, slots)

    def consume_repeated_lookup() -> None:
        for _ in range(repeats):
            resolve_refined_relation_source_slots_unchecked(
                selected,
                counts,
                sources,
                repeated_lookup_actions,
            )

    started = time.perf_counter()
    resolve_refined_relation_source_slots_unchecked(
        selected, counts, sources, slots
    )
    materialization_seconds = time.perf_counter() - started
    materialized_scan_seconds = statistics.median(
        [
            (lambda start=time.perf_counter(): (
                consume_materialized(), time.perf_counter() - start
            ))()[1]
            for _ in range(3)
        ]
    )
    repeated_lookup_seconds = statistics.median(
        [
            (lambda start=time.perf_counter(): (
                consume_repeated_lookup(), time.perf_counter() - start
            ))()[1]
            for _ in range(3)
        ]
    )
    exact = arrays_equal(materialized_actions, repeated_lookup_actions)
    if not exact or not arrays_equal(materialized_actions, slots):
        raise AssertionError("repeated action-scan slot arrays disagree")
    materialized_total = materialization_seconds + materialized_scan_seconds
    return {
        "action_scan_repetitions": repeats,
        "description": (
            "native identity-only reuse: tensor copy versus repeated unchecked "
            "resolution; no kinds, targets, regions, or values"
        ),
        "materialization_path": "unchecked after upstream validation",
        "materialization_seconds": materialization_seconds,
        "materialized_scan_seconds": materialized_scan_seconds,
        "materialized_total_seconds": materialized_total,
        "repeated_native_resolution_seconds": repeated_lookup_seconds,
        "repeated_resolution_over_materialized_total": (
            repeated_lookup_seconds / materialized_total
        ),
        "exact_full_action_slot_arrays": exact,
        "materialized_slot_artifact_bytes": slots.nbytes,
        "action_output_bytes": materialized_actions.nbytes,
    }


def forest_artifact(root: np.ndarray, flags: np.ndarray):
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
    validate_refined_forest_arrays(
        root,
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
    relation_args = (
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    validate_refined_all_touch_2to1(*relation_args)
    return forest, relation_args


def copy_window_rows(
    accepted: int,
    window_count: int,
    candidate_ids: np.ndarray,
    kinds: np.ndarray,
    masks: np.ndarray,
    counts: np.ndarray,
    sources: np.ndarray,
) -> int:
    remaining = window_count - accepted
    for row in range(remaining):
        old = accepted + row
        candidate_ids[row] = candidate_ids[old]
        kinds[row] = kinds[old]
        masks[row] = masks[old]
        counts[row] = counts[old]
        sources[row] = sources[old]
    return remaining


def weno_capacity_case(
    capacity: int,
    leaf_count: int,
    relation_args,
) -> dict:
    selected = np.empty(capacity, dtype=np.int64)
    candidate_ids = np.empty(capacity, dtype=np.int64)
    kinds = np.empty((capacity, 26), dtype=np.uint8)
    masks = np.empty_like(kinds)
    counts = np.empty_like(kinds)
    sources = np.empty((capacity, 26, 4), dtype=np.int64)
    slots = np.empty((capacity, 26, 4), dtype=np.int64)

    first = 0
    window_count = min(capacity, leaf_count)
    for row in range(window_count):
        candidate_ids[row] = row
    relation_seconds = 0.0
    compaction_seconds = 0.0
    planning_seconds = 0.0
    rsl_seconds = 0.0
    roundtrip_seconds = 0.0
    started = time.perf_counter()
    fill_balanced_refined_relations(
        *relation_args,
        candidate_ids[:window_count],
        ALL_DIRECTIONS,
        kinds[:window_count],
        masks[:window_count],
        counts[:window_count],
        sources[:window_count],
    )
    relation_seconds += time.perf_counter() - started
    relation_rows_generated = window_count
    chunks = 0
    primary_total = 0
    selected_total = 0
    active_total = 0
    tensor_total = 0
    one_pass_comparisons_total = 0
    checked_uniqueness_comparisons_total = 0
    output_bytes_total = 0
    peak_output_bytes = 0
    primary_min = leaf_count
    primary_max = 0
    selected_min = leaf_count
    selected_max = 0
    exact = True

    while first < leaf_count:
        expected_window = min(capacity, leaf_count - first)
        if window_count != expected_window:
            raise AssertionError("WENO sliding REL window has wrong size")
        started = time.perf_counter()
        primary_count, selected_count = plan_balanced_refined_support_prefix(
            first,
            leaf_count,
            counts[:window_count],
            sources[:window_count],
            selected,
        )
        planning_seconds += time.perf_counter() - started
        accepted_counts = counts[:primary_count]
        accepted_sources = sources[:primary_count]
        accepted_slots = slots[:primary_count]

        one_pass_comparisons_total += int(
            count_relation_slot_comparisons_unchecked(
                selected[:selected_count], accepted_counts, accepted_sources
            )
        )
        checked_uniqueness_comparisons_total += (
            selected_count * (selected_count - 1) // 2
        )
        started = time.perf_counter()
        resolve_refined_relation_source_slots(
            leaf_count,
            selected[:selected_count],
            accepted_counts,
            accepted_sources,
            accepted_slots,
        )
        rsl_seconds += time.perf_counter() - started

        started = time.perf_counter()
        for primary in range(primary_count):
            for direction in range(26):
                count = int(accepted_counts[primary, direction])
                for source in range(count):
                    slot = int(accepted_slots[primary, direction, source])
                    exact &= (
                        selected[slot]
                        == accepted_sources[primary, direction, source]
                    )
                for source in range(count, 4):
                    exact &= accepted_slots[primary, direction, source] == -1
        roundtrip_seconds += time.perf_counter() - started

        active = int(np.sum(accepted_counts, dtype=np.int64))
        tensor = primary_count * 26 * 4
        active_total += active
        tensor_total += tensor
        output_bytes = 8 * tensor
        output_bytes_total += output_bytes
        peak_output_bytes = max(peak_output_bytes, output_bytes)
        primary_total += primary_count
        selected_total += selected_count
        primary_min = min(primary_min, primary_count)
        primary_max = max(primary_max, primary_count)
        selected_min = min(selected_min, selected_count)
        selected_max = max(selected_max, selected_count)
        chunks += 1

        first += primary_count
        started = time.perf_counter()
        remaining = copy_window_rows(
            primary_count,
            window_count,
            candidate_ids,
            kinds,
            masks,
            counts,
            sources,
        )
        compaction_seconds += time.perf_counter() - started
        next_window = min(capacity, leaf_count - first)
        refill = next_window - remaining
        for row in range(remaining, next_window):
            candidate_ids[row] = first + row
        if refill:
            started = time.perf_counter()
            fill_balanced_refined_relations(
                *relation_args,
                candidate_ids[remaining:next_window],
                ALL_DIRECTIONS,
                kinds[remaining:next_window],
                masks[remaining:next_window],
                counts[remaining:next_window],
                sources[remaining:next_window],
            )
            relation_seconds += time.perf_counter() - started
            relation_rows_generated += refill
        window_count = next_window

    if not exact or primary_total != leaf_count:
        raise AssertionError("WENO bounded RSL roundtrip failed")
    return {
        "capacity": capacity,
        "chunks": chunks,
        "accepted_primary_total": primary_total,
        "selected_occurrences": selected_total,
        "selected_amplification": selected_total / leaf_count,
        "accepted_primary_min": primary_min,
        "accepted_primary_mean": primary_total / chunks,
        "accepted_primary_max": primary_max,
        "selected_min": selected_min,
        "selected_mean": selected_total / chunks,
        "selected_max": selected_max,
        "active_lookups": active_total,
        "tensor_entries": tensor_total,
        "one_lookup_pass_comparisons": one_pass_comparisons_total,
        "comparisons_per_active_per_pass": (
            one_pass_comparisons_total / active_total
        ),
        "checked_selected_uniqueness_comparisons": (
            checked_uniqueness_comparisons_total
        ),
        "checked_two_lookup_pass_comparisons": (
            2 * one_pass_comparisons_total
        ),
        "checked_total_identity_comparisons": (
            checked_uniqueness_comparisons_total
            + 2 * one_pass_comparisons_total
        ),
        "rsl_output_bytes_total": output_bytes_total,
        "rsl_peak_output_bytes": peak_output_bytes,
        "relation_window_bytes": int(
            kinds.nbytes + masks.nbytes + counts.nbytes + sources.nbytes
        ),
        "rsl_capacity_output_bytes": slots.nbytes,
        "relation_rows_generated": relation_rows_generated,
        "relation_row_recurrence": relation_rows_generated / leaf_count,
        "relation_generation_seconds": relation_seconds,
        "relation_compaction_seconds": compaction_seconds,
        "sto004_planning_seconds": planning_seconds,
        "rsl_seconds": rsl_seconds,
        "roundtrip_validation_seconds": roundtrip_seconds,
        "active_lookups_per_second": active_total / rsl_seconds,
        "tensor_entries_per_second": tensor_total / rsl_seconds,
        "exact_active_roundtrip_and_trailing_sentinels": bool(exact),
        "full_rel_comparator_live": False,
    }


def weno_bounded_cases(capacities: list[int]) -> dict:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        return {"available": False, "description": "WENO metadata unavailable"}
    from simesh.amrvac.datio import get_metadata

    header, flags, _ = get_metadata(str(path))
    root = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags, dtype=np.bool_),
    )
    validate_refined_forest_arrays(
        root,
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
    relation_args = (
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    validate_refined_all_touch_2to1(*relation_args)
    leaf_count = int(forest.leaf_node_ids.size)
    return {
        "available": True,
        "description": (
            "bounded sliding REL/STO/RSL phases; no full REL comparator is materialized"
        ),
        "leaves": leaf_count,
        "capacities": [
            weno_capacity_case(capacity, leaf_count, relation_args)
            for capacity in capacities
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--action-repeats", type=int, default=10)
    parser.add_argument("--weno-capacities", default="57,64,128,256,512,1024")
    args = parser.parse_args()
    if args.repeats < 1 or args.action_repeats < 1:
        raise ValueError("repeat counts must be positive")
    capacities = [int(value) for value in args.weno_capacities.split(",")]
    if any(value < 57 for value in capacities):
        raise ValueError("WENO capacities must be at least the universal 57 bound")

    standard, standard_inputs = synthetic_case(
        "standard",
        256,
        128,
        26,
        "mixed",
        "mixed",
        args.repeats,
        trace_checked=True,
    )
    selected_cases = [
        synthetic_case(
            f"selected_{selected}",
            selected,
            min(64, selected),
            26,
            2,
            "mixed",
            args.repeats,
        )[0]
        for selected in (57, 64, 128, 512, 1024)
    ]
    primary_cases = [
        synthetic_case(
            f"accepted_{primary}",
            256,
            primary,
            26,
            2,
            "mixed",
            args.repeats,
        )[0]
        for primary in (1, 8, 64, 256)
    ]
    direction_cases = [
        synthetic_case(
            f"directions_{directions}",
            256,
            64,
            directions,
            2,
            "mixed",
            args.repeats,
        )[0]
        for directions in (0, 1, 6, 26)
    ]
    count_cases = [
        synthetic_case(
            f"count_{count}",
            256,
            64,
            26,
            count,
            "mixed",
            args.repeats,
        )[0]
        for count in (0, 1, 2, 4)
    ]
    position_cases = [
        synthetic_case(
            f"position_{position}",
            256,
            64,
            26,
            2,
            position,
            args.repeats,
        )[0]
        for position in ("first", "middle", "last", "mixed")
    ]

    report = {
        "capability": "RSL-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats": args.repeats,
        "standard": standard,
        "selected_scaling": selected_cases,
        "accepted_primary_scaling": primary_cases,
        "direction_scaling": direction_cases,
        "source_count_scaling": count_cases,
        "lookup_position_scaling": position_cases,
        "repeated_identity_consumer": repeated_consumer_comparison(
            *standard_inputs,
            args.action_repeats,
        ),
        "weno_bounded": weno_bounded_cases(capacities),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
