"""RST-001 direct-kernel and bounded refined-composition measurements."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite._restriction import restrict_cartesian_2to1_into_unchecked
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_support import (
    maximum_balanced_refined_support_slots,
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relations import (
    RELATION_FINER,
    balanced_refined_relations,
    fill_balanced_refined_relations,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into
from simesh_rewrite.restriction_reference import (
    restrict_cartesian_2to1_reference,
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


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def interleaved_median_seconds(operations: list, repeats: int) -> list[float]:
    samples = [[] for _ in operations]
    for repetition in range(repeats):
        for offset in range(len(operations)):
            index = (repetition + offset) % len(operations)
            started = time.perf_counter()
            operations[index]()
            samples[index].append(time.perf_counter() - started)
    return [statistics.median(values) for values in samples]


def bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def numpy_fixed_order_into(fine: np.ndarray, coarse: np.ndarray) -> None:
    """Vectorized caller-buffered comparator with the contracted add tree."""
    v000 = fine[:, :, 0::2, 0::2, 0::2]
    v100 = fine[:, :, 1::2, 0::2, 0::2]
    v010 = fine[:, :, 0::2, 1::2, 0::2]
    v110 = fine[:, :, 1::2, 1::2, 0::2]
    v001 = fine[:, :, 0::2, 0::2, 1::2]
    v101 = fine[:, :, 1::2, 0::2, 1::2]
    v011 = fine[:, :, 0::2, 1::2, 1::2]
    v111 = fine[:, :, 1::2, 1::2, 1::2]
    with np.errstate(over="ignore", invalid="ignore"):
        np.add(v000, v100, out=coarse)
        np.add(coarse, v010, out=coarse)
        np.add(coarse, v110, out=coarse)
        np.add(coarse, v001, out=coarse)
        np.add(coarse, v101, out=coarse)
        np.add(coarse, v011, out=coarse)
        np.add(coarse, v111, out=coarse)
        np.multiply(coarse, np.float64(0.125), out=coarse)


def throughput_metrics(
    seconds: float,
    slots: int,
    fields: int,
    fine_shape: tuple[int, int, int],
) -> dict:
    coarse_cells = slots * fields
    for extent in fine_shape:
        coarse_cells *= extent // 2
    return {
        "coarse_field_cells": coarse_cells,
        "fine_values": 8 * coarse_cells,
        "million_coarse_field_cells_per_second": coarse_cells / seconds / 1.0e6,
        "million_fine_values_per_second": 8 * coarse_cells / seconds / 1.0e6,
        "effective_72byte_gb_per_second": 72 * coarse_cells / seconds / 1.0e9,
    }


def direct_case(
    slots: int,
    fields: int,
    fine_shape: tuple[int, int, int],
    repeats: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed + 1000003 * slots + 1009 * fields + sum(fine_shape))
    fine = np.ascontiguousarray(
        rng.normal(size=(slots, fields, *fine_shape)), dtype=np.float64
    )
    coarse_shape = tuple(extent // 2 for extent in fine_shape)
    checked_output = np.empty((slots, fields, *coarse_shape), dtype=np.float64)
    unchecked_output = np.empty_like(checked_output)
    numpy_output = np.empty_like(checked_output)
    lower = i3(0, 0, 0)
    upper = i3(*fine_shape)

    def checked() -> None:
        restrict_cartesian_2to1_into(
            fine,
            lower,
            upper,
            checked_output,
            lower,
        )

    def unchecked() -> None:
        restrict_cartesian_2to1_into_unchecked(
            fine,
            lower,
            upper,
            unchecked_output,
            lower,
        )

    def numpy_comparator() -> None:
        numpy_fixed_order_into(fine, numpy_output)

    checked()
    unchecked()
    numpy_comparator()
    exact_unchecked = bits_equal(checked_output, unchecked_output)
    exact_numpy = bits_equal(checked_output, numpy_output)
    if not exact_unchecked or not exact_numpy:
        raise AssertionError("direct RST comparators disagree bitwise")

    checked_seconds, unchecked_seconds, numpy_seconds = (
        interleaved_median_seconds(
            [checked, unchecked, numpy_comparator], repeats
        )
    )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    checked()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "slots": slots,
        "fields": fields,
        "fine_shape": list(fine_shape),
        "coarse_shape": list(coarse_shape),
        "fine_input_bytes": fine.nbytes,
        "coarse_output_bytes": checked_output.nbytes,
        "checked_seconds": checked_seconds,
        "unchecked_seconds": unchecked_seconds,
        "numpy_fixed_order_seconds": numpy_seconds,
        "checked_over_unchecked": checked_seconds / unchecked_seconds,
        "checked_over_numpy_fixed_order": checked_seconds / numpy_seconds,
        "exact_checked_unchecked": exact_unchecked,
        "exact_checked_numpy_fixed_order": exact_numpy,
        "checked_traced_retained_bytes": after_current - before_current,
        "checked_traced_peak_bytes": peak - before_current,
    }
    result.update(throughput_metrics(checked_seconds, slots, fields, fine_shape))
    result["unchecked_throughput"] = throughput_metrics(
        unchecked_seconds, slots, fields, fine_shape
    )
    result["numpy_fixed_order_throughput"] = throughput_metrics(
        numpy_seconds, slots, fields, fine_shape
    )
    return result


def scalar_reference_case(repeats: int) -> dict:
    fine = np.asarray(
        [
            1.0e16,
            1.0,
            -1.0e16,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    ).reshape(1, 1, 2, 2, 2)
    lower = i3(0, 0, 0)
    upper = i3(2, 2, 2)
    compiled_output = np.empty((1, 1, 1, 1, 1), dtype=np.float64)
    reference_output = np.empty_like(compiled_output)

    def compiled() -> None:
        restrict_cartesian_2to1_into(
            fine, lower, upper, compiled_output, lower
        )

    def scalar() -> None:
        restrict_cartesian_2to1_reference(
            fine, lower, upper, reference_output, lower
        )

    compiled()
    scalar()
    exact = bits_equal(compiled_output, reference_output)
    if not exact:
        raise AssertionError("small scalar reference disagrees with RST")
    scalar_repeats = max(1, min(repeats, 5))
    return {
        "fine_shape": [2, 2, 2],
        "exact": exact,
        "compiled_value": float(compiled_output[0, 0, 0, 0, 0]),
        "scalar_value": float(reference_output[0, 0, 0, 0, 0]),
        "compiled_seconds": median_seconds(compiled, scalar_repeats),
        "scalar_seconds": median_seconds(scalar, scalar_repeats),
    }


def current_amrmesh_aggregate_case(repeats: int, seed: int) -> dict:
    """Describe full current ghost exchange; this is not a kernel comparator."""
    flags = np.asarray([False, *([True] * 8), True], dtype=np.int32)
    forest = AMRForest(3, 2, 1, 1, flags)
    block = np.asarray([4, 4, 4], dtype=np.uint32)
    ghost_width = 2
    field_count = 2
    mesh = AMRMesh(
        3,
        block,
        np.asarray([8, 4, 4], dtype=np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(ghost_width),
        np.uint32(field_count),
        forest,
    )
    rng = np.random.default_rng(seed + 19001)
    interior = np.ascontiguousarray(
        rng.normal(size=(forest.nleafs, field_count, 4, 4, 4)),
        dtype=np.float64,
    )
    eligible = np.flatnonzero(
        np.any(np.asarray(forest.neighbor_type) == 2, axis=1)
    )
    if eligible.size == 0:
        raise AssertionError("current fixture has no restriction-eligible leaves")
    expected = np.empty(
        (eligible.size, field_count, 2, 2, 2), dtype=np.float64
    )
    restrict_cartesian_2to1_into(
        np.ascontiguousarray(interior[eligible]),
        i3(0, 0, 0),
        i3(4, 4, 4),
        expected,
        i3(0, 0, 0),
    )

    samples = []
    for _ in range(repeats):
        mesh.load_interior_data(interior)
        started = time.perf_counter()
        mesh.apply_ghost_cells()
        samples.append(time.perf_counter() - started)
    seconds = statistics.median(samples)
    datac = np.asarray(mesh.datac)
    current = np.transpose(
        datac[eligible, 2:4, 2:4, 2:4, :],
        (0, 4, 1, 2, 3),
    ).copy()
    exact = bits_equal(current, expected)
    if not exact:
        raise AssertionError("current datac interior disagrees with RST")
    padded = np.asarray(mesh.padded_view())
    eligible_cells = int(eligible.size * field_count * 2**3)
    return {
        "description": (
            "current AMRMesh full apply_ghost_cells aggregate; includes physical, "
            "same-level, restriction, prolongation, scratch, and scheduling work"
        ),
        "not_a_direct_kernel_comparator": True,
        "root_shape": [2, 1, 1],
        "leaves": int(forest.nleafs),
        "eligible_leaves": eligible.tolist(),
        "block_shape": block.tolist(),
        "fields": field_count,
        "ghost_width": ghost_width,
        "seconds": seconds,
        "eligible_restricted_coarse_field_cells": eligible_cells,
        "descriptive_eligible_cells_per_second": eligible_cells / seconds,
        "eager_padded_data_bytes": padded.nbytes,
        "eager_datac_bytes": datac.nbytes,
        "bitwise_equal_eligible_datac_interior": exact,
        "interior_reload_excluded_from_timing": True,
    }


def make_one_level_flags(
    root_shape: np.ndarray,
    root_coordinates: np.ndarray,
) -> np.ndarray:
    refined_roots: list[tuple[int, int, int]] = []
    for coordinate in root_coordinates:
        value = tuple(int(component) for component in coordinate)
        if (value[0] + 2 * value[1] + 3 * value[2]) % 4 == 0:
            refined_roots.append(value)

    flags: list[bool] = []
    for coordinate in root_coordinates:
        value = tuple(int(component) for component in coordinate)
        split = value in refined_roots
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    return np.asarray(flags, dtype=np.bool_)


def forest_inputs(root_shape: np.ndarray):
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = make_one_level_flags(root_shape, rank_to_coord)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flags,
    )
    validate_refined_forest_arrays(
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
    relation_args = (
        root_shape,
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


def append_unique(values: list[int], value: int) -> None:
    if value not in values:
        values.append(value)


def finer_sources_for_primaries(
    first: int,
    primary_count: int,
    relation_kinds: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> list[int]:
    result: list[int] = []
    for primary in range(first, first + primary_count):
        for direction in range(ALL_DIRECTIONS.shape[0]):
            if relation_kinds[primary, direction] != RELATION_FINER:
                continue
            for source in range(int(source_counts[primary, direction])):
                append_unique(
                    result,
                    int(source_leaf_ids[primary, direction, source]),
                )
    return result


def finer_sources_for_window_rows(
    primary_count: int,
    relation_kinds: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> list[int]:
    result: list[int] = []
    for row in range(primary_count):
        for direction in range(ALL_DIRECTIONS.shape[0]):
            if relation_kinds[row, direction] != RELATION_FINER:
                continue
            for source in range(int(source_counts[row, direction])):
                append_unique(
                    result,
                    int(source_leaf_ids[row, direction, source]),
                )
    return result


def all_finer_sources(
    relation_kinds: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> list[int]:
    return finer_sources_for_primaries(
        0,
        relation_kinds.shape[0],
        relation_kinds,
        source_counts,
        source_leaf_ids,
    )


def unique_capacities(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def composed_case(
    capacities: list[int],
    repeats: int,
    seed: int,
) -> dict:
    root_shape = i3(8, 6, 4)
    forest, relation_args = forest_inputs(root_shape)
    leaf_count = int(forest.leaf_node_ids.size)
    leaf_ids = np.arange(leaf_count, dtype=np.int64)

    relation_started = time.perf_counter()
    relation_kinds, relation_masks, source_counts, source_leaf_ids = (
        balanced_refined_relations(
            *relation_args,
            leaf_ids,
            ALL_DIRECTIONS,
        )
    )
    resident_relation_seconds = time.perf_counter() - relation_started
    minimum_capacity = maximum_balanced_refined_support_slots(
        0,
        leaf_count,
        source_counts,
        source_leaf_ids,
    )
    resident_relation_output_bytes = int(
        relation_kinds.nbytes
        + relation_masks.nbytes
        + source_counts.nbytes
        + source_leaf_ids.nbytes
    )

    block_shape = i3(8, 8, 8)
    coarse_shape = i3(4, 4, 4)
    field_count = 3
    fields = np.arange(field_count, dtype=np.int64)
    zero = i3(0, 0, 0)
    rng = np.random.default_rng(seed + 4004)
    backing = np.ascontiguousarray(
        rng.normal(
            size=(leaf_count, field_count, 8, 8, 8)
        ),
        dtype=np.float64,
    )
    backing.setflags(write=False)

    resident_source_list = all_finer_sources(
        relation_kinds,
        source_counts,
        source_leaf_ids,
    )
    if not resident_source_list:
        raise AssertionError("refined composition produced no FINER sources")
    resident_ids = np.asarray(resident_source_list, dtype=np.int64)
    resident_fine = np.ascontiguousarray(backing[resident_ids])
    resident_coarse = np.empty(
        (resident_ids.size, field_count, 4, 4, 4), dtype=np.float64
    )
    resident_seconds = median_seconds(
        lambda: restrict_cartesian_2to1_into(
            resident_fine,
            zero,
            block_shape,
            resident_coarse,
            zero,
        ),
        repeats,
    )
    resident_compact_input_bytes = resident_fine.nbytes
    resident_output_bytes = resident_coarse.nbytes
    sentinel = np.asarray([0x7FF800000000A401], dtype=np.uint64).view(np.float64)[0]
    resident_global = np.full(
        (leaf_count, field_count, 4, 4, 4), sentinel, dtype=np.float64
    )
    resident_global[resident_ids] = resident_coarse
    resident_compact_source_count = int(resident_ids.size)
    del resident_fine
    del resident_coarse
    del resident_ids
    del leaf_ids
    del relation_kinds
    del relation_masks
    del source_counts
    del source_leaf_ids

    requested_capacities = unique_capacities(
        [minimum_capacity, *capacities, leaf_count]
    )
    cases = []
    for capacity_value in requested_capacities:
        capacity = min(max(int(capacity_value), minimum_capacity), leaf_count)
        if capacity in [case["capacity"] for case in cases]:
            continue
        selected_ids = np.empty(capacity, dtype=np.int64)
        compact_ids = np.empty(capacity, dtype=np.int64)
        fine_workspace = np.empty(
            (capacity, field_count, 8, 8, 8), dtype=np.float64
        )
        coarse_workspace = np.empty(
            (capacity, field_count, 4, 4, 4), dtype=np.float64
        )
        candidate_ids = np.empty(capacity, dtype=np.int64)
        window_kinds = np.empty(
            (capacity, ALL_DIRECTIONS.shape[0]), dtype=np.uint8
        )
        window_masks = np.empty_like(window_kinds)
        window_counts = np.empty_like(window_kinds)
        window_sources = np.empty(
            (capacity, ALL_DIRECTIONS.shape[0], 4), dtype=np.int64
        )
        bounded_global = np.full_like(resident_global, sentinel)

        def traverse() -> dict:
            wall_started = time.perf_counter()
            bounded_global.fill(sentinel)
            first = 0
            chunks = 0
            selected_occurrences = 0
            compact_occurrences = 0
            relation_rows_generated = 0
            relation_generation_seconds = 0.0
            relation_compaction_seconds = 0.0
            relation_compaction_bytes = 0
            planning_seconds = 0.0
            extraction_seconds = 0.0
            gather_seconds = 0.0
            restriction_seconds = 0.0
            output_store_seconds = 0.0
            seen_sources: list[int] = []
            window_count = min(capacity, leaf_count)
            for row in range(window_count):
                candidate_ids[row] = row
            started = time.perf_counter()
            fill_balanced_refined_relations(
                *relation_args,
                candidate_ids[:window_count],
                ALL_DIRECTIONS,
                window_kinds[:window_count],
                window_masks[:window_count],
                window_counts[:window_count],
                window_sources[:window_count],
            )
            relation_generation_seconds += time.perf_counter() - started
            relation_rows_generated += window_count
            while first < leaf_count:
                expected_window_count = min(capacity, leaf_count - first)
                if window_count != expected_window_count:
                    raise AssertionError("sliding REL window has wrong row count")
                started = time.perf_counter()
                primary_count, selected_count = (
                    plan_balanced_refined_support_prefix(
                        first,
                        leaf_count,
                        window_counts[:window_count],
                        window_sources[:window_count],
                        selected_ids,
                    )
                )
                planning_seconds += time.perf_counter() - started
                selected_occurrences += selected_count

                started = time.perf_counter()
                chunk_sources = finer_sources_for_window_rows(
                    primary_count,
                    window_kinds,
                    window_counts,
                    window_sources,
                )
                for index, source_leaf in enumerate(chunk_sources):
                    if not np.any(selected_ids[:selected_count] == source_leaf):
                        raise AssertionError(
                            "FINER source is absent from STO-004 selection"
                        )
                    compact_ids[index] = source_leaf
                    append_unique(seen_sources, source_leaf)
                compact_count = len(chunk_sources)
                extraction_seconds += time.perf_counter() - started
                compact_occurrences += compact_count

                started = time.perf_counter()
                gather_blocks_into(
                    backing,
                    zero,
                    block_shape,
                    compact_ids[:compact_count],
                    fields,
                    fine_workspace[:compact_count],
                    zero,
                )
                gather_seconds += time.perf_counter() - started

                started = time.perf_counter()
                restrict_cartesian_2to1_into(
                    fine_workspace[:compact_count],
                    zero,
                    block_shape,
                    coarse_workspace[:compact_count],
                    zero,
                )
                restriction_seconds += time.perf_counter() - started

                started = time.perf_counter()
                bounded_global[compact_ids[:compact_count]] = coarse_workspace[
                    :compact_count
                ]
                output_store_seconds += time.perf_counter() - started
                first += primary_count
                chunks += 1

                remaining_rows = window_count - primary_count
                started = time.perf_counter()
                for row in range(remaining_rows):
                    old_row = primary_count + row
                    candidate_ids[row] = candidate_ids[old_row]
                    window_kinds[row] = window_kinds[old_row]
                    window_masks[row] = window_masks[old_row]
                    window_counts[row] = window_counts[old_row]
                    window_sources[row] = window_sources[old_row]
                relation_compaction_seconds += time.perf_counter() - started
                relation_compaction_bytes += remaining_rows * (
                    35 * ALL_DIRECTIONS.shape[0] + 8
                )

                next_window_count = min(capacity, leaf_count - first)
                refill_count = next_window_count - remaining_rows
                if refill_count < 0:
                    raise AssertionError("sliding REL window retained excess rows")
                for row in range(remaining_rows, next_window_count):
                    candidate_ids[row] = first + row
                if refill_count:
                    started = time.perf_counter()
                    fill_balanced_refined_relations(
                        *relation_args,
                        candidate_ids[remaining_rows:next_window_count],
                        ALL_DIRECTIONS,
                        window_kinds[remaining_rows:next_window_count],
                        window_masks[remaining_rows:next_window_count],
                        window_counts[remaining_rows:next_window_count],
                        window_sources[remaining_rows:next_window_count],
                    )
                    relation_generation_seconds += time.perf_counter() - started
                    relation_rows_generated += refill_count
                window_count = next_window_count

            exact = bits_equal(bounded_global, resident_global)
            if not exact or seen_sources != resident_source_list:
                raise AssertionError(
                    "bounded compact FINER restriction disagrees with resident"
                )
            stage_total = (
                relation_generation_seconds
                + relation_compaction_seconds
                + planning_seconds
                + extraction_seconds
                + gather_seconds
                + restriction_seconds
                + output_store_seconds
            )
            wall_seconds = time.perf_counter() - wall_started
            return {
                "capacity": capacity,
                "chunks": chunks,
                "selected_occurrences": selected_occurrences,
                "selected_load_amplification": selected_occurrences / leaf_count,
                "compact_finer_source_occurrences": compact_occurrences,
                "unique_finer_sources": len(seen_sources),
                "compact_load_recurrence": compact_occurrences / len(seen_sources),
                "avoided_nonfiner_selected_loads": (
                    selected_occurrences - compact_occurrences
                ),
                "planning_seconds": planning_seconds,
                "relation_rows_generated": relation_rows_generated,
                "relation_row_recurrence": relation_rows_generated / leaf_count,
                "relation_generation_seconds": relation_generation_seconds,
                "relation_compaction_seconds": relation_compaction_seconds,
                "relation_compaction_bytes": relation_compaction_bytes,
                "finer_source_extraction_seconds": extraction_seconds,
                "gather_seconds": gather_seconds,
                "restriction_seconds": restriction_seconds,
                "output_store_seconds": output_store_seconds,
                "stage_total_seconds": stage_total,
                "wall_seconds": wall_seconds,
                "exact_full_output_array": exact,
            }

        traverse()
        samples = [traverse() for _ in range(repeats)]
        chosen = min(
            samples,
            key=lambda sample: abs(
                sample["stage_total_seconds"]
                - statistics.median(
                    value["stage_total_seconds"] for value in samples
                )
            ),
        )
        processed_cells = (
            chosen["compact_finer_source_occurrences"]
            * field_count
            * int(np.prod(coarse_shape))
        )
        gathered_bytes = (
            chosen["compact_finer_source_occurrences"]
            * field_count
            * int(np.prod(block_shape))
            * 8
        )
        chosen["restricted_coarse_field_cells"] = processed_cells
        chosen["restriction_million_cells_per_second"] = (
            processed_cells / chosen["restriction_seconds"] / 1.0e6
        )
        chosen["restriction_effective_72byte_gb_per_second"] = (
            72 * processed_cells / chosen["restriction_seconds"] / 1.0e9
        )
        chosen["gather_gb_per_second"] = (
            gathered_bytes / chosen["gather_seconds"] / 1.0e9
        )
        chosen["relation_window_output_bytes"] = int(
            window_kinds.nbytes
            + window_masks.nbytes
            + window_counts.nbytes
            + window_sources.nbytes
        )
        chosen["candidate_id_bytes"] = candidate_ids.nbytes
        chosen["selected_id_bytes"] = selected_ids.nbytes
        chosen["compact_id_bytes"] = compact_ids.nbytes
        chosen["fine_workspace_bytes"] = fine_workspace.nbytes
        chosen["coarse_workspace_bytes"] = coarse_workspace.nbytes
        chosen["managed_bounded_working_bytes"] = int(
            chosen["relation_window_output_bytes"]
            + candidate_ids.nbytes
            + selected_ids.nbytes
            + compact_ids.nbytes
            + fine_workspace.nbytes
            + coarse_workspace.nbytes
        )
        chosen["final_result_bytes_outside_working_budget"] = (
            bounded_global.nbytes
        )
        cases.append(chosen)

    return {
        "root_shape": root_shape.tolist(),
        "nodes": int(forest.node_levels.size),
        "leaves": leaf_count,
        "fields": field_count,
        "block_shape": block_shape.tolist(),
        "minimum_support_capacity": int(minimum_capacity),
        "resident_relation_generation_seconds": resident_relation_seconds,
        "resident_relation_output_bytes": resident_relation_output_bytes,
        "resident_full_relation_arrays_live_during_bounded_timings": False,
        "bounded_process_peak_memory_measured": False,
        "bounded_window_bytes_are_exact_array_sizes_not_process_peak": True,
        "resident_compact_source_count": resident_compact_source_count,
        "resident_compact_input_bytes": resident_compact_input_bytes,
        "resident_output_bytes": resident_output_bytes,
        "resident_restriction_seconds": resident_seconds,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--composition-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--composition-capacities", default="64,128,256")
    args = parser.parse_args()
    if args.repeats < 1 or args.composition_repeats < 1:
        raise ValueError("repeat counts must be positive")

    standard = direct_case(64, 4, (16, 16, 16), args.repeats, args.seed)
    shape_scaling = [
        direct_case(64, 4, (extent, extent, extent), args.repeats, args.seed)
        for extent in (4, 8, 32)
    ]
    field_scaling = [
        direct_case(64, fields, (16, 16, 16), args.repeats, args.seed)
        for fields in (1, 8)
    ]
    slot_scaling = [
        direct_case(slots, 4, (16, 16, 16), args.repeats, args.seed)
        for slots in (1, 8, 256)
    ]
    composition_capacities = [
        int(value) for value in args.composition_capacities.split(",")
    ]
    if any(value <= 0 for value in composition_capacities):
        raise ValueError("composition capacities must be positive")

    report = {
        "capability": "RST-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats": args.repeats,
        "composition_repeats": args.composition_repeats,
        "standard": standard,
        "shape_scaling": [
            shape_scaling[0],
            shape_scaling[1],
            standard,
            shape_scaling[2],
        ],
        "field_scaling": [*field_scaling[:1], standard, *field_scaling[1:]],
        "slot_scaling": [*slot_scaling[:2], standard, *slot_scaling[2:]],
        "small_scalar_reference": scalar_reference_case(args.repeats),
        "current_amrmesh_aggregate_descriptive": current_amrmesh_aggregate_case(
            args.repeats,
            args.seed,
        ),
        "bounded_refined_composition": composed_case(
            composition_capacities,
            args.composition_repeats,
            args.seed,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
