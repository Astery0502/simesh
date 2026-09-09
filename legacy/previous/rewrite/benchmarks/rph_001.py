"""RPH-001 balanced scaling, bounded WENO, reuse, and consumer evidence."""

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

from simesh.utils.lib.amr.forest import AMRForest as CurrentAMRForest
from simesh_rewrite._relation_phases import (
    fill_refined_relation_phase_codes_unchecked,
)
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.prolongation import prolong_cartesian_2to1_into
from simesh_rewrite.refined_support import (
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relation_phases import (
    fill_refined_relation_phase_codes,
)
from simesh_rewrite.relation_phases_reference import (
    refined_relation_phase_codes_reference,
)
from simesh_rewrite.relation_slots import (
    resolve_refined_relation_source_slots,
)
from simesh_rewrite.relations import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    balanced_refined_relations,
    fill_balanced_refined_relations,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into


NO_PHASE = np.uint8(255)
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
FACE_DIRECTIONS = np.asarray(
    [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ],
    dtype=np.int64,
)
EDGE_DIRECTIONS = np.ascontiguousarray(
    ALL_DIRECTIONS[np.count_nonzero(ALL_DIRECTIONS, axis=1) == 2]
)
CORNER_DIRECTIONS = np.ascontiguousarray(
    ALL_DIRECTIONS[np.count_nonzero(ALL_DIRECTIONS, axis=1) == 3]
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


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


def one_level_flags(
    root: np.ndarray,
    refined_roots: list[tuple[int, int, int]],
) -> np.ndarray:
    _, root_coords = level1_morton(root)
    flags: list[bool] = []
    for coord_array in root_coords:
        coord = tuple(int(value) for value in coord_array)
        split = any(coord == candidate for candidate in refined_roots)
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    return np.asarray(flags, dtype=np.bool_)


def nested_level_three_flags(root: np.ndarray) -> np.ndarray:
    """Refine all roots once and the low child of the center root again."""
    _, root_coords = level1_morton(root)
    center = tuple(int(value // 2) for value in root)
    flags: list[bool] = []
    for coord_array in root_coords:
        coord = tuple(int(value) for value in coord_array)
        flags.append(False)
        for child in range(8):
            split = coord == center and child == 0
            flags.append(not split)
            if split:
                flags.extend([True] * 8)
    return np.asarray(flags, dtype=np.bool_)


def full_selection_with_primary_prefix(
    primary_leaf_ids: np.ndarray,
    leaf_count: int,
) -> np.ndarray:
    selected = [int(value) for value in primary_leaf_ids]
    for leaf_id in range(leaf_count):
        if leaf_id not in selected:
            selected.append(leaf_id)
    return np.asarray(selected, dtype=np.int64)


def forest_context(root_shape: tuple[int, int, int], flags: np.ndarray) -> dict:
    root = i3(*root_shape)
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
    return {
        "root": root,
        "coord_to_rank": coord_to_rank,
        "flags": flags,
        "forest": forest,
        "relation_args": relation_args,
        "leaf_count": int(forest.leaf_node_ids.size),
    }


def relation_case(
    context: dict,
    primary_leaf_ids: np.ndarray,
    directions: np.ndarray,
    *,
    selected_leaf_ids: np.ndarray | None = None,
    selected_target: int | None = None,
) -> dict:
    primary_leaf_ids = np.ascontiguousarray(primary_leaf_ids, dtype=np.int64)
    directions = np.ascontiguousarray(directions, dtype=np.int64)
    kinds, masks, counts, source_leaf_ids = balanced_refined_relations(
        *context["relation_args"],
        primary_leaf_ids,
        directions,
    )
    leaf_count = context["leaf_count"]
    if selected_leaf_ids is None:
        selected_list = [int(value) for value in primary_leaf_ids]
        for primary in range(primary_leaf_ids.size):
            for direction in range(directions.shape[0]):
                for source in range(int(counts[primary, direction])):
                    leaf_id = int(source_leaf_ids[primary, direction, source])
                    if leaf_id not in selected_list:
                        selected_list.append(leaf_id)
        minimum_selected = len(selected_list)
        if selected_target is None:
            selected_target = minimum_selected
        if selected_target < minimum_selected or selected_target > leaf_count:
            raise ValueError(
                "selected target must contain every primary and active source"
            )
        for leaf_id in range(leaf_count):
            if len(selected_list) == selected_target:
                break
            if leaf_id not in selected_list:
                selected_list.append(leaf_id)
        selected_leaf_ids = np.asarray(selected_list, dtype=np.int64)
    else:
        selected_leaf_ids = np.ascontiguousarray(
            selected_leaf_ids, dtype=np.int64
        )
        minimum_selected = None
    if not np.array_equal(
        selected_leaf_ids[: primary_leaf_ids.size], primary_leaf_ids
    ):
        raise AssertionError("accepted primary rows must be selected prefix")
    slots = np.empty_like(source_leaf_ids)
    resolve_refined_relation_source_slots(
        leaf_count,
        selected_leaf_ids,
        counts,
        source_leaf_ids,
        slots,
    )
    return {
        "context": context,
        "selected": selected_leaf_ids,
        "directions": directions,
        "kinds": kinds,
        "masks": masks,
        "counts": counts,
        "source_leaf_ids": source_leaf_ids,
        "slots": slots,
        "minimum_selected": minimum_selected,
    }


def phase_arguments(case: dict) -> tuple[np.ndarray, ...]:
    forest = case["context"]["forest"]
    return (
        case["selected"],
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        case["directions"],
        case["kinds"],
        case["masks"],
        case["counts"],
        case["slots"],
    )


def active_phase_count(kinds: np.ndarray, counts: np.ndarray) -> int:
    return int(
        np.count_nonzero(kinds == RELATION_COARSER)
        + np.sum(counts[kinds == RELATION_FINER], dtype=np.int64)
    )


def distribution(values: np.ndarray, possible: tuple[int, ...]) -> dict:
    return {
        str(value): int(np.count_nonzero(values == value))
        for value in possible
    }


def run_phase_case(
    label: str,
    case: dict,
    repeats: int,
    *,
    trace_checked: bool = False,
) -> tuple[dict, np.ndarray]:
    arguments = phase_arguments(case)
    output_shape = case["slots"].shape
    checked_output = np.empty(output_shape, dtype=np.uint8)
    unchecked_output = np.empty(output_shape, dtype=np.uint8)
    reference_output = np.empty(output_shape, dtype=np.uint8)
    selected, _, node_coords, leaf_node_ids, _, kinds, _, counts, slots = arguments

    def checked() -> None:
        fill_refined_relation_phase_codes(*arguments, checked_output)

    def unchecked() -> None:
        fill_refined_relation_phase_codes_unchecked(
            selected,
            node_coords,
            leaf_node_ids,
            kinds,
            counts,
            slots,
            unchecked_output,
        )

    def python_reference() -> None:
        expected = refined_relation_phase_codes_reference(*arguments)
        np.copyto(reference_output, expected)

    checked()
    unchecked()
    python_reference()
    exact_unchecked = bool(np.array_equal(checked_output, unchecked_output))
    exact_reference = bool(np.array_equal(checked_output, reference_output))
    if not exact_unchecked or not exact_reference:
        raise AssertionError("RPH complete phase arrays disagree")

    timings, timing_orders = interleaved_timings(
        [
            ("checked", checked),
            ("unchecked", unchecked),
            ("python_reference_allocating", python_reference),
        ],
        repeats,
    )
    if not np.array_equal(checked_output, unchecked_output) or not np.array_equal(
        checked_output, reference_output
    ):
        raise AssertionError("timed RPH complete phase arrays disagree")

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

    primary_count, direction_count = kinds.shape
    record_count = primary_count * direction_count
    active_count = active_phase_count(kinds, counts)
    output_bytes = checked_output.nbytes
    phase_distribution = distribution(checked_output, tuple(range(8)) + (255,))
    result = {
        "label": label,
        "selected_support_count": int(selected.size),
        "accepted_primary_count": int(primary_count),
        "direction_count": int(direction_count),
        "record_count": int(record_count),
        "active_phase_count": active_count,
        "node_level_min": (
            None
            if selected.size == 0
            else int(
                np.min(
                    arguments[1][leaf_node_ids[selected]], initial=np.iinfo(np.int64).max
                )
            )
        ),
        "node_level_max": (
            None
            if selected.size == 0
            else int(np.max(arguments[1][leaf_node_ids[selected]], initial=0))
        ),
        "kind_record_counts": distribution(
            kinds,
            (
                RELATION_PHYSICAL,
                RELATION_COARSER,
                RELATION_SAME,
                RELATION_FINER,
            ),
        ),
        "physical_mask_record_counts": distribution(
            case["masks"], tuple(range(8))
        ),
        "finer_source_count_record_counts": {
            str(count): int(
                np.count_nonzero(
                    (kinds == RELATION_FINER) & (counts == count)
                )
            )
            for count in (1, 2, 4)
        },
        "phase_code_entry_counts": phase_distribution,
        "output_bytes": output_bytes,
        "expected_4PD_bytes": 4 * primary_count * direction_count,
        "exact_checked_unchecked_complete_arrays": exact_unchecked,
        "exact_checked_python_reference_complete_arrays": exact_reference,
        "python_reference_allocates_return_array": True,
        "timing_orders": timing_orders,
        "seconds": timings,
        "records_per_second": {
            name: 0.0 if record_count == 0 else record_count / seconds
            for name, seconds in timings.items()
        },
        "active_phases_per_second": {
            name: 0.0 if active_count == 0 else active_count / seconds
            for name, seconds in timings.items()
        },
        "checked_over_unchecked": timings["checked"] / timings["unchecked"],
        "checked_over_python_reference": (
            timings["checked"] / timings["python_reference_allocating"]
        ),
        "checked_traced_retained_bytes": traced_retained,
        "checked_traced_peak_bytes": traced_peak,
    }
    if output_bytes != 4 * primary_count * direction_count:
        raise AssertionError("RPH output byte formula is not exact")
    return result, checked_output


def materialization_reuse(case: dict, scans: int, repeats: int) -> dict:
    selected, _, node_coords, leaf_node_ids, _, kinds, _, counts, slots = (
        phase_arguments(case)
    )
    artifact = np.empty(slots.shape, dtype=np.uint8)
    reuse_output = np.empty_like(artifact)
    derive_output = np.empty_like(artifact)

    started = time.perf_counter()
    fill_refined_relation_phase_codes_unchecked(
        selected,
        node_coords,
        leaf_node_ids,
        kinds,
        counts,
        slots,
        artifact,
    )
    creation_seconds = time.perf_counter() - started

    def reuse() -> None:
        for _ in range(scans):
            np.copyto(reuse_output, artifact)

    def derive() -> None:
        for _ in range(scans):
            fill_refined_relation_phase_codes_unchecked(
                selected,
                node_coords,
                leaf_node_ids,
                kinds,
                counts,
                slots,
                derive_output,
            )

    timings, timing_orders = interleaved_timings(
        [
            ("materialized_np_copyto", reuse),
            ("repeated_unchecked_derivation", derive),
        ],
        repeats,
    )
    exact = bool(
        np.array_equal(reuse_output, artifact)
        and np.array_equal(reuse_output, derive_output)
    )
    if not exact:
        raise AssertionError("RPH reuse and repeated derivation arrays disagree")
    reuse_total = creation_seconds + timings["materialized_np_copyto"]
    return {
        "description": (
            "native phase-only consumer output: materialized tensor copy versus "
            "repeated unchecked phase derivation; no target or action policy"
        ),
        "scan_count": scans,
        "artifact_creation_seconds": creation_seconds,
        "artifact_bytes": artifact.nbytes,
        "consumer_output_shape": list(reuse_output.shape),
        "consumer_output_bytes": reuse_output.nbytes,
        "materialized_copy_seconds": timings["materialized_np_copyto"],
        "materialized_total_including_creation_seconds": reuse_total,
        "repeated_unchecked_derivation_seconds": timings[
            "repeated_unchecked_derivation"
        ],
        "repeated_derivation_over_materialized_total": (
            timings["repeated_unchecked_derivation"] / reuse_total
        ),
        "timing_orders": timing_orders,
        "exact_complete_consumer_arrays": exact,
    }


def first_record(
    kinds: np.ndarray,
    kind: int,
    counts: np.ndarray | None = None,
    count: int | None = None,
    masks: np.ndarray | None = None,
    require_unmasked: bool = False,
) -> tuple[int, int]:
    for primary in range(kinds.shape[0]):
        for direction in range(kinds.shape[1]):
            if int(kinds[primary, direction]) != kind:
                continue
            if counts is not None and count is not None:
                if int(counts[primary, direction]) != count:
                    continue
            if require_unmasked and int(masks[primary, direction]) != 0:
                continue
            return primary, direction
    raise AssertionError("requested relation record is absent")


def rst_prl_phase_consumer(case: dict, phases: np.ndarray) -> dict:
    kinds = case["kinds"]
    counts = case["counts"]
    masks = case["masks"]
    finer_primary, finer_direction = first_record(
        kinds,
        RELATION_FINER,
        counts,
        4,
        masks,
        True,
    )
    active_phases = phases[finer_primary, finer_direction, :4]
    fine_payload = np.empty((4, 1, 2, 2, 2), dtype=np.float64)
    for source in range(4):
        fine_payload[source].fill(float(int(active_phases[source]) + 1))
    restricted = np.full((4, 1, 1, 1, 1), np.nan)
    restrict_cartesian_2to1_into(
        fine_payload,
        i3(0, 0, 0),
        i3(2, 2, 2),
        restricted,
        i3(0, 0, 0),
    )
    direction = case["directions"][finer_direction]
    neutral_axes = [
        axis for axis in range(3) if int(direction[axis]) == 0
    ]
    tile = np.full((2, 2), np.nan)
    for source in range(4):
        phase = int(active_phases[source])
        tile[
            (phase >> neutral_axes[0]) & 1,
            (phase >> neutral_axes[1]) & 1,
        ] = restricted[source, 0, 0, 0, 0]
    restricted_exact = bool(
        np.array_equal(
            restricted[:, 0, 0, 0, 0],
            active_phases.astype(np.float64) + 1.0,
        )
        and np.all(np.isfinite(tile))
    )

    coarser_primary, coarser_direction = first_record(
        kinds, RELATION_COARSER
    )
    phase = int(phases[coarser_primary, coarser_direction, 0])
    coarse = np.empty((1, 1, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                coarse[0, 0, i, j, k] = float(i + 2 * j + 4 * k)
    prolonged = np.full((1, 1, 2, 2, 2), np.nan)
    prolong_cartesian_2to1_into(
        coarse,
        i3(0, 0, 0),
        i3(3, 3, 3),
        i3(1, 1, 1),
        prolonged,
        i3(0, 0, 0),
        i3(2, 2, 2),
        i3(0, 0, 0),
    )
    bits = tuple((phase >> axis) & 1 for axis in range(3))
    eta = [0.25 if bit else -0.25 for bit in bits]
    expected = 7.0 + eta[0] + 2.0 * eta[1] + 4.0 * eta[2]
    actual = float(prolonged[(0, 0, *bits)])
    prolongation_exact = actual == expected
    if not restricted_exact or not prolongation_exact:
        raise AssertionError("RST/PRL phase consumer disagrees")
    return {
        "description": (
            "lightweight phase-code consumption by fixed RST/PRL probes; no "
            "target-box or relation-action policy"
        ),
        "finer_record": [finer_primary, finer_direction],
        "finer_direction": direction.tolist(),
        "finer_active_phases": active_phases.tolist(),
        "restricted_values_by_source_column": restricted[:, 0, 0, 0, 0].tolist(),
        "phase_tiled_restricted_values": tile.tolist(),
        "exact_restriction_phase_columns": restricted_exact,
        "coarser_record": [coarser_primary, coarser_direction],
        "coarser_primary_phase": phase,
        "coarser_phase_bits": list(bits),
        "complete_prolonged_array": prolonged.tolist(),
        "phase_selected_prolonged_value": actual,
        "expected_phase_selected_value": expected,
        "exact_prolongation_phase_selection": prolongation_exact,
    }


def current_fine_phase_positions(
    direction: np.ndarray,
) -> tuple[list[int], list[int]]:
    phases: list[int] = []
    positions: list[int] = []
    for phase in range(8):
        bits = [(phase >> axis) & 1 for axis in range(3)]
        if not all(
            int(direction[axis]) == 0
            or bits[axis] == (1 if int(direction[axis]) < 0 else 0)
            for axis in range(3)
        ):
            continue
        shell = [
            0
            if int(direction[axis]) < 0
            else 3
            if int(direction[axis]) > 0
            else 1 + bits[axis]
            for axis in range(3)
        ]
        phases.append(phase)
        positions.append(shell[0] + 4 * shell[1] + 16 * shell[2])
    return phases, positions


def safe_current_unmasked_evidence() -> dict:
    root = i3(2, 2, 2)
    flags = one_level_flags(root, [(0, 0, 0)])
    context = forest_context((2, 2, 2), flags)
    leaf_ids = np.arange(context["leaf_count"], dtype=np.int64)
    case = relation_case(
        context,
        leaf_ids,
        ALL_DIRECTIONS,
        selected_leaf_ids=leaf_ids,
    )
    phases = np.empty(case["slots"].shape, dtype=np.uint8)
    fill_refined_relation_phase_codes(*phase_arguments(case), phases)
    current = CurrentAMRForest(3, 2, 2, 2, flags.astype(np.int32))
    current_types = np.asarray(current.neighbor_type)
    current_ids = np.asarray(current.neighbor_index)
    current_children = np.asarray(current.neighbor_children)
    compared = 0
    coarser = 0
    finer = 0
    exact = True
    for primary in range(case["kinds"].shape[0]):
        for direction_index, direction in enumerate(ALL_DIRECTIONS):
            if int(case["masks"][primary, direction_index]) != 0:
                continue
            column = (
                (int(direction[2]) + 1) * 9
                + (int(direction[1]) + 1) * 3
                + int(direction[0])
                + 1
            )
            kind = int(case["kinds"][primary, direction_index])
            exact &= int(current_types[primary, column]) == kind
            compared += 1
            if kind == RELATION_COARSER:
                slot = int(case["slots"][primary, direction_index, 0])
                exact &= int(current_ids[primary, column]) - 1 == int(
                    case["selected"][slot]
                )
                node = int(context["forest"].leaf_node_ids[primary])
                expected_phase = sum(
                    (int(context["forest"].node_coords[node, axis]) & 1)
                    << axis
                    for axis in range(3)
                )
                exact &= int(phases[primary, direction_index, 0]) == expected_phase
                coarser += 1
            elif kind == RELATION_FINER:
                expected_phases, positions = current_fine_phase_positions(direction)
                count = int(case["counts"][primary, direction_index])
                exact &= expected_phases == phases[
                    primary, direction_index, :count
                ].tolist()
                for source in range(count):
                    slot = int(case["slots"][primary, direction_index, source])
                    exact &= int(
                        current_children[primary, positions[source]]
                    ) - 1 == int(case["selected"][slot])
                finer += 1
            else:
                exact &= bool(
                    np.all(phases[primary, direction_index] == NO_PHASE)
                )
    if not exact:
        raise AssertionError("safe current unmasked phase facts disagree")
    return {
        "description": (
            "descriptive current-table comparison limited to unmasked records; "
            "mixed physical records are intentionally excluded"
        ),
        "unmasked_records_compared": compared,
        "coarser_phase_records_compared": coarser,
        "finer_phase_records_compared": finer,
        "exact_kind_identity_source_and_phase_facts": bool(exact),
        "current_neighbor_table_bytes": int(
            current_types.nbytes + current_ids.nbytes + current_children.nbytes
        ),
        "rph_phase_bytes": phases.nbytes,
    }


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
    context: dict,
) -> dict:
    leaf_count = context["leaf_count"]
    relation_args = context["relation_args"]
    forest = context["forest"]
    selected = np.empty(capacity, dtype=np.int64)
    candidate_ids = np.empty(capacity, dtype=np.int64)
    kinds = np.empty((capacity, 26), dtype=np.uint8)
    masks = np.empty_like(kinds)
    counts = np.empty_like(kinds)
    sources = np.empty((capacity, 26, 4), dtype=np.int64)
    slots = np.empty_like(sources)
    checked_phases = np.empty((capacity, 26, 4), dtype=np.uint8)
    unchecked_phases = np.empty_like(checked_phases)
    reference_phases = np.empty_like(checked_phases)

    first = 0
    window_count = min(capacity, leaf_count)
    candidate_ids[:window_count] = np.arange(window_count, dtype=np.int64)
    relation_seconds = 0.0
    compaction_seconds = 0.0
    planning_seconds = 0.0
    rsl_seconds = 0.0
    checked_seconds = 0.0
    unchecked_seconds = 0.0
    reference_seconds = 0.0
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
    chunk_count = 0
    primary_total = 0
    selected_total = 0
    record_total = 0
    active_total = 0
    output_bytes_total = 0
    peak_output_bytes = 0
    primary_min = leaf_count
    primary_max = 0
    selected_min = leaf_count
    selected_max = 0
    kind_counts = np.zeros(5, dtype=np.int64)
    mask_counts = np.zeros(8, dtype=np.int64)
    finer_count_counts = np.zeros(5, dtype=np.int64)
    phase_counts = np.zeros(256, dtype=np.int64)
    exact = True
    representative_timing_orders = []

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
        accepted_kinds = kinds[:primary_count]
        accepted_masks = masks[:primary_count]
        accepted_counts = counts[:primary_count]
        accepted_sources = sources[:primary_count]
        accepted_slots = slots[:primary_count]
        started = time.perf_counter()
        resolve_refined_relation_source_slots(
            leaf_count,
            selected[:selected_count],
            accepted_counts,
            accepted_sources,
            accepted_slots,
        )
        rsl_seconds += time.perf_counter() - started
        arguments = (
            selected[:selected_count],
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            ALL_DIRECTIONS,
            accepted_kinds,
            accepted_masks,
            accepted_counts,
            accepted_slots,
        )

        def checked() -> None:
            fill_refined_relation_phase_codes(
                *arguments, checked_phases[:primary_count]
            )

        def unchecked() -> None:
            fill_refined_relation_phase_codes_unchecked(
                selected[:selected_count],
                forest.node_coords,
                forest.leaf_node_ids,
                accepted_kinds,
                accepted_counts,
                accepted_slots,
                unchecked_phases[:primary_count],
            )

        def reference() -> None:
            expected = refined_relation_phase_codes_reference(*arguments)
            np.copyto(reference_phases[:primary_count], expected)

        operations = [
            ("checked", checked),
            ("unchecked", unchecked),
            ("python_reference_allocating", reference),
        ]
        order = [
            operations[(chunk_count + offset) % len(operations)]
            for offset in range(len(operations))
        ]
        if chunk_count < len(operations):
            representative_timing_orders.append([name for name, _ in order])
        for name, operation in order:
            started = time.perf_counter()
            operation()
            seconds = time.perf_counter() - started
            if name == "checked":
                checked_seconds += seconds
            elif name == "unchecked":
                unchecked_seconds += seconds
            else:
                reference_seconds += seconds
        exact &= bool(
            np.array_equal(
                checked_phases[:primary_count],
                unchecked_phases[:primary_count],
            )
            and np.array_equal(
                checked_phases[:primary_count],
                reference_phases[:primary_count],
            )
        )

        records = primary_count * 26
        active = active_phase_count(accepted_kinds, accepted_counts)
        output_bytes = checked_phases[:primary_count].nbytes
        primary_total += primary_count
        selected_total += selected_count
        record_total += records
        active_total += active
        output_bytes_total += output_bytes
        peak_output_bytes = max(peak_output_bytes, output_bytes)
        primary_min = min(primary_min, primary_count)
        primary_max = max(primary_max, primary_count)
        selected_min = min(selected_min, selected_count)
        selected_max = max(selected_max, selected_count)
        for kind in range(1, 5):
            kind_counts[kind] += np.count_nonzero(accepted_kinds == kind)
        for mask in range(8):
            mask_counts[mask] += np.count_nonzero(accepted_masks == mask)
        for count in (1, 2, 4):
            finer_count_counts[count] += np.count_nonzero(
                (accepted_kinds == RELATION_FINER)
                & (accepted_counts == count)
            )
        unique_phases, occurrences = np.unique(
            checked_phases[:primary_count], return_counts=True
        )
        for phase, occurrence in zip(unique_phases, occurrences, strict=True):
            phase_counts[int(phase)] += int(occurrence)
        chunk_count += 1
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
        for row in range(remaining, next_window):
            candidate_ids[row] = first + row
        refill = next_window - remaining
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
        raise AssertionError("bounded WENO RPH phase/reference traversal failed")
    if output_bytes_total != 4 * record_total:
        raise AssertionError("bounded WENO RPH byte total is not exact")
    return {
        "capacity": capacity,
        "chunks": chunk_count,
        "accepted_primary_total": primary_total,
        "selected_support_occurrences": selected_total,
        "selected_amplification": selected_total / leaf_count,
        "accepted_primary_min": primary_min,
        "accepted_primary_mean": primary_total / chunk_count,
        "accepted_primary_max": primary_max,
        "selected_support_min": selected_min,
        "selected_support_mean": selected_total / chunk_count,
        "selected_support_max": selected_max,
        "record_count": record_total,
        "active_phase_count": active_total,
        "kind_record_counts": {
            str(kind): int(kind_counts[kind]) for kind in range(1, 5)
        },
        "physical_mask_record_counts": {
            str(mask): int(mask_counts[mask]) for mask in range(8)
        },
        "finer_source_count_record_counts": {
            str(count): int(finer_count_counts[count]) for count in (1, 2, 4)
        },
        "phase_code_entry_counts": {
            **{str(phase): int(phase_counts[phase]) for phase in range(8)},
            "255": int(phase_counts[255]),
        },
        "rph_output_bytes_total": output_bytes_total,
        "expected_4PD_bytes_total": 4 * record_total,
        "one_phase_artifact_peak_bytes": peak_output_bytes,
        "rph_capacity_output_bytes_per_array": checked_phases.nbytes,
        "three_comparison_phase_arrays_capacity_bytes": int(
            checked_phases.nbytes
            + unchecked_phases.nbytes
            + reference_phases.nbytes
        ),
        "relation_window_bytes": int(
            candidate_ids.nbytes
            + kinds.nbytes
            + masks.nbytes
            + counts.nbytes
            + sources.nbytes
        ),
        "selected_capacity_bytes": selected.nbytes,
        "source_slot_capacity_bytes": slots.nbytes,
        "relation_rows_generated": relation_rows_generated,
        "relation_row_recurrence": relation_rows_generated / leaf_count,
        "relation_generation_seconds": relation_seconds,
        "relation_compaction_seconds": compaction_seconds,
        "sto004_planning_seconds": planning_seconds,
        "rsl_seconds": rsl_seconds,
        "rph_checked_seconds": checked_seconds,
        "rph_unchecked_seconds": unchecked_seconds,
        "python_reference_seconds": reference_seconds,
        "checked_records_per_second": record_total / checked_seconds,
        "unchecked_records_per_second": record_total / unchecked_seconds,
        "reference_records_per_second": record_total / reference_seconds,
        "checked_active_phases_per_second": active_total / checked_seconds,
        "unchecked_active_phases_per_second": active_total / unchecked_seconds,
        "reference_active_phases_per_second": active_total / reference_seconds,
        "interleaving_policy": (
            "checked, unchecked, and allocating Python reference rotate by chunk"
        ),
        "representative_timing_orders": representative_timing_orders,
        "exact_checked_unchecked_reference_complete_arrays": bool(exact),
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
    context = forest_context(
        tuple(int(value) for value in root),
        np.ascontiguousarray(flags, dtype=np.bool_),
    )
    return {
        "available": True,
        "description": (
            "bounded sliding REL/STO/RSL/RPH traversal with exact Python phase "
            "reference per accepted chunk; no full REL comparator is materialized"
        ),
        "leaves": context["leaf_count"],
        "capacities": [
            weno_capacity_case(capacity, context) for capacity in capacities
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reuse-scans", type=int, default=20)
    parser.add_argument("--weno-capacities", default="57,64,128,256,512,1024")
    args = parser.parse_args()
    if args.repeats < 1 or args.reuse_scans < 1:
        raise ValueError("repeat counts must be positive")
    capacities = [int(value) for value in args.weno_capacities.split(",")]
    if any(value < 57 for value in capacities):
        raise ValueError("WENO capacities must be at least the universal 57 bound")

    root_shape = (4, 4, 4)
    refined_roots = [
        (x, y, z)
        for z in range(root_shape[2])
        for y in range(root_shape[1])
        for x in range(root_shape[0])
        if (x + 2 * y + 3 * z) % 3 == 0
    ]
    root = i3(*root_shape)
    context = forest_context(
        root_shape,
        one_level_flags(root, refined_roots),
    )
    all_leaf_ids = np.arange(context["leaf_count"], dtype=np.int64)
    standard_case = relation_case(
        context,
        all_leaf_ids,
        ALL_DIRECTIONS,
        selected_leaf_ids=all_leaf_ids,
    )
    standard, standard_phases = run_phase_case(
        "balanced_mixed_standard",
        standard_case,
        args.repeats,
        trace_checked=True,
    )
    expected_standard = {
        "kinds": [1, 2, 3, 4],
        "masks": list(range(8)),
        "phases": list(range(8)),
        "finer_counts": [1, 2, 4],
    }
    observed_standard = {
        "kinds": sorted(int(value) for value in np.unique(standard_case["kinds"])),
        "masks": sorted(int(value) for value in np.unique(standard_case["masks"])),
        "phases": sorted(
            int(value)
            for value in np.unique(standard_phases)
            if int(value) != 255
        ),
        "finer_counts": sorted(
            int(value)
            for value in np.unique(
                standard_case["counts"][
                    standard_case["kinds"] == RELATION_FINER
                ]
            )
        ),
    }
    if observed_standard != expected_standard:
        raise AssertionError(
            f"standard case lacks required RPH coverage: {observed_standard}"
        )
    standard["required_coverage"] = observed_standard

    fixed_primary_ids = np.arange(min(12, context["leaf_count"]), dtype=np.int64)
    minimum_selected_case = relation_case(
        context,
        fixed_primary_ids,
        ALL_DIRECTIONS,
    )
    minimum_selected = int(minimum_selected_case["selected"].size)
    selected_targets = sorted(
        {
            minimum_selected,
            min(context["leaf_count"], minimum_selected + 16),
            min(context["leaf_count"], max(128, minimum_selected)),
            context["leaf_count"],
        }
    )
    selected_scaling = [
        run_phase_case(
            f"selected_support_{target}",
            relation_case(
                context,
                fixed_primary_ids,
                ALL_DIRECTIONS,
                selected_target=target,
            ),
            args.repeats,
        )[0]
        for target in selected_targets
    ]

    primary_scaling = [
        run_phase_case(
            f"accepted_primary_{primary_count}",
            relation_case(
                context,
                all_leaf_ids[:primary_count],
                ALL_DIRECTIONS,
                selected_leaf_ids=all_leaf_ids,
            ),
            args.repeats,
        )[0]
        for primary_count in sorted(
            {0, 1, 8, min(64, context["leaf_count"]), context["leaf_count"]}
        )
    ]

    direction_options = [
        np.empty((0, 3), dtype=np.int64),
        np.ascontiguousarray(ALL_DIRECTIONS[[0]]),
        FACE_DIRECTIONS,
        ALL_DIRECTIONS,
    ]
    direction_scaling = [
        run_phase_case(
            f"directions_{directions.shape[0]}",
            relation_case(
                context,
                all_leaf_ids[: min(64, context["leaf_count"])],
                directions,
                selected_leaf_ids=all_leaf_ids,
            ),
            args.repeats,
        )[0]
        for directions in direction_options
    ]

    depth_root = i3(3, 3, 3)
    depth_contexts = [
        (
            "active_ratio_two_maximum_level_2",
            forest_context(
                (3, 3, 3),
                one_level_flags(depth_root, [(1, 1, 1)]),
            ),
        ),
        (
            "active_ratio_two_maximum_level_3",
            forest_context(
                (3, 3, 3), nested_level_three_flags(depth_root)
            ),
        ),
    ]
    depth_primaries = []
    depth_minimum_cases = []
    for label, depth_context in depth_contexts:
        depth_ids = np.arange(depth_context["leaf_count"], dtype=np.int64)
        full_depth_case = relation_case(
            depth_context,
            depth_ids,
            ALL_DIRECTIONS,
            selected_leaf_ids=depth_ids,
        )
        active_rows = np.flatnonzero(
            np.any(
                (full_depth_case["kinds"] == RELATION_COARSER)
                | (full_depth_case["kinds"] == RELATION_FINER),
                axis=1,
            )
        ).astype(np.int64)
        if active_rows.size == 0:
            raise AssertionError(f"{label} lacks ratio-two phase records")
        primary_ids = np.ascontiguousarray(active_rows[:1])
        depth_primaries.append(primary_ids)
        depth_minimum_cases.append(
            relation_case(depth_context, primary_ids, ALL_DIRECTIONS)
        )
    fixed_depth_selected = max(
        int(case["selected"].size) for case in depth_minimum_cases
    )
    depth_scaling = []
    for (label, depth_context), primary_ids in zip(
        depth_contexts, depth_primaries, strict=True
    ):
        result, _ = run_phase_case(
            label,
            relation_case(
                depth_context,
                primary_ids,
                ALL_DIRECTIONS,
                selected_target=fixed_depth_selected,
            ),
            args.repeats,
        )
        if result["active_phase_count"] == 0:
            raise AssertionError(f"{label} did not exercise active phases")
        result["fixed_primary_and_selected_across_depth_cases"] = True
        depth_scaling.append(result)

    masked_by_row = np.count_nonzero(standard_case["masks"], axis=1)
    interior_ids = np.flatnonzero(masked_by_row == 0).astype(np.int64)
    boundary_ids = np.flatnonzero(masked_by_row > 0).astype(np.int64)
    mask_primary_count = min(16, interior_ids.size, boundary_ids.size)
    mixed_ids = np.empty(mask_primary_count, dtype=np.int64)
    for index in range(mask_primary_count):
        source = interior_ids if index % 2 == 0 else boundary_ids
        mixed_ids[index] = source[index // 2]
    mask_groups = [
        ("unmasked_primaries", interior_ids[:mask_primary_count]),
        ("boundary_masked_primaries", boundary_ids[:mask_primary_count]),
        ("mixed_mask_primaries", mixed_ids),
    ]
    mask_scaling = [
        run_phase_case(
            label,
            relation_case(
                context,
                primary_ids,
                ALL_DIRECTIONS,
                selected_leaf_ids=full_selection_with_primary_prefix(
                    primary_ids, context["leaf_count"]
                ),
            ),
            args.repeats,
        )[0]
        for label, primary_ids in mask_groups
    ]

    count_candidates = []
    for label, nonzero_axes, expected_count in (
        ("isolated_finer_face_count_four", 1, 4),
        ("isolated_finer_edge_count_two", 2, 2),
        ("isolated_finer_corner_count_one", 3, 1),
    ):
        best_direction = -1
        best_rows = np.empty(0, dtype=np.int64)
        for direction in np.flatnonzero(
            np.count_nonzero(ALL_DIRECTIONS, axis=1) == nonzero_axes
        ):
            rows = np.flatnonzero(
                (standard_case["kinds"][:, direction] == RELATION_FINER)
                & (standard_case["masks"][:, direction] == 0)
                & (standard_case["counts"][:, direction] == expected_count)
            ).astype(np.int64)
            if rows.size > best_rows.size:
                best_direction = int(direction)
                best_rows = rows
        if best_direction < 0 or best_rows.size == 0:
            raise AssertionError(f"{label} lacks an unmasked FINER record")
        count_candidates.append(
            (label, expected_count, best_direction, best_rows)
        )
    fixed_count_primaries = min(
        int(rows.size) for _, _, _, rows in count_candidates
    )
    count_scaling = []
    for label, expected_count, direction, rows in count_candidates:
        primary_ids = np.ascontiguousarray(rows[:fixed_count_primaries])
        repeated_directions = np.ascontiguousarray(
            np.repeat(ALL_DIRECTIONS[direction][None, :], 26, axis=0)
        )
        result, _ = run_phase_case(
            label,
            relation_case(
                context,
                primary_ids,
                repeated_directions,
                selected_leaf_ids=full_selection_with_primary_prefix(
                    primary_ids, context["leaf_count"]
                ),
            ),
            args.repeats,
        )
        expected_records = fixed_count_primaries * 26
        if (
            result["kind_record_counts"][str(RELATION_FINER)]
            != expected_records
            or result["finer_source_count_record_counts"][str(expected_count)]
            != expected_records
        ):
            raise AssertionError(f"{label} did not isolate the requested count")
        result["base_direction_repeated_26_times"] = ALL_DIRECTIONS[
            direction
        ].tolist()
        result["fixed_primary_selected_and_direction_counts"] = True
        count_scaling.append(result)

    report = {
        "capability": "RPH-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats": args.repeats,
        "standard": standard,
        "selected_support_scaling_independent_of_primary": selected_scaling,
        "accepted_primary_scaling_at_fixed_selected_support": primary_scaling,
        "direction_scaling": direction_scaling,
        "depth_scaling": depth_scaling,
        "physical_mask_scaling": mask_scaling,
        "finer_source_count_scaling": count_scaling,
        "native_materialization_reuse": materialization_reuse(
            standard_case, args.reuse_scans, args.repeats
        ),
        "rst_prl_phase_consumer": rst_prl_phase_consumer(
            standard_case, standard_phases
        ),
        "safe_current_unmasked_phase_facts": safe_current_unmasked_evidence(),
        "weno_bounded": weno_bounded_cases(capacities),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
