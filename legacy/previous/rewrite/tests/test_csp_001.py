from __future__ import annotations

import itertools

import numpy as np
import pytest

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.coarser_support import (
    CANONICAL_DIRECTIONS,
    NO_SOURCE,
    PLAN_CAPACITY,
    SOURCE_COARSE,
    SOURCE_FINE,
    fill_coarser_slope_support_plan,
)
from simesh_rewrite.coarser_support_reference import (
    coarser_slope_support_plan_reference,
)
from simesh_rewrite.coarser_workspace import fill_coarser_workspace_boxes
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.relation_phases import fill_refined_relation_phase_codes
from simesh_rewrite.relation_slots import resolve_refined_relation_source_slots
from simesh_rewrite.relations import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    balanced_refined_relations,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def direction_index(direction: tuple[int, int, int]) -> int:
    column = (
        (direction[2] + 1) * 9
        + (direction[1] + 1) * 3
        + direction[0]
        + 1
    )
    return column if column < 13 else column - 1


def make_flags(
    root_shape: tuple[int, int, int],
    refined_roots: list[tuple[int, int, int]],
) -> np.ndarray:
    root = i3(*root_shape)
    _, root_coords = level1_morton(root)
    flags: list[bool] = []
    for root_coord_array in root_coords:
        root_coord = tuple(int(value) for value in root_coord_array)
        if root_coord in refined_roots:
            flags.append(False)
            flags.extend([True] * 8)
        else:
            flags.append(True)
    return np.asarray(flags, dtype=np.bool_)


def relation_case(
    root_shape: tuple[int, int, int],
    refined_roots: list[tuple[int, int, int]],
):
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        make_flags(root_shape, refined_roots),
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
    validate_refined_all_touch_2to1(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    selected = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    kinds, masks, counts, source_ids = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        selected,
        CANONICAL_DIRECTIONS,
    )
    slots = np.empty_like(source_ids)
    resolve_refined_relation_source_slots(
        selected.size, selected, counts, source_ids, slots
    )
    phases = np.empty_like(source_ids, dtype=np.uint8)
    fill_refined_relation_phase_codes(
        selected,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        CANONICAL_DIRECTIONS,
        kinds,
        masks,
        counts,
        slots,
        phases,
    )
    return selected, kinds, masks, counts, slots, phases


def fresh_outputs(fill: int = 0x5A) -> tuple[np.ndarray, ...]:
    return (
        np.full(PLAN_CAPACITY, -fill, dtype=np.int64),
        np.full(PLAN_CAPACITY, fill, dtype=np.uint8),
        np.full(PLAN_CAPACITY, fill, dtype=np.uint8),
        *(
            np.full((PLAN_CAPACITY, 3), -fill, dtype=np.int64)
            for _ in range(10)
        ),
    )


def make_plan_case(
    refined_root: tuple[int, int, int] = (0, 0, 0),
    direction: tuple[int, int, int] = (1, 0, 0),
    phase: int = 1,
    *,
    block_extent: int = 4,
    reach: int | None = None,
):
    selected, kinds, masks, counts, slots, phases = relation_case(
        (2, 2, 2), [refined_root]
    )
    row = direction_index(direction)
    primary = next(
        primary
        for primary in range(selected.size)
        if int(kinds[primary, row]) == RELATION_COARSER
        and int(masks[primary, row]) == 0
        and int(phases[primary, row, 0]) == phase
    )
    lower = i3(2, 2, 2)
    upper = lower + block_extent
    if reach is None:
        reach = block_extent // 2
    target_lower = lower.copy()
    target_upper = upper.copy()
    for axis, component in enumerate(direction):
        if component < 0:
            target_lower[axis] = lower[axis] - reach
            target_upper[axis] = lower[axis]
        elif component > 0:
            target_lower[axis] = upper[axis]
            target_upper[axis] = upper[axis] + reach
    cwp = tuple(np.empty(3, dtype=np.int64) for _ in range(7))
    fill_coarser_workspace_boxes(
        lower,
        upper,
        np.asarray([direction], dtype=np.int64),
        np.asarray([phase], dtype=np.uint8),
        target_lower.reshape(1, 3),
        target_upper.reshape(1, 3),
        *(value.reshape(1, 3) for value in cwp),
    )
    inputs = (
        lower,
        upper,
        int(selected.size),
        primary,
        phase,
        np.asarray(direction, dtype=np.int64),
        CANONICAL_DIRECTIONS,
        kinds[primary],
        masks[primary],
        counts[primary],
        slots[primary],
        target_lower,
        target_upper,
        *cwp,
    )
    return inputs


def production(inputs, *, fill: int = 0x5A):
    outputs = fresh_outputs(fill)
    counts = fill_coarser_slope_support_plan(*inputs, *outputs)
    return counts, outputs


MIRRORED_CASES = [
    ((0, 0, 0), (1, 0, 0), 1, (5, 15)),
    ((0, 0, 0), (1, 1, 0), 3, (7, 11)),
    ((0, 0, 0), (1, 1, 1), 7, (8, 8)),
    ((1, 1, 1), (-1, 0, 0), 6, (5, 15)),
    ((1, 1, 1), (-1, -1, 0), 4, (7, 11)),
    ((1, 1, 1), (-1, -1, -1), 0, (8, 8)),
]


@pytest.mark.parametrize(
    ("refined_root", "direction", "phase", "expected_counts"),
    MIRRORED_CASES,
)
def test_mirrored_real_relations_match_reference_and_exact_layout(
    refined_root, direction, phase, expected_counts
) -> None:
    inputs = make_plan_case(refined_root, direction, phase)
    counts, outputs = production(inputs)
    reference = coarser_slope_support_plan_reference(*inputs)
    assert counts == expected_counts == reference[:2]
    for actual, expected in zip(outputs, reference[2:], strict=True):
        assert np.array_equal(actual, expected)
    assert sum(value.nbytes for value in outputs) == 4_500

    transfer_count, record_count = counts
    source_slots, source_is_fine, masks, directions = outputs[:4]
    source_lower, source_upper = outputs[4:6]
    target_lower, target_upper = outputs[8:10]
    required_lower, required_upper = inputs[17], inputs[18]

    assert int(source_is_fine[0]) == int(SOURCE_COARSE)
    assert int(masks[0]) == 0
    assert np.array_equal(source_lower[0], inputs[13])
    assert np.array_equal(source_upper[0], inputs[14])
    assert np.array_equal(target_lower[0], inputs[15])
    assert np.array_equal(target_upper[0], inputs[16])

    spatial_shape = tuple(int(value) for value in required_upper)
    coverage = np.zeros(spatial_shape, dtype=np.uint8)
    for record in range(record_count):
        box = tuple(
            slice(int(target_lower[record, axis]), int(target_upper[record, axis]))
            for axis in range(3)
        )
        assert not np.any(coverage[box])
        coverage[box] = 1
    required_box = tuple(
        slice(int(required_lower[axis]), int(required_upper[axis]))
        for axis in range(3)
    )
    assert np.all(coverage[required_box] == 1)
    outside = coverage.copy()
    outside[required_box] = 0
    assert not np.any(outside)

    permitted_slots = {int(inputs[3])}
    for row in range(26):
        for source in range(int(inputs[9][row])):
            permitted_slots.add(int(inputs[10][row, source]))
    for record in range(transfer_count):
        assert int(source_slots[record]) in permitted_slots
        extent = source_upper[record] - source_lower[record]
        target_extent = target_upper[record] - target_lower[record]
        ratio = 2 if int(source_is_fine[record]) else 1
        assert np.array_equal(extent, ratio * target_extent)
        assert np.all(source_lower[record] >= inputs[0])
        assert np.all(source_upper[record] <= inputs[1])

    columns = [
        (int(row[2]) + 1) * 9
        + (int(row[1]) + 1) * 3
        + int(row[0])
        + 1
        for row in directions
    ]
    assert columns[1:transfer_count] == sorted(columns[1:transfer_count])
    assert columns[transfer_count:record_count] == sorted(
        columns[transfer_count:record_count]
    )


def test_all_coarser_phase_direction_rows_match_reference() -> None:
    selected, kinds, masks, counts, slots, phases = relation_case(
        (3, 3, 3), [(1, 1, 1)]
    )
    phases_seen: set[int] = set()
    directions_seen: set[tuple[int, int, int]] = set()
    class_counts: set[int] = set()
    mismatched_class_counts: set[int] = set()
    lower = i3(2, 2, 2)
    upper = i3(6, 6, 6)
    for primary, row in np.argwhere(kinds == RELATION_COARSER):
        if int(masks[primary, row]) != 0:
            continue
        direction = tuple(int(value) for value in CANONICAL_DIRECTIONS[row])
        phase = int(phases[primary, row, 0])
        target_lower = lower.copy()
        target_upper = upper.copy()
        for axis, component in enumerate(direction):
            if component < 0:
                target_lower[axis] = 0
                target_upper[axis] = 2
            elif component > 0:
                target_lower[axis] = 6
                target_upper[axis] = 8
        cwp = tuple(np.empty(3, dtype=np.int64) for _ in range(7))
        fill_coarser_workspace_boxes(
            lower,
            upper,
            np.asarray([direction], dtype=np.int64),
            np.asarray([phase], dtype=np.uint8),
            target_lower.reshape(1, 3),
            target_upper.reshape(1, 3),
            *(value.reshape(1, 3) for value in cwp),
        )
        inputs = (
            lower,
            upper,
            int(selected.size),
            int(primary),
            phase,
            np.asarray(direction, dtype=np.int64),
            CANONICAL_DIRECTIONS,
            kinds[primary],
            masks[primary],
            counts[primary],
            slots[primary],
            target_lower,
            target_upper,
            *cwp,
        )
        actual_counts, actual = production(inputs)
        expected = coarser_slope_support_plan_reference(*inputs)
        assert actual_counts == expected[:2]
        for value, reference in zip(actual, expected[2:], strict=True):
            assert np.array_equal(value, reference)
        phases_seen.add(phase)
        directions_seen.add(direction)
        direction_class = int(np.count_nonzero(direction))
        class_counts.add(direction_class)
        mismatched = any(
            component != 0
            and ((phase >> axis) & 1) != (0 if component < 0 else 1)
            for axis, component in enumerate(direction)
        )
        if mismatched:
            mismatched_class_counts.add(direction_class)

        transfer_count, record_count = actual_counts
        assert record_count <= {1: 15, 2: 11, 3: 8}[direction_class]
        target_lower, target_upper = actual[8:10]
        required_extent = inputs[18] - inputs[17]
        target_volume = 0
        for record in range(record_count):
            target_volume += int(np.prod(target_upper[record] - target_lower[record]))
            plan_direction = tuple(int(value) for value in actual[3][record])
            if plan_direction != (0, 0, 0):
                assert int(inputs[7][direction_index(plan_direction)]) != RELATION_FINER
            for prior in range(record):
                assert any(
                    int(target_upper[record, axis])
                    <= int(target_lower[prior, axis])
                    or int(target_upper[prior, axis])
                    <= int(target_lower[record, axis])
                    for axis in range(3)
                )
        assert target_volume == int(np.prod(required_extent))
        assert transfer_count <= record_count <= PLAN_CAPACITY

    assert phases_seen == set(range(8))
    assert directions_seen == {
        tuple(int(value) for value in row) for row in CANONICAL_DIRECTIONS
    }
    assert class_counts == {1, 2, 3}
    # A genuine face COARSER contact fixes its normal phase.  Edge and corner
    # contacts supply the independently varying phase/sign cases.
    assert mismatched_class_counts == {2, 3}


def test_physical_suffix_base_owner_and_logical_metadata() -> None:
    owner_scales: set[int] = set()
    physical_kinds: set[int] = set()
    physical_masks_seen: set[int] = set()
    physical_signs: set[int] = set()
    for refined_root, direction, phase, _ in MIRRORED_CASES:
        inputs = make_plan_case(refined_root, direction, phase)
        (transfer_count, record_count), outputs = production(inputs)
        slots, source_is_fine, masks, plan_directions = outputs[:4]
        source_lower = outputs[4]
        base_lower, base_upper = outputs[6:8]
        target_lower, target_upper = outputs[8:10]
        logical_lower, logical_upper, logical_offsets = outputs[10:13]
        lower, upper = inputs[:2]
        half = (upper - lower) // 2

        for record in range(transfer_count):
            assert int(masks[record]) == 0
            assert np.array_equal(base_lower[record], target_lower[record])
            assert np.array_equal(base_upper[record], target_upper[record])
            assert not np.any(logical_lower[record])
            assert not np.any(logical_upper[record])
            assert not np.any(logical_offsets[record])

        for record in range(transfer_count, record_count):
            mask = int(masks[record])
            assert mask != 0
            assert int(slots[record]) == -1
            assert int(source_is_fine[record]) == int(NO_SOURCE)
            assert not np.any(outputs[4][record])
            assert not np.any(outputs[5][record])
            expected_base_lower = target_lower[record].copy()
            expected_base_upper = target_upper[record].copy()
            for axis in range(3):
                if mask & (1 << axis):
                    physical_signs.add(int(plan_directions[record, axis]))
                    expected_base_lower[axis] -= plan_directions[record, axis]
                    expected_base_upper[axis] -= plan_directions[record, axis]
                    assert target_upper[record, axis] - target_lower[record, axis] == 1
            assert np.array_equal(base_lower[record], expected_base_lower)
            assert np.array_equal(base_upper[record], expected_base_upper)

            owners = [
                owner
                for owner in range(transfer_count)
                if np.all(base_lower[record] >= target_lower[owner])
                and np.all(base_upper[record] <= target_upper[owner])
            ]
            assert len(owners) == 1
            owner = owners[0]
            owner_flag = int(source_is_fine[owner])
            owner_scales.add(owner_flag)
            if owner_flag:
                assert np.array_equal(logical_lower[record], i3(0, 0, 0))
                assert np.array_equal(logical_upper[record], half)
                expected_offset = (
                    (source_lower[owner] - lower) // 2 - target_lower[owner]
                )
            else:
                assert np.array_equal(logical_lower[record], lower)
                assert np.array_equal(logical_upper[record], upper)
                expected_offset = source_lower[owner] - target_lower[owner]
            assert np.array_equal(logical_offsets[record], expected_offset)
            direction_tuple = tuple(int(value) for value in plan_directions[record])
            direction_row = direction_index(direction_tuple)
            physical_kinds.add(int(inputs[7][direction_row]))
            physical_masks_seen.add(mask)

    assert owner_scales == {int(SOURCE_COARSE), int(SOURCE_FINE)}
    assert RELATION_PHYSICAL in physical_kinds
    assert RELATION_SAME in physical_kinds or RELATION_COARSER in physical_kinds
    assert physical_masks_seen
    assert physical_signs == {-1, 1}


def test_inactive_suffix_is_exact_and_success_overwrites_every_output() -> None:
    inputs = make_plan_case()
    counts, outputs = production(inputs, fill=0x37)
    _, record_count = counts
    assert np.all(outputs[0][record_count:] == -1)
    assert np.all(outputs[1][record_count:] == NO_SOURCE)
    assert not np.any(outputs[2][record_count:])
    for value in outputs[3:]:
        assert not np.any(value[record_count:])


def assert_atomic_failure(inputs, outputs, error, match: str | None = None) -> None:
    before = tuple(value.copy() for value in outputs)
    context = pytest.raises(error, match=match) if match else pytest.raises(error)
    with context:
        fill_coarser_slope_support_plan(*inputs, *outputs)
    for value, original in zip(outputs, before, strict=True):
        assert np.array_equal(value, original)


def test_scope_cwp_identity_direction_and_finer_errors_are_atomic() -> None:
    b2 = make_plan_case(block_extent=2, reach=1)
    assert_atomic_failure(b2, fresh_outputs(), ValueError, "at least four")

    wide = make_plan_case(block_extent=4, reach=3)
    assert_atomic_failure(wide, fresh_outputs(), ValueError, "B/2 reach")

    inputs = list(make_plan_case())
    for index in range(13, 20):
        changed = list(inputs)
        changed[index] = changed[index].copy()
        changed[index][0] += 1
        assert_atomic_failure(changed, fresh_outputs(), ValueError, "exact CWP")

    changed = list(inputs)
    changed[6] = changed[6].copy()
    changed[6][0], changed[6][1] = changed[6][1].copy(), changed[6][0].copy()
    assert_atomic_failure(changed, fresh_outputs(), ValueError, "canonical")

    counts, outputs = production(tuple(inputs))
    transfer_count, _ = counts
    candidate = next(
        record
        for record in range(1, transfer_count)
        if tuple(int(value) for value in outputs[3][record]) != (0, 0, 0)
    )
    row = direction_index(tuple(int(value) for value in outputs[3][candidate]))
    changed = list(inputs)
    changed[7] = changed[7].copy()
    changed[8] = changed[8].copy()
    changed[9] = changed[9].copy()
    changed[10] = changed[10].copy()
    changed[7][row] = RELATION_FINER
    changed[8][row] = 0
    neutral = sum(int(value) == 0 for value in changed[6][row])
    expected_count = 1 << neutral
    changed[9][row] = expected_count
    changed[10][row] = -1
    changed[10][row, :expected_count] = inputs[3]
    assert_atomic_failure(changed, fresh_outputs(), ValueError, "must not be FINER")


def test_relation_slot_scalar_and_representation_failures_are_atomic() -> None:
    inputs = list(make_plan_case())
    outputs = fresh_outputs()

    changed = list(inputs)
    changed[2] = 0
    assert_atomic_failure(changed, outputs, ValueError, "primary_slot")
    changed = list(inputs)
    changed[3] = inputs[2]
    assert_atomic_failure(changed, fresh_outputs(), ValueError, "primary_slot")
    changed = list(inputs)
    changed[10] = changed[10].copy()
    active = next(
        (row, source)
        for row in range(26)
        for source in range(int(inputs[9][row]))
    )
    changed[10][active] = inputs[2]
    assert_atomic_failure(changed, fresh_outputs(), ValueError, "out of range")
    changed = list(inputs)
    changed[10] = changed[10].copy()
    row = next(row for row in range(26) if int(inputs[9][row]) < 4)
    changed[10][row, int(inputs[9][row])] = 0
    assert_atomic_failure(changed, fresh_outputs(), ValueError, "must be -1")

    for index, bad, error in (
        (2, True, TypeError),
        (3, np.asarray(0, dtype=np.int64), TypeError),
        (4, 2.5, TypeError),
        (4, -1, ValueError),
        (2, int(np.iinfo(np.int64).max) + 1, OverflowError),
    ):
        changed = list(inputs)
        changed[index] = bad
        assert_atomic_failure(changed, fresh_outputs(), error)

    changed = list(inputs)
    changed[5] = changed[5].astype(np.int32)
    assert_atomic_failure(changed, fresh_outputs(), TypeError)
    changed = list(inputs)
    changed[7] = changed[7][::2]
    assert_atomic_failure(changed, fresh_outputs(), ValueError)
    bad_outputs = list(fresh_outputs())
    bad_outputs[0] = bad_outputs[0].astype(np.int32)
    assert_atomic_failure(inputs, bad_outputs, TypeError)
    bad_outputs = list(fresh_outputs())
    bad_outputs[3] = bad_outputs[3][:, ::-1]
    assert_atomic_failure(inputs, bad_outputs, ValueError)
    bad_outputs = list(fresh_outputs())
    bad_outputs[1].setflags(write=False)
    assert_atomic_failure(inputs, bad_outputs, ValueError, "writable")


def test_input_output_and_cross_dtype_output_aliases_are_atomic() -> None:
    inputs = list(make_plan_case())

    aliased = inputs[10].reshape(-1)[:54].reshape(18, 3)
    outputs = list(fresh_outputs())
    outputs[3] = aliased
    before_input = inputs[10].copy()
    assert_atomic_failure(inputs, outputs, ValueError, "overlap")
    assert np.array_equal(inputs[10], before_input)

    raw = np.full(18 * 3 * 8, 0xA5, dtype=np.uint8)
    directions = raw.view(np.int64).reshape(18, 3)
    masks = raw[:18]
    outputs = list(fresh_outputs())
    outputs[2] = masks
    outputs[3] = directions
    before_raw = raw.copy()
    assert_atomic_failure(inputs, outputs, ValueError, "overlap")
    assert np.array_equal(raw, before_raw)


def test_derived_overflow_is_atomic() -> None:
    inputs = list(
        make_plan_case(
            (0, 0, 0),
            (1, 0, 0),
            3,
            block_extent=8,
            reach=4,
        )
    )
    maximum = int(np.iinfo(np.int64).max)
    inputs[0] = i3(2, maximum - 8, 2)
    inputs[1] = i3(10, maximum, 10)
    inputs[11] = i3(10, maximum - 8, 2)
    inputs[12] = i3(14, maximum, 10)
    assert_atomic_failure(inputs, fresh_outputs(), OverflowError, "required upper")


def test_near_int64_valid_band_relative_sources_match_reference() -> None:
    selected, kinds, masks, counts, slots, phases = relation_case(
        (3, 3, 3), [(1, 1, 1), (1, 2, 1)]
    )
    direction = (1, 0, 0)
    phase = 3
    direction_row = direction_index(direction)
    primary = next(
        primary
        for primary in range(selected.size)
        if int(kinds[primary, direction_row]) == RELATION_COARSER
        and int(masks[primary, direction_row]) == 0
        and int(phases[primary, direction_row, 0]) == phase
    )

    maximum = int(np.iinfo(np.int64).max)
    block = maximum - 1
    lower = i3(0, 0, 0)
    upper = i3(block, block, block)
    target_lower = i3(block, 0, 0)
    target_upper = i3(maximum, block, block)
    cwp = tuple(np.empty(3, dtype=np.int64) for _ in range(7))
    fill_coarser_workspace_boxes(
        lower,
        upper,
        np.asarray([direction], dtype=np.int64),
        np.asarray([phase], dtype=np.uint8),
        target_lower.reshape(1, 3),
        target_upper.reshape(1, 3),
        *(value.reshape(1, 3) for value in cwp),
    )
    inputs = (
        lower,
        upper,
        int(selected.size),
        primary,
        phase,
        np.asarray(direction, dtype=np.int64),
        CANONICAL_DIRECTIONS,
        kinds[primary],
        masks[primary],
        counts[primary],
        slots[primary],
        target_lower,
        target_upper,
        *cwp,
    )
    (transfer_count, record_count), outputs = production(inputs)
    reference = coarser_slope_support_plan_reference(*inputs)
    assert (transfer_count, record_count) == reference[:2]
    for actual, expected in zip(outputs, reference[2:], strict=True):
        assert np.array_equal(actual, expected)

    source_is_fine = outputs[1]
    plan_directions = outputs[3]
    plan_source_lower, plan_source_upper = outputs[4:6]
    assert set(int(value) for value in source_is_fine[:transfer_count]) == {0, 1}
    dangerous = next(
        record
        for record in range(transfer_count)
        if int(source_is_fine[record]) == int(SOURCE_FINE)
        and int(plan_directions[record, 1]) == 1
    )
    old_doubled_intermediate = 2 * (
        int(outputs[9][dangerous, 1]) - int(cwp[6][1])
    )
    assert old_doubled_intermediate > maximum
    assert np.all(plan_source_lower[:transfer_count] >= lower)
    assert np.all(plan_source_upper[:transfer_count] <= upper)
