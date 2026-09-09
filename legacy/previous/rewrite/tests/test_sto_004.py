from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.balance_reference import is_refined_all_touch_2to1_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_support import (
    maximum_balanced_refined_support_slots,
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.refined_support_reference import (
    maximum_balanced_refined_support_slots_reference,
    plan_balanced_refined_support_prefix_reference,
)
from simesh_rewrite.relations import balanced_refined_relations


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
CURRENT_COLUMNS = np.asarray(
    [
        (dz + 1) * 9 + (dy + 1) * 3 + dx + 1
        for dx, dy, dz in ALL_DIRECTIONS
    ],
    dtype=np.int64,
)
INT64_MIN = int(np.iinfo(np.int64).min)
INT64_MAX = int(np.iinfo(np.int64).max)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def rows(candidate_count: int, direction_count: int):
    return (
        np.zeros((candidate_count, direction_count), dtype=np.uint8),
        np.full(
            (candidate_count, direction_count, 4),
            -1,
            dtype=np.int64,
        ),
    )


def outcome(function, *arguments):
    try:
        return ("return", function(*arguments))
    except Exception as error:
        return (type(error), str(error))


def make_flags(root_shape: np.ndarray, refine) -> np.ndarray:
    _, roots = level1_morton(root_shape)
    flags: list[bool] = []

    def visit(level: int, coord: tuple[int, int, int]) -> None:
        split = bool(refine(level, coord))
        flags.append(not split)
        if not split:
            return
        for child in range(8):
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            visit(
                level + 1,
                tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
            )

    for root in roots:
        visit(1, tuple(int(value) for value in root))
    return np.asarray(flags, dtype=bool)


def artifact(
    shape: tuple[int, int, int],
    refine,
    *,
    require_balance: bool = True,
):
    root = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = make_flags(root, refine)
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
    if require_balance:
        validate_refined_all_touch_2to1(*relation_args)
    return root, coord_to_rank, rank_to_coord, flags, forest, relation_args


def stable_sources(counts: np.ndarray, sources: np.ndarray, row: int) -> list[int]:
    result: list[int] = []
    for direction in range(counts.shape[1]):
        for value in sources[row, direction, : int(counts[row, direction])]:
            source = int(value)
            if source not in result:
                result.append(source)
    return result


def current_fine_positions(direction) -> list[int]:
    positions = []
    for child in range(8):
        bits = tuple((child >> axis) & 1 for axis in range(3))
        if any(
            direction[axis]
            and bits[axis] != (1 if direction[axis] < 0 else 0)
            for axis in range(3)
        ):
            continue
        shell = tuple(
            0
            if direction[axis] < 0
            else 3
            if direction[axis] > 0
            else 1 + bits[axis]
            for axis in range(3)
        )
        positions.append(shell[0] + 4 * shell[1] + 16 * shell[2])
    return positions


def test_pure_rows_duplicates_and_direction_source_order_are_exact() -> None:
    counts, sources = rows(1, 3)
    counts[0] = [4, 3, 2]
    sources[0, 0] = [5, 6, 5, 7]
    sources[0, 1, :3] = [7, 4, 6]
    sources[0, 2, :2] = [2, 3]
    selected = np.full(8, -91, dtype=np.int64)
    before = selected.copy()
    actual = plan_balanced_refined_support_prefix(
        9, 10, counts, sources, selected
    )
    expected = plan_balanced_refined_support_prefix_reference(
        9, 10, counts, sources, selected.size
    )
    assert actual == expected[1:] == (1, 7)
    assert selected[:7].tolist() == [9, 5, 6, 7, 4, 2, 3]
    assert selected[:7].tolist() == expected[0]
    assert np.array_equal(selected[7:], before[7:])
    assert maximum_balanced_refined_support_slots(
        9, 10, counts, sources
    ) == 7


def test_future_primary_promotion_and_later_nonfit_preserve_support_and_suffix() -> None:
    counts, sources = rows(6, 1)
    counts[:3, 0] = [2, 2, 2]
    sources[0, 0, :2] = [1, 8]
    sources[1, 0, :2] = [2, 9]
    sources[2, 0, :2] = [3, 4]
    selected = np.asarray([101, 102, 103, 104, 105, 106], dtype=np.int64)
    before = selected.copy()
    actual = plan_balanced_refined_support_prefix(
        0, 10, counts, sources, selected
    )
    expected = plan_balanced_refined_support_prefix_reference(
        0, 10, counts, sources, selected.size
    )
    assert actual == expected[1:] == (2, 5)
    assert selected[:5].tolist() == expected[0] == [0, 1, 8, 2, 9]
    assert selected[5] == before[5]


def test_first_nonfit_is_atomic_and_later_nonfit_is_normal_success() -> None:
    counts, sources = rows(2, 1)
    counts[0, 0] = 2
    sources[0, 0, :2] = [1, 2]
    selected = np.full(2, -77, dtype=np.int64)
    before = selected.copy()
    with pytest.raises(
        ValueError,
        match="^selected capacity cannot fit the first refined support closure$",
    ):
        plan_balanced_refined_support_prefix(
            0, 5, counts, sources, selected
        )
    assert np.array_equal(selected, before)

    counts, sources = rows(4, 1)
    counts[:, 0] = [2, 1, 2, 0]
    sources[0, 0, :2] = [1, 4]
    sources[1, 0, 0] = 2
    sources[2, 0, :2] = [3, 5]
    selected = np.full(4, -79, dtype=np.int64)
    actual = plan_balanced_refined_support_prefix(
        0, 8, counts, sources, selected
    )
    assert actual == (2, 4)
    assert selected.tolist() == [0, 1, 4, 2]


@pytest.mark.parametrize(
    ("first", "leaf_count", "capacity", "expected_rows"),
    [(0, 10, 4, 4), (8, 10, 4, 2), (10, 10, 4, 0), (0, 3, 9, 3)],
)
def test_candidate_window_formula_and_short_or_long_windows_are_rejected(
    first, leaf_count, capacity, expected_rows
) -> None:
    selected = np.full(capacity, -1, dtype=np.int64)
    valid_counts, valid_sources = rows(expected_rows, 0)
    assert plan_balanced_refined_support_prefix(
        first,
        leaf_count,
        valid_counts,
        valid_sources,
        selected,
    ) == (expected_rows, expected_rows)
    for wrong_rows in {max(0, expected_rows - 1), expected_rows + 1} - {
        expected_rows
    }:
        counts, sources = rows(wrong_rows, 0)
        before = selected.copy()
        with pytest.raises(
            ValueError,
            match=rf"source rows must equal min\(capacity, remaining\)={expected_rows}",
        ):
            plan_balanced_refined_support_prefix(
                first, leaf_count, counts, sources, selected
            )
        assert np.array_equal(selected, before)


def test_end_zero_capacity_and_zero_direction_semantics() -> None:
    selected = np.asarray([INT64_MIN, -1, INT64_MAX], dtype=np.int64)
    before = selected.copy()
    counts, sources = rows(0, 5)
    assert plan_balanced_refined_support_prefix(
        7, 7, counts, sources, selected
    ) == (0, 0)
    assert np.array_equal(selected, before)

    empty_selected = np.empty(0, dtype=np.int64)
    counts, sources = rows(0, 3)
    with pytest.raises(
        ValueError,
        match="^selected capacity must be positive before the end$",
    ):
        plan_balanced_refined_support_prefix(
            0, 1, counts, sources, empty_selected
        )

    counts, sources = rows(4, 0)
    selected = np.full(4, -3, dtype=np.int64)
    assert plan_balanced_refined_support_prefix(
        3, 10, counts, sources, selected
    ) == (4, 4)
    assert selected.tolist() == [3, 4, 5, 6]
    assert maximum_balanced_refined_support_slots(
        3, 10, counts, sources
    ) == 1
    empty_counts, empty_sources = rows(0, 0)
    assert maximum_balanced_refined_support_slots(
        10, 10, empty_counts, empty_sources
    ) == 0


def test_maximum_helper_partitions_compose_to_exact_global_maximum() -> None:
    rng = np.random.default_rng(4004)
    leaf_count = 30
    counts, sources = rows(leaf_count, 3)
    for row in range(leaf_count):
        for direction in range(3):
            count = int(rng.integers(0, 5))
            counts[row, direction] = count
            if count:
                sources[row, direction, :count] = rng.integers(
                    0, leaf_count, size=count
                )
    expected = maximum_balanced_refined_support_slots_reference(
        0, leaf_count, counts, sources
    )
    actual = maximum_balanced_refined_support_slots(
        0, leaf_count, counts, sources
    )
    assert actual == expected
    partition_maxima = []
    for start, stop in ((0, 7), (7, 19), (19, 30)):
        partition_maxima.append(
            maximum_balanced_refined_support_slots(
                start,
                leaf_count,
                counts[start:stop].copy(),
                sources[start:stop].copy(),
            )
        )
    assert max(partition_maxima) == actual


def test_plain_input_fallback_above_balanced_56_source_bound_is_exact() -> None:
    capacity = 65
    counts, sources = rows(capacity, 15)
    counts[0] = 4
    sources[0] = np.arange(60, dtype=np.int64).reshape(15, 4)
    selected = np.full(capacity, -81, dtype=np.int64)
    expected = plan_balanced_refined_support_prefix_reference(
        100, 200, counts, sources, capacity
    )
    actual = plan_balanced_refined_support_prefix(
        100, 200, counts, sources, selected
    )
    assert actual == expected[1:] == (5, 65)
    assert selected.tolist() == expected[0]
    assert maximum_balanced_refined_support_slots(
        100, 200, counts, sources
    ) == maximum_balanced_refined_support_slots_reference(
        100, 200, counts, sources
    ) == 61

    first_fit = np.full(60, -83, dtype=np.int64)
    before = first_fit.copy()
    short_counts = counts[:60].copy()
    short_sources = sources[:60].copy()
    with pytest.raises(
        ValueError,
        match="^selected capacity cannot fit the first refined support closure$",
    ):
        plan_balanced_refined_support_prefix(
            100, 200, short_counts, short_sources, first_fit
        )
    assert np.array_equal(first_fit, before)


def all_finer_artifact():
    return artifact(
        (3, 3, 3),
        lambda level, coord: level == 1 and coord != (1, 1, 1),
    )


@pytest.mark.parametrize(
    ("directions", "capacity", "expected_max"),
    [(ALL_DIRECTIONS, 57, 57), (FACE_DIRECTIONS, 25, 25)],
)
def test_balanced_rel_composition_achieves_universal_all26_and_face_bounds(
    directions, capacity, expected_max
) -> None:
    _, coord_to_rank, _, _, forest, relation_args = all_finer_artifact()
    center_node = int(
        forest.root_node_ids[int(coord_to_rank[1, 1, 1])]
    )
    first = int(forest.node_leaf_ids[center_node])
    leaf_count = forest.leaf_node_ids.size
    candidate_count = min(capacity, leaf_count - first)
    candidate_ids = np.arange(first, first + candidate_count, dtype=np.int64)
    _, _, counts, sources = balanced_refined_relations(
        *relation_args,
        candidate_ids,
        directions,
    )
    assert maximum_balanced_refined_support_slots(
        first,
        leaf_count,
        counts[:1].copy(),
        sources[:1].copy(),
    ) == expected_max
    selected = np.full(capacity, -1, dtype=np.int64)
    actual = plan_balanced_refined_support_prefix(
        first, leaf_count, counts, sources, selected
    )
    assert actual == (1, expected_max)
    assert len(np.unique(selected[:expected_max])) == expected_max
    if directions.shape[0] == 26:
        assert dict(
            zip(*[value.tolist() for value in np.unique(counts[0], return_counts=True)])
        ) == {1: 8, 2: 12, 4: 6}


def test_random_balanced_rel_windows_match_list_reference() -> None:
    rng = np.random.default_rng(20260828)
    accepted = 0
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 4, size=3))
        decisions: dict[tuple[int, tuple[int, int, int]], bool] = {}

        def refine(level, coord):
            key = (level, coord)
            if key not in decisions:
                decisions[key] = level < 3 and bool(rng.random() < 0.25)
            return decisions[key]

        _, _, _, _, forest, relation_args = artifact(
            shape, refine, require_balance=False
        )
        if not is_refined_all_touch_2to1_reference(
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
        ):
            continue
        validate_refined_all_touch_2to1(*relation_args)
        leaf_count = forest.leaf_node_ids.size
        first = int(rng.integers(0, leaf_count))
        capacity = min(57, leaf_count)
        candidate_count = min(capacity, leaf_count - first)
        candidate_ids = np.arange(
            first, first + candidate_count, dtype=np.int64
        )
        _, _, counts, sources = balanced_refined_relations(
            *relation_args,
            candidate_ids,
            ALL_DIRECTIONS,
        )
        initial = rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=capacity,
            dtype=np.int64,
        )
        selected = initial.copy()
        expected = plan_balanced_refined_support_prefix_reference(
            first, leaf_count, counts, sources, capacity
        )
        actual = plan_balanced_refined_support_prefix(
            first, leaf_count, counts, sources, selected
        )
        assert actual == expected[1:]
        assert selected[: expected[2]].tolist() == expected[0]
        assert np.array_equal(selected[expected[2] :], initial[expected[2] :])
        accepted += 1
        if accepted == 20:
            break
    assert accepted == 20


def test_read_only_relation_inputs_are_accepted_and_preserved() -> None:
    counts, sources = rows(3, 2)
    counts[:, 0] = 1
    sources[:, 0, 0] = [3, 4, 5]
    before = counts.copy(), sources.copy()
    counts.setflags(write=False)
    sources.setflags(write=False)
    selected = np.full(3, -1, dtype=np.int64)
    plan_balanced_refined_support_prefix(
        0, 6, counts, sources, selected
    )
    assert np.array_equal(counts, before[0])
    assert np.array_equal(sources, before[1])
    assert not counts.flags.writeable
    assert not sources.flags.writeable


def test_count_source_and_trailing_sentinel_errors_are_atomic() -> None:
    selected = np.full(3, -17, dtype=np.int64)

    def check(counts, sources, message):
        before = selected.copy()
        with pytest.raises(ValueError, match=message):
            plan_balanced_refined_support_prefix(
                0, 6, counts, sources, selected
            )
        assert np.array_equal(selected, before)

    counts, sources = rows(3, 1)
    counts[1, 0] = 5
    check(counts, sources, "source count exceeds four")

    counts, sources = rows(3, 1)
    counts[1, 0] = 1
    sources[1, 0, 0] = 6
    check(counts, sources, "active source leaf ID is out of range")

    counts, sources = rows(3, 1)
    sources[2, 0, 3] = 0
    check(counts, sources, "unused source leaf ID must be -1")


def test_shape_dtype_layout_output_overlap_and_validation_precedence() -> None:
    counts, sources = rows(2, 1)
    selected = np.full(2, -1, dtype=np.int64)
    cases = [
        (
            [0, 0],
            sources,
            selected,
            TypeError,
            "source_counts must be a NumPy array",
        ),
        (
            counts.astype(np.int64),
            sources,
            selected,
            TypeError,
            "source_counts must have dtype uint8",
        ),
        (
            np.empty((2, 2), dtype=np.uint8)[:, ::2],
            sources,
            selected,
            ValueError,
            "source_counts must be a C-contiguous matrix",
        ),
        (
            counts,
            sources.astype(np.int32),
            selected,
            TypeError,
            "source_leaf_ids must have dtype int64",
        ),
        (
            counts,
            np.full((2, 1, 3), -1, dtype=np.int64),
            selected,
            ValueError,
            "source_leaf_ids must have shape",
        ),
        (
            counts,
            np.full((2, 2, 4), -1, dtype=np.int64)[:, ::2],
            selected,
            ValueError,
            "source_leaf_ids must be C-contiguous",
        ),
        (
            counts,
            sources,
            [-1, -1],
            TypeError,
            "selected_leaf_ids must be a NumPy array",
        ),
        (
            counts,
            sources,
            selected.astype(np.int32),
            TypeError,
            "selected_leaf_ids must have dtype int64",
        ),
        (
            counts,
            sources,
            np.full((2, 1), -1, dtype=np.int64),
            ValueError,
            "selected_leaf_ids must be a C-contiguous vector",
        ),
        (
            counts,
            sources,
            np.full(4, -1, dtype=np.int64)[::2],
            ValueError,
            "selected_leaf_ids must be a C-contiguous vector",
        ),
    ]
    for bad_counts, bad_sources, bad_selected, error_type, message in cases:
        before = bad_selected.copy() if isinstance(bad_selected, np.ndarray) else None
        with pytest.raises(error_type, match=message):
            plan_balanced_refined_support_prefix(
                0, 4, bad_counts, bad_sources, bad_selected
            )
        if before is not None:
            assert np.array_equal(bad_selected, before)

    overlap_sources = np.full((2, 1, 4), -1, dtype=np.int64)
    overlap_selected = overlap_sources.ravel()[:2]
    before = overlap_sources.copy()
    with pytest.raises(ValueError, match="must not overlap relation inputs"):
        plan_balanced_refined_support_prefix(
            0, 4, counts, overlap_sources, overlap_selected
        )
    assert np.array_equal(overlap_sources, before)

    bad_counts = counts.astype(np.int64)
    bad_selected = selected.astype(np.int32)
    assert outcome(
        plan_balanced_refined_support_prefix,
        True,
        -1,
        bad_counts,
        sources,
        bad_selected,
    ) == (TypeError, "first_primary_id must be an integer")
    assert outcome(
        plan_balanced_refined_support_prefix,
        0,
        True,
        bad_counts,
        sources,
        bad_selected,
    ) == (TypeError, "leaf_count must be an integer")

    readonly = selected.copy()
    readonly.setflags(write=False)
    assert outcome(
        plan_balanced_refined_support_prefix,
        0,
        4,
        counts,
        sources,
        readonly,
    ) == (ValueError, "selected_leaf_ids must be writable")


@pytest.mark.parametrize(
    ("first", "leaf_count", "error_type", "message"),
    [
        (-1, 4, ValueError, "first_primary_id must be non-negative"),
        (0, -1, ValueError, "leaf_count must be non-negative"),
        (
            INT64_MAX + 1,
            INT64_MAX,
            OverflowError,
            "first_primary_id does not fit in int64",
        ),
        (
            0,
            INT64_MAX + 1,
            OverflowError,
            "leaf_count does not fit in int64",
        ),
        (3, 2, ValueError, "first_primary_id exceeds leaf count"),
    ],
)
def test_scalar_ranges_and_cross_range_are_atomic(
    first, leaf_count, error_type, message
) -> None:
    counts, sources = rows(2, 1)
    selected = np.full(2, -95, dtype=np.int64)
    before = selected.copy()
    with pytest.raises(error_type, match=message):
        plan_balanced_refined_support_prefix(
            first, leaf_count, counts, sources, selected
        )
    assert np.array_equal(selected, before)


def test_output_representation_precedes_cross_range_validation() -> None:
    counts, sources = rows(2, 1)
    with pytest.raises(TypeError, match="selected_leaf_ids must have dtype int64"):
        plan_balanced_refined_support_prefix(
            3,
            2,
            counts,
            sources,
            np.full(2, -1, dtype=np.int32),
        )


def test_maximum_helper_rejects_rows_beyond_remaining_and_validates_values() -> None:
    counts, sources = rows(2, 1)
    with pytest.raises(ValueError, match="source rows exceed remaining leaf count"):
        maximum_balanced_refined_support_slots(
            4, 5, counts, sources
        )
    counts, sources = rows(1, 1)
    counts[0, 0] = 1
    sources[0, 0, 0] = -1
    with pytest.raises(ValueError, match="active source leaf ID is out of range"):
        maximum_balanced_refined_support_slots(
            0, 5, counts, sources
        )


def test_balanced_rel_full_traversal_matches_exact_sliding_window_trace() -> None:
    _, _, _, _, forest, relation_args = all_finer_artifact()
    leaf_count = forest.leaf_node_ids.size
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    _, _, full_counts, full_sources = balanced_refined_relations(
        *relation_args, leaf_ids, ALL_DIRECTIONS
    )
    capacity = 57

    naive_trace = []
    first = 0
    primary_coverage: list[int] = []
    while first < leaf_count:
        candidate_count = min(capacity, leaf_count - first)
        selected = np.full(capacity, -101, dtype=np.int64)
        primary_count, selected_count = plan_balanced_refined_support_prefix(
            first,
            leaf_count,
            full_counts[first : first + candidate_count],
            full_sources[first : first + candidate_count],
            selected,
        )
        naive_trace.append(
            (
                primary_count,
                selected_count,
                selected[:selected_count].copy(),
            )
        )
        primary_coverage.extend(selected[:primary_count].tolist())
        first += primary_count

    sliding_trace = []
    first = 0
    next_leaf = min(capacity, leaf_count)
    window_count = next_leaf
    window_counts = np.empty_like(full_counts[:capacity])
    window_sources = np.empty_like(full_sources[:capacity])
    window_counts[:window_count] = full_counts[:window_count]
    window_sources[:window_count] = full_sources[:window_count]
    while first < leaf_count:
        selected = np.full(capacity, -103, dtype=np.int64)
        primary_count, selected_count = plan_balanced_refined_support_prefix(
            first,
            leaf_count,
            window_counts[:window_count],
            window_sources[:window_count],
            selected,
        )
        sliding_trace.append(
            (
                primary_count,
                selected_count,
                selected[:selected_count].copy(),
            )
        )
        remaining = window_count - primary_count
        window_counts[:remaining] = window_counts[
            primary_count:window_count
        ].copy()
        window_sources[:remaining] = window_sources[
            primary_count:window_count
        ].copy()
        refill = min(primary_count, leaf_count - next_leaf)
        window_counts[remaining : remaining + refill] = full_counts[
            next_leaf : next_leaf + refill
        ]
        window_sources[remaining : remaining + refill] = full_sources[
            next_leaf : next_leaf + refill
        ]
        first += primary_count
        next_leaf += refill
        window_count = remaining + refill

    assert len(naive_trace) > 1
    assert len(sliding_trace) == len(naive_trace)
    for naive, sliding in zip(naive_trace, sliding_trace):
        assert naive[:2] == sliding[:2]
        assert np.array_equal(naive[2], sliding[2])
    assert primary_coverage == list(range(leaf_count))
    assert next_leaf == leaf_count


def test_mixed_physical_direction_subset_retains_rel_sources() -> None:
    root, coord_to_rank, _, flags, forest, relation_args = artifact(
        (1, 2, 1),
        lambda level, coord: level == 1 and coord == (0, 1, 0),
    )
    source_node = int(forest.root_node_ids[int(coord_to_rank[0, 0, 0])])
    first = int(forest.node_leaf_ids[source_node])
    leaf_count = forest.leaf_node_ids.size
    capacity = min(5, leaf_count - first)
    candidate_ids = np.arange(first, first + capacity, dtype=np.int64)
    mixed_direction = np.asarray([[-1, 1, -1]], dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        *relation_args, candidate_ids, mixed_direction
    )
    assert int(kinds[0, 0]) == 4
    assert int(masks[0, 0]) == 0b101
    assert int(counts[0, 0]) == 4

    current = AMRForest(
        3,
        *tuple(int(value) for value in root),
        flags.astype(np.int32),
    )
    current_column = (-1 + 1) * 9 + (1 + 1) * 3 + (-1 + 1)
    assert int(np.asarray(current.neighbor_type)[first, current_column]) == 1

    selected = np.full(capacity, -105, dtype=np.int64)
    assert plan_balanced_refined_support_prefix(
        first,
        leaf_count,
        counts,
        sources,
        selected,
    ) == (1, 5)
    assert selected[:5].tolist() == [first, *sources[0, 0].tolist()]


def test_weno_exact_maximum_and_current_ordered_union_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags_input, _ = get_metadata(str(path))
    root = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = np.ascontiguousarray(flags_input, dtype=bool)
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
    leaf_count = forest.leaf_node_ids.size
    current = AMRForest(
        3,
        *tuple(int(value) for value in root),
        flags.astype(np.int32),
    )
    current_types = np.asarray(current.neighbor_type)
    current_ids = np.asarray(current.neighbor_index)
    current_children = np.asarray(current.neighbor_children)
    maximum = 0
    window = 1024
    for start in range(0, leaf_count, window):
        stop = min(start + window, leaf_count)
        candidate_ids = np.arange(start, stop, dtype=np.int64)
        _, _, counts, sources = balanced_refined_relations(
            *relation_args,
            candidate_ids,
            ALL_DIRECTIONS,
        )
        maximum = max(
            maximum,
            maximum_balanced_refined_support_slots(
                start, leaf_count, counts, sources
            ),
        )
        for row, leaf_id in enumerate(candidate_ids):
            rel_support = stable_sources(counts, sources, row)
            current_support: list[int] = []
            for direction_index, direction_value in enumerate(ALL_DIRECTIONS):
                direction = tuple(int(value) for value in direction_value)
                column = int(CURRENT_COLUMNS[direction_index])
                kind = int(current_types[leaf_id, column])
                values = []
                if kind in (2, 3):
                    values = [int(current_ids[leaf_id, column]) - 1]
                elif kind == 4:
                    values = [
                        int(current_children[leaf_id, position]) - 1
                        for position in current_fine_positions(direction)
                    ]
                for value in values:
                    if value not in current_support:
                        current_support.append(value)
            assert current_support == rel_support
    assert maximum == 53
