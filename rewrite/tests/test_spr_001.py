from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.refined_support import (
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.selected_refined_support import (
    maximum_selected_refined_support_slots,
    plan_selected_refined_support_prefix,
)
from simesh_rewrite.selected_refined_support_reference import (
    maximum_selected_refined_support_slots_reference,
    plan_selected_refined_support_prefix_reference,
)
from simesh_rewrite.storage import gather_blocks_into


def relation_rows(row_count: int, direction_count: int = 2):
    return (
        np.zeros((row_count, direction_count), dtype=np.uint8),
        np.full((row_count, direction_count, 4), -1, dtype=np.int64),
    )


def test_sparse_order_promotion_and_gap_omission_are_exact() -> None:
    primaries = np.asarray([2, 9, 15], dtype=np.int64)
    counts, sources = relation_rows(3)
    counts[0, 0] = 2
    sources[0, 0, :2] = [9, 4]
    counts[1, 0] = 2
    sources[1, 0, :2] = [15, 5]
    selected = np.full(5, -91, dtype=np.int64)

    actual = plan_selected_refined_support_prefix(
        primaries, 0, 20, counts, sources, selected
    )
    expected = plan_selected_refined_support_prefix_reference(
        primaries, 0, 20, counts, sources, selected.size
    )

    assert actual == expected[1:] == (3, 5)
    assert selected.tolist() == expected[0] == [2, 9, 15, 4, 5]
    assert not any(value in selected for value in (3, 6, 8, 10, 14))


def test_later_nonfit_returns_maximal_prefix_and_preserves_suffix() -> None:
    primaries = np.asarray([3, 10, 18, 22], dtype=np.int64)
    counts, sources = relation_rows(4, 1)
    counts[:, 0] = [1, 2, 2, 0]
    sources[0, 0, 0] = 4
    sources[1, 0, :2] = [5, 6]
    sources[2, 0, :2] = [8, 9]
    selected = np.asarray([101, 102, 103, 104, 105], dtype=np.int64)
    before = selected.copy()

    actual = plan_selected_refined_support_prefix(
        primaries, 0, 24, counts, sources, selected
    )
    expected = plan_selected_refined_support_prefix_reference(
        primaries, 0, 24, counts, sources, selected.size
    )

    assert actual == expected[1:] == (2, 5)
    assert selected[:5].tolist() == expected[0] == [3, 10, 4, 5, 6]
    assert np.array_equal(selected[actual[1] :], before[actual[1] :])


def test_first_nonfit_is_atomic() -> None:
    primaries = np.asarray([1, 8], dtype=np.int64)
    counts, sources = relation_rows(2, 1)
    counts[0, 0] = 2
    sources[0, 0, :2] = [3, 4]
    selected = np.asarray([-71, -73], dtype=np.int64)
    before = selected.copy()
    with pytest.raises(
        ValueError,
        match="^selected capacity cannot fit the first refined support closure$",
    ):
        plan_selected_refined_support_prefix(
            primaries, 0, 12, counts, sources, selected
        )
    assert np.array_equal(selected, before)


def test_position_and_complete_sparse_traversal_are_exact() -> None:
    primaries = np.asarray([2, 5, 11, 17, 23, 29], dtype=np.int64)
    all_counts, all_sources = relation_rows(primaries.size, 1)
    support_by_row = [3, 17, 7, 23, 13, 5]
    all_counts[:, 0] = 1
    all_sources[:, 0, 0] = support_by_row
    capacity = 3
    visited: list[int] = []
    position = 0
    while position < primaries.size:
        row_count = min(capacity, primaries.size - position)
        selected = np.full(capacity, -1, dtype=np.int64)
        primary_count, selected_count = plan_selected_refined_support_prefix(
            primaries,
            position,
            32,
            all_counts[position : position + row_count],
            all_sources[position : position + row_count],
            selected,
        )
        assert selected_count <= capacity
        visited.extend(selected[:primary_count].tolist())
        position += primary_count
    assert visited == primaries.tolist()


def test_dense_selection_is_bit_exact_with_sto_004() -> None:
    first = 3
    leaf_count = 11
    capacity = 6
    primaries = np.arange(first, leaf_count, dtype=np.int64)
    counts, sources = relation_rows(capacity, 2)
    for row in range(capacity):
        counts[row, 0] = 2
        sources[row, 0, :2] = [row + 1, (row + 7) % leaf_count]
        counts[row, 1] = 1
        sources[row, 1, 0] = (row + 4) % leaf_count

    dense_selected = np.full(capacity, -1, dtype=np.int64)
    sparse_selected = np.full(capacity, -1, dtype=np.int64)
    dense_counts = plan_balanced_refined_support_prefix(
        first, leaf_count, counts, sources, dense_selected
    )
    sparse_counts = plan_selected_refined_support_prefix(
        primaries,
        0,
        leaf_count,
        counts,
        sources,
        sparse_selected,
    )
    assert sparse_counts == dense_counts
    assert np.array_equal(sparse_selected, dense_selected)


def test_maximum_helper_matches_reference_and_partitions() -> None:
    primaries = np.asarray([1, 8, 14, 19], dtype=np.int64)
    counts, sources = relation_rows(4, 2)
    counts[:, 0] = [2, 1, 4, 0]
    sources[0, 0, :2] = [2, 3]
    sources[1, 0, 0] = 1
    sources[2, 0] = [4, 5, 6, 7]
    expected = maximum_selected_refined_support_slots_reference(
        primaries, counts, sources
    )
    actual = maximum_selected_refined_support_slots(
        primaries, 24, counts, sources
    )
    assert actual == expected == 5
    left = maximum_selected_refined_support_slots(
        primaries[:2], 24, counts[:2].copy(), sources[:2].copy()
    )
    right = maximum_selected_refined_support_slots(
        primaries[2:], 24, counts[2:].copy(), sources[2:].copy()
    )
    assert max(left, right) == actual


def test_end_zero_capacity_zero_directions_and_read_only_input() -> None:
    primaries = np.asarray([2, 7], dtype=np.int64)
    primaries.setflags(write=False)
    counts, sources = relation_rows(0, 3)
    selected = np.asarray([81, 82], dtype=np.int64)
    before = selected.copy()
    assert plan_selected_refined_support_prefix(
        primaries, 2, 10, counts, sources, selected
    ) == (0, 0)
    assert np.array_equal(selected, before)

    counts, sources = relation_rows(2, 0)
    selected = np.full(2, -1, dtype=np.int64)
    assert plan_selected_refined_support_prefix(
        primaries, 0, 10, counts, sources, selected
    ) == (2, 2)
    assert selected.tolist() == [2, 7]

    empty = np.empty(0, dtype=np.int64)
    counts, sources = relation_rows(0, 1)
    with pytest.raises(
        ValueError, match="^selected capacity must be positive before the end$"
    ):
        plan_selected_refined_support_prefix(
            primaries, 0, 10, counts, sources, empty
        )


@pytest.mark.parametrize(
    "bad_primary",
    [
        np.asarray([2, 2], dtype=np.int64),
        np.asarray([2, 12], dtype=np.int64),
        np.asarray([2, 7, 5], dtype=np.int64),
    ],
)
def test_invalid_primary_selection_is_atomic(bad_primary: np.ndarray) -> None:
    counts, sources = relation_rows(2, 1)
    selected = np.asarray([91, 92], dtype=np.int64)
    before = selected.copy()
    with pytest.raises(ValueError):
        plan_selected_refined_support_prefix(
            bad_primary, 0, 10, counts, sources, selected
        )
    assert np.array_equal(selected, before)


def test_window_relation_and_overlap_errors_are_atomic() -> None:
    primaries = np.asarray([2, 7, 9], dtype=np.int64)
    selected = np.asarray([71, 72, 73], dtype=np.int64)
    before = selected.copy()
    short_counts, short_sources = relation_rows(2, 1)
    with pytest.raises(ValueError, match="source rows must equal"):
        plan_selected_refined_support_prefix(
            primaries, 0, 12, short_counts, short_sources, selected
        )
    assert np.array_equal(selected, before)

    counts, sources = relation_rows(3, 1)
    counts[2, 0] = 1
    sources[2, 0, 0] = 99
    with pytest.raises(ValueError, match="active source leaf ID is out of range"):
        plan_selected_refined_support_prefix(
            primaries, 0, 12, counts, sources, selected
        )
    assert np.array_equal(selected, before)

    overlapping = primaries.copy()
    with pytest.raises(ValueError, match="must not overlap"):
        plan_selected_refined_support_prefix(
            overlapping,
            0,
            12,
            *relation_rows(3, 0),
            overlapping,
        )


def test_sparse_plan_gathers_only_primaries_and_support() -> None:
    primaries = np.asarray([1, 100, 200], dtype=np.int64)
    counts, sources = relation_rows(3, 1)
    counts[:, 0] = 1
    sources[:, 0, 0] = [2, 101, 201]
    selected = np.full(8, -1, dtype=np.int64)
    primary_count, selected_count = plan_selected_refined_support_prefix(
        primaries, 0, 256, counts, sources, selected
    )
    assert (primary_count, selected_count) == (3, 6)
    assert selected[:selected_count].tolist() == [1, 100, 200, 2, 101, 201]

    backing = np.arange(256 * 2 * 8, dtype=np.float64).reshape(
        256, 2, 2, 2, 2
    )
    destination = np.full((8, 1, 2, 2, 2), np.nan, dtype=np.float64)
    lower = np.zeros(3, dtype=np.int64)
    upper = np.full(3, 2, dtype=np.int64)
    gather_blocks_into(
        backing,
        lower,
        upper,
        selected[:selected_count],
        np.asarray([1], dtype=np.int64),
        destination[:selected_count],
        lower,
    )
    assert np.array_equal(
        destination[:selected_count, 0],
        backing[selected[:selected_count], 1],
    )
    assert np.isnan(destination[selected_count:]).all()


def test_random_sparse_plans_match_list_reference() -> None:
    rng = np.random.default_rng(20260904)
    for _ in range(500):
        leaf_count = int(rng.integers(1, 33))
        primary_total = int(rng.integers(0, leaf_count + 1))
        primaries = np.sort(
            rng.choice(leaf_count, size=primary_total, replace=False)
        ).astype(np.int64)
        first = int(rng.integers(0, primary_total + 1))
        capacity = int(rng.integers(0, 13))
        row_count = min(capacity, primary_total - first)
        counts, sources = relation_rows(row_count, 3)
        for row in range(row_count):
            for direction in range(3):
                count = int(rng.integers(0, 5))
                counts[row, direction] = count
                if count:
                    sources[row, direction, :count] = rng.integers(
                        0, leaf_count, size=count
                    )
        selected = np.full(capacity, -97, dtype=np.int64)
        before = selected.copy()
        try:
            expected = plan_selected_refined_support_prefix_reference(
                primaries,
                first,
                leaf_count,
                counts,
                sources,
                capacity,
            )
        except ValueError as reference_error:
            with pytest.raises(ValueError, match=str(reference_error)):
                plan_selected_refined_support_prefix(
                    primaries,
                    first,
                    leaf_count,
                    counts,
                    sources,
                    selected,
                )
            assert np.array_equal(selected, before)
            continue

        actual = plan_selected_refined_support_prefix(
            primaries,
            first,
            leaf_count,
            counts,
            sources,
            selected,
        )
        assert actual == expected[1:]
        assert selected[: actual[1]].tolist() == expected[0]
        assert np.array_equal(selected[actual[1] :], before[actual[1] :])


def test_checked_representation_and_scalar_errors_are_atomic() -> None:
    primaries = np.asarray([2, 7], dtype=np.int64)
    counts, sources = relation_rows(2, 1)
    selected = np.asarray([71, 72], dtype=np.int64)
    before = selected.copy()

    invalid_calls = (
        (list(primaries), 0, 10, counts, sources, selected, TypeError),
        (
            primaries.astype(np.int32),
            0,
            10,
            counts,
            sources,
            selected,
            TypeError,
        ),
        (primaries, True, 10, counts, sources, selected, TypeError),
        (primaries, 1 << 63, 10, counts, sources, selected, OverflowError),
    )
    for *arguments, error in invalid_calls:
        with pytest.raises(error):
            plan_selected_refined_support_prefix(*arguments)
        assert np.array_equal(selected, before)

    noncontiguous = np.arange(6, dtype=np.int64)[::2]
    three_counts, three_sources = relation_rows(2, 1)
    with pytest.raises(ValueError, match="C-contiguous vector"):
        plan_selected_refined_support_prefix(
            noncontiguous,
            0,
            10,
            three_counts,
            three_sources,
            selected,
        )
    assert np.array_equal(selected, before)

    readonly = selected.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="must be writable"):
        plan_selected_refined_support_prefix(
            primaries, 0, 10, counts, sources, readonly
        )
    assert np.array_equal(readonly, before)


def test_maximum_helper_empty_and_contract_errors() -> None:
    empty = np.empty(0, dtype=np.int64)
    counts, sources = relation_rows(0, 2)
    assert maximum_selected_refined_support_slots(
        empty, 10, counts, sources
    ) == 0

    primaries = np.asarray([2, 7], dtype=np.int64)
    with pytest.raises(ValueError, match="source rows must equal"):
        maximum_selected_refined_support_slots(
            primaries, 10, counts, sources
        )
    bad_counts, bad_sources = relation_rows(2, 1)
    bad_counts[1, 0] = 1
    bad_sources[1, 0, 0] = 10
    with pytest.raises(ValueError, match="active source leaf ID is out of range"):
        maximum_selected_refined_support_slots(
            primaries, 10, bad_counts, bad_sources
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        maximum_selected_refined_support_slots(
            np.asarray([7, 2], dtype=np.int64),
            10,
            *relation_rows(2, 1),
        )
