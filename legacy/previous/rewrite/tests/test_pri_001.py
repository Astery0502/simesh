from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.primary import fill_ascending_primary_prefix
from simesh_rewrite.reductions import accumulate_field_sum
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.workspace import workspace_nbytes, workspace_slot_capacity


INT64_MAX = int(np.iinfo(np.int64).max)
INT64_MIN = int(np.iinfo(np.int64).min)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def reference_prefix(
    first_primary_id: int,
    block_count: int,
    capacity: int,
) -> list[int]:
    count = min(capacity, block_count - first_primary_id)
    return list(range(first_primary_id, first_primary_id + count))


def outcome(function, *arguments):
    try:
        return ("return", function(*arguments))
    except Exception as error:
        return (type(error), str(error))


@pytest.mark.parametrize(
    ("first", "block_count", "capacity"),
    [
        (0, 0, 0),
        (0, 0, 5),
        (0, 1, 1),
        (0, 7, 3),
        (3, 7, 9),
        (6, 7, 4),
        (17, 29, 12),
        (INT64_MAX - 2, INT64_MAX, 4),
        (INT64_MAX, INT64_MAX, 3),
    ],
)
def test_exact_maximal_prefix_and_suffix_preservation(
    first, block_count, capacity
) -> None:
    sentinel = np.asarray(
        [INT64_MIN + index for index in range(capacity)], dtype=np.int64
    )
    primary_ids = sentinel.copy()
    count = fill_ascending_primary_prefix(first, block_count, primary_ids)
    expected = reference_prefix(first, block_count, capacity)
    assert type(count) is int
    assert count == len(expected)
    assert primary_ids[:count].tolist() == expected
    assert np.array_equal(primary_ids[count:], sentinel[count:])
    assert count == min(capacity, block_count - first)
    if first < block_count:
        assert count > 0
        assert count == capacity or first + count == block_count


def test_prior_prefix_contents_do_not_affect_result_or_suffix_bits() -> None:
    left = np.asarray(
        [INT64_MIN, -1, 0, 1, INT64_MAX, -17, 29], dtype=np.int64
    )
    right = np.asarray(
        [91, 92, 93, 94, 95, -17, 29], dtype=np.int64
    )
    left_before = left.copy()
    right_before = right.copy()
    assert fill_ascending_primary_prefix(11, 16, left) == 5
    assert fill_ascending_primary_prefix(11, 16, right) == 5
    assert left[:5].tolist() == right[:5].tolist() == [11, 12, 13, 14, 15]
    assert np.array_equal(left[5:], left_before[5:])
    assert np.array_equal(right[5:], right_before[5:])


def test_end_is_noop_but_unfinished_zero_capacity_is_atomic_error() -> None:
    for output in (
        np.empty(0, dtype=np.int64),
        np.asarray([INT64_MIN, -1, INT64_MAX], dtype=np.int64),
    ):
        before = output.copy()
        assert fill_ascending_primary_prefix(9, 9, output) == 0
        assert np.array_equal(output, before)

    empty = np.empty(0, dtype=np.int64)
    before = empty.copy()
    with pytest.raises(
        ValueError,
        match="^primary capacity must be positive before the end$",
    ):
        fill_ascending_primary_prefix(8, 9, empty)
    assert np.array_equal(empty, before)


@pytest.mark.parametrize(
    "integer_type",
    [int, np.int8, np.int32, np.int64, np.uint8, np.uint32, np.uint64],
)
def test_python_and_numpy_integer_scalars_are_accepted(integer_type) -> None:
    output = np.full(4, -1, dtype=np.int64)
    count = fill_ascending_primary_prefix(
        integer_type(2), integer_type(5), output
    )
    assert type(count) is int
    assert count == 3
    assert output.tolist() == [2, 3, 4, -1]


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(True), 1.0, np.float64(1.0), np.array(1, dtype=np.int64)],
)
def test_noninteger_scalar_kinds_have_exact_messages(invalid) -> None:
    output = np.full(1, -1, dtype=np.int64)
    before = output.copy()
    assert outcome(fill_ascending_primary_prefix, invalid, 1, output) == (
        TypeError,
        "first_primary_id must be an integer",
    )
    assert outcome(fill_ascending_primary_prefix, 0, invalid, output) == (
        TypeError,
        "block_count must be an integer",
    )
    assert np.array_equal(output, before)


def test_scalar_range_messages_and_int64_endpoints_are_exact() -> None:
    output = np.full(2, -7, dtype=np.int64)
    cases = [
        (
            (-1, 1, output),
            (ValueError, "first_primary_id must be non-negative"),
        ),
        (
            (0, -1, output),
            (ValueError, "block_count must be non-negative"),
        ),
        (
            (INT64_MAX + 1, INT64_MAX, output),
            (OverflowError, "first_primary_id does not fit in int64"),
        ),
        (
            (0, INT64_MAX + 1, output),
            (OverflowError, "block_count does not fit in int64"),
        ),
    ]
    for arguments, expected in cases:
        before = output.copy()
        assert outcome(fill_ascending_primary_prefix, *arguments) == expected
        assert np.array_equal(output, before)

    endpoint = np.full(3, -9, dtype=np.int64)
    count = fill_ascending_primary_prefix(
        np.uint64(INT64_MAX - 1),
        np.uint64(INT64_MAX),
        endpoint,
    )
    assert count == 1
    assert endpoint.tolist() == [INT64_MAX - 1, -9, -9]
    before = endpoint.copy()
    assert fill_ascending_primary_prefix(INT64_MAX, INT64_MAX, endpoint) == 0
    assert np.array_equal(endpoint, before)


def test_output_kind_dtype_rank_layout_and_writability_messages_are_exact() -> None:
    noncontiguous = np.arange(6, dtype=np.int64)[::2]
    readonly = np.full(3, -1, dtype=np.int64)
    readonly.setflags(write=False)
    nonnative_dtype = np.dtype(">i8" if np.little_endian else "<i8")
    cases = [
        (
            [1, 2, 3],
            TypeError,
            "primary_ids must be a NumPy array",
        ),
        (
            np.full(3, -1, dtype=np.int32),
            TypeError,
            "primary_ids must have dtype int64",
        ),
        (
            np.full(3, -1, dtype=nonnative_dtype),
            TypeError,
            "primary_ids must have dtype int64",
        ),
        (
            np.full((1, 3), -1, dtype=np.int64),
            ValueError,
            "primary_ids must be a C-contiguous vector",
        ),
        (
            noncontiguous,
            ValueError,
            "primary_ids must be a C-contiguous vector",
        ),
        (
            readonly,
            ValueError,
            "primary_ids must be writable",
        ),
    ]
    for output, error_type, message in cases:
        before = output.copy() if isinstance(output, np.ndarray) else list(output)
        assert outcome(fill_ascending_primary_prefix, 0, 3, output) == (
            error_type,
            message,
        )
        if isinstance(output, np.ndarray):
            assert np.array_equal(output, before)
        else:
            assert output == before


def test_validation_precedence_is_first_then_count_then_output_then_range() -> None:
    invalid_output = np.full(1, -1, dtype=np.int32)
    valid_output = np.full(1, -1, dtype=np.int64)
    cases = [
        (
            (True, True, invalid_output),
            (TypeError, "first_primary_id must be an integer"),
        ),
        (
            (0, True, invalid_output),
            (TypeError, "block_count must be an integer"),
        ),
        (
            (2, 1, invalid_output),
            (TypeError, "primary_ids must have dtype int64"),
        ),
        (
            (2, 1, valid_output),
            (ValueError, "first_primary_id exceeds block count"),
        ),
    ]
    for arguments, expected in cases:
        output = arguments[-1]
        before = output.copy()
        assert outcome(fill_ascending_primary_prefix, *arguments) == expected
        assert np.array_equal(output, before)

    readonly_end = np.full(1, -1, dtype=np.int64)
    readonly_end.setflags(write=False)
    assert outcome(fill_ascending_primary_prefix, 1, 1, readonly_end) == (
        ValueError,
        "primary_ids must be writable",
    )


def assert_full_traversal(block_count: int, capacity: int) -> None:
    output = np.full(capacity, INT64_MIN, dtype=np.int64)
    first = 0
    visited: list[int] = []
    while first < block_count:
        before = output.copy()
        count = fill_ascending_primary_prefix(first, block_count, output)
        expected = reference_prefix(first, block_count, capacity)
        assert count == len(expected) > 0
        assert output[:count].tolist() == expected
        assert np.array_equal(output[count:], before[count:])
        assert count == capacity or first + count == block_count
        visited.extend(output[:count].tolist())
        first += count
    before = output.copy()
    assert fill_ascending_primary_prefix(first, block_count, output) == 0
    assert np.array_equal(output, before)
    assert visited == list(range(block_count))


def test_exhaustive_small_full_traversals_are_exactly_once() -> None:
    assert_full_traversal(0, 0)
    for block_count in range(1, 65):
        for capacity in range(1, 17):
            assert_full_traversal(block_count, capacity)


def test_random_large_full_traversals_match_python_range_reference() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(200):
        block_count = int(rng.integers(1, 20_001))
        capacity = int(rng.integers(1, 1025))
        assert_full_traversal(block_count, capacity)


def test_arbitrary_face_values_are_ignored_by_legacy_false_wrapper() -> None:
    rng = np.random.default_rng(12001)
    block_count = 37
    faces = rng.integers(
        INT64_MIN,
        INT64_MAX,
        size=(block_count, 6),
        dtype=np.int64,
    )
    faces[0, 0] = INT64_MIN
    faces[0, 1] = INT64_MAX
    new_output = np.full(11, -91, dtype=np.int64)
    old_output = new_output.copy()
    new_count = fill_ascending_primary_prefix(13, block_count, new_output)
    old_counts = plan_level1_chunk(13, faces, False, old_output)
    assert old_counts == (new_count, new_count)
    assert np.array_equal(old_output, new_output)
    assert new_output[:new_count].tolist() == list(range(13, 24))


def test_random_old_new_result_conformance_and_suffix_identity() -> None:
    rng = np.random.default_rng(20260829)
    for _ in range(500):
        block_count = int(rng.integers(0, 513))
        first = int(rng.integers(0, block_count + 1))
        capacity = int(rng.integers(0 if first == block_count else 1, 129))
        faces = rng.integers(
            -10_000,
            10_001,
            size=(block_count, 6),
            dtype=np.int64,
        )
        initial = rng.integers(
            INT64_MIN,
            INT64_MAX,
            size=capacity,
            dtype=np.int64,
        )
        new_output = initial.copy()
        old_output = initial.copy()
        new_count = fill_ascending_primary_prefix(
            first, block_count, new_output
        )
        old_count = plan_level1_chunk(first, faces, False, old_output)
        assert old_count == (new_count, new_count)
        assert np.array_equal(old_output, new_output)
        assert np.array_equal(new_output[new_count:], initial[new_count:])


def test_common_legacy_scalar_errors_match_new_type_and_message() -> None:
    faces = np.full((3, 6), INT64_MAX, dtype=np.int64)
    output = np.full(2, -1, dtype=np.int64)
    for first in (True, np.bool_(False), -1, INT64_MAX + 1, 4):
        new = outcome(fill_ascending_primary_prefix, first, 3, output)
        old = outcome(plan_level1_chunk, first, faces, False, output)
        assert old == new

    empty = np.empty(0, dtype=np.int64)
    assert outcome(fill_ascending_primary_prefix, 0, 3, empty) == (
        ValueError,
        "primary capacity must be positive before the end",
    )
    assert outcome(plan_level1_chunk, 0, faces, False, empty) == (
        ValueError,
        "chunk capacity must be positive before the end",
    )


def test_wsp_capacity_composes_to_exact_bounded_primary_traversal() -> None:
    block_count = 257
    field_count = 2
    workspace_shape = i3(8, 8, 8)
    per_slot = workspace_nbytes(1, field_count, workspace_shape)
    budget = 7 * per_slot
    capacity = workspace_slot_capacity(
        budget,
        block_count,
        field_count,
        workspace_shape,
    )
    assert capacity == 7
    primary_ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty(
        (
            capacity,
            field_count,
            *(int(value) for value in workspace_shape),
        ),
        dtype=np.float64,
    )
    assert primary_ids.nbytes + payload.nbytes == budget
    first = 0
    chunks = 0
    while first < block_count:
        count = fill_ascending_primary_prefix(
            first, block_count, primary_ids
        )
        assert primary_ids[:count].tolist() == list(range(first, first + count))
        first += count
        chunks += 1
    assert first == block_count
    assert chunks == 37


def serial_sum(values: np.ndarray) -> np.float64:
    result = np.float64(0.0)
    for value in values:
        result = np.float64(result + np.float64(value))
    return result


def test_topology_free_red_stream_preserves_exact_ascending_addition_order() -> None:
    values = np.resize(
        np.asarray([1.0e16, 1.0, -1.0e16, 1.0], dtype=np.float64),
        65,
    )
    backing = values.reshape(65, 1, 1, 1, 1).copy()
    expected = serial_sum(values)
    results = []
    zero = i3(0, 0, 0)
    unit = i3(1, 1, 1)
    field_ids = i3(0)
    for capacity in (1, 2, 7, 31, 65):
        primary_ids = np.empty(capacity, dtype=np.int64)
        payload = np.empty((capacity, 1, 1, 1, 1), dtype=np.float64)
        accumulator = np.asarray([0.0], dtype=np.float64)
        first = 0
        visited: list[int] = []
        while first < backing.shape[0]:
            count = fill_ascending_primary_prefix(
                first, backing.shape[0], primary_ids
            )
            selected = primary_ids[:count]
            gather_blocks_into(
                backing,
                zero,
                unit,
                selected,
                field_ids,
                payload[:count],
                zero,
            )
            accumulate_field_sum(
                payload[:count],
                zero,
                unit,
                0,
                accumulator,
            )
            visited.extend(selected.tolist())
            first += count
        assert visited == list(range(backing.shape[0]))
        assert accumulator.view(np.uint64)[0] == expected.view(np.uint64)
        results.append(int(accumulator.view(np.uint64)[0]))
    assert len(set(results)) == 1
