from __future__ import annotations

import numpy as np
import pytest

import simesh_rewrite
from simesh_rewrite import chunking
from simesh_rewrite.workspace import (
    workspace_nbytes,
    workspace_slot_capacity,
)


INT64_MAX = int(np.iinfo(np.int64).max)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def reference_nbytes(slot_capacity: int, field_count: int, shape) -> int:
    volume = 1
    for extent in shape:
        volume *= int(extent)
    return int(slot_capacity) * 8 * (int(field_count) * volume + 1)


def reference_capacity(
    budget_bytes: int,
    block_count: int,
    field_count: int,
    shape,
) -> int:
    per_slot = reference_nbytes(1, field_count, shape)
    return min(int(block_count), int(budget_bytes) // per_slot)


def forward_entrypoints():
    return (
        workspace_nbytes,
        chunking.workspace_nbytes,
        simesh_rewrite.workspace_nbytes,
    )


def inverse_entrypoints():
    return (
        workspace_slot_capacity,
        chunking.workspace_slot_capacity,
        simesh_rewrite.workspace_slot_capacity,
    )


def outcome(function, *arguments):
    try:
        return ("return", function(*arguments))
    except Exception as error:
        return (type(error), str(error))


def assert_forward_conformance(arguments, expected) -> None:
    for function in forward_entrypoints():
        assert outcome(function, *arguments) == expected


def assert_inverse_conformance(arguments, expected) -> None:
    for function in inverse_entrypoints():
        assert outcome(function, *arguments) == expected


@pytest.mark.parametrize(
    ("slot_capacity", "field_count", "shape"),
    [
        (0, 1, i3(1, 1, 1)),
        (1, 1, i3(1, 1, 1)),
        (7, 3, i3(10, 12, 14)),
        (19, 5, i3(2, 3, 7)),
        (23, 2, i3(1, 17, 4)),
    ],
)
def test_forward_formula_is_exact_and_linear(slot_capacity, field_count, shape) -> None:
    expected = reference_nbytes(slot_capacity, field_count, shape)
    assert workspace_nbytes(slot_capacity, field_count, shape) == expected
    assert workspace_nbytes(slot_capacity + 1, field_count, shape) - expected == (
        reference_nbytes(1, field_count, shape)
    )


def test_inverse_is_exact_maximal_and_block_clamped_at_byte_boundaries() -> None:
    shape = i3(10, 12, 14)
    per_slot = reference_nbytes(1, 3, shape)
    block_count = 100
    for requested in (0, 1, 2, 7, 31, block_count):
        exact_budget = requested * per_slot
        assert workspace_slot_capacity(
            exact_budget, block_count, 3, shape
        ) == requested
        if requested:
            assert workspace_slot_capacity(
                exact_budget - 1, block_count, 3, shape
            ) == requested - 1

    assert workspace_slot_capacity(200 * per_slot, block_count, 3, shape) == 100
    maximum = workspace_slot_capacity(
        INT64_MAX,
        INT64_MAX,
        1,
        i3(1, 1, 1),
    )
    assert maximum == INT64_MAX // 16
    assert workspace_nbytes(maximum, 1, i3(1, 1, 1)) <= INT64_MAX


def test_forward_and_inverse_are_monotone_in_owned_scalar_inputs() -> None:
    shape = i3(3, 4, 5)
    forward = [workspace_nbytes(slots, 2, shape) for slots in range(40)]
    assert forward == sorted(forward)
    per_slot = workspace_nbytes(1, 2, shape)
    budgets = [0, 1, per_slot - 1, per_slot, 2 * per_slot, INT64_MAX]
    capacities = [
        workspace_slot_capacity(budget, 1000, 2, shape) for budget in budgets
    ]
    assert capacities == sorted(capacities)
    block_limits = [0, 1, 7, 31, 1000]
    clamped = [
        workspace_slot_capacity(INT64_MAX, limit, 2, shape)
        for limit in block_limits
    ]
    assert clamped == block_limits
    assert workspace_slot_capacity(per_slot * 20, 100, 2, i3(3, 4, 6)) <= (
        workspace_slot_capacity(per_slot * 20, 100, 2, shape)
    )


def test_zero_capacity_budget_and_block_count_still_validate_descriptor() -> None:
    shape = i3(2, 3, 4)
    assert workspace_nbytes(0, 2, shape) == 0
    assert workspace_slot_capacity(0, 10, 2, shape) == 0
    assert workspace_slot_capacity(INT64_MAX, 0, 2, shape) == 0

    assert outcome(workspace_nbytes, 0, 0, shape) == (
        ValueError,
        "field_count must be positive",
    )
    assert outcome(workspace_nbytes, 0, 1, i3(1, 0, 1)) == (
        ValueError,
        "workspace_shape entries must be positive",
    )
    assert outcome(workspace_slot_capacity, 0, 0, 0, shape) == (
        ValueError,
        "field_count must be positive",
    )
    assert outcome(
        workspace_slot_capacity, 0, 0, 1, i3(1, -1, 1)
    ) == (
        ValueError,
        "workspace_shape entries must be positive",
    )


@pytest.mark.parametrize(
    "integer_type",
    [int, np.int8, np.int32, np.int64, np.uint8, np.uint32, np.uint64],
)
def test_python_and_numpy_integer_scalar_kinds_are_accepted(integer_type) -> None:
    shape = i3(2, 3, 4)
    forward = workspace_nbytes(integer_type(3), integer_type(2), shape)
    inverse = workspace_slot_capacity(
        integer_type(100), integer_type(9), integer_type(2), shape
    )
    assert type(forward) is int
    assert type(inverse) is int
    assert forward == reference_nbytes(3, 2, shape)
    assert inverse == reference_capacity(100, 9, 2, shape)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(True), 1.0, np.float64(1.0), np.array(1, dtype=np.int64)],
)
def test_noninteger_scalar_kinds_have_exact_legacy_errors(invalid) -> None:
    shape = i3(1, 1, 1)
    assert_forward_conformance(
        (invalid, 1, shape),
        (TypeError, "slot_capacity must be an integer"),
    )
    assert_forward_conformance(
        (1, invalid, shape),
        (TypeError, "field_count must be an integer"),
    )
    assert_inverse_conformance(
        (invalid, 1, 1, shape),
        (TypeError, "budget_bytes must be an integer"),
    )
    assert_inverse_conformance(
        (1, invalid, 1, shape),
        (TypeError, "block_count must be an integer"),
    )
    assert_inverse_conformance(
        (1, 1, invalid, shape),
        (TypeError, "field_count must be an integer"),
    )


def test_scalar_range_and_positivity_messages_are_exact() -> None:
    shape = i3(1, 1, 1)
    assert_forward_conformance(
        (-1, 1, shape),
        (ValueError, "slot_capacity must be non-negative"),
    )
    assert_forward_conformance(
        (0, -1, shape),
        (ValueError, "field_count must be non-negative"),
    )
    assert_forward_conformance(
        (0, 0, shape),
        (ValueError, "field_count must be positive"),
    )
    assert_inverse_conformance(
        (-1, 1, 1, shape),
        (ValueError, "budget_bytes must be non-negative"),
    )
    assert_inverse_conformance(
        (0, -1, 1, shape),
        (ValueError, "block_count must be non-negative"),
    )


def test_shape_kind_dtype_rank_layout_and_extent_messages_are_exact() -> None:
    noncontiguous = np.arange(6, dtype=np.int64)[::2]
    nonnative_dtype = np.dtype(">i8" if np.little_endian else "<i8")
    cases = [
        ([1, 1, 1], TypeError, "workspace_shape must be a NumPy array"),
        (
            np.asarray([1, 1, 1], dtype=np.int32),
            TypeError,
            "workspace_shape must have dtype int64",
        ),
        (
            np.asarray([1, 1, 1], dtype=nonnative_dtype),
            TypeError,
            "workspace_shape must have dtype int64",
        ),
        (
            np.asarray(1, dtype=np.int64),
            ValueError,
            "workspace_shape must have shape (3,), got ()",
        ),
        (
            np.asarray([1, 1], dtype=np.int64),
            ValueError,
            "workspace_shape must have shape (3,), got (2,)",
        ),
        (
            noncontiguous,
            ValueError,
            "workspace_shape must be C-contiguous",
        ),
        (
            i3(1, 0, 1),
            ValueError,
            "workspace_shape entries must be positive",
        ),
    ]
    for shape, error_type, message in cases:
        assert_forward_conformance((1, 1, shape), (error_type, message))
        assert_inverse_conformance((100, 10, 1, shape), (error_type, message))


def test_validation_precedence_is_stable_for_multiply_invalid_inputs() -> None:
    bad_shape = [0, 0, 0]
    forward_cases = [
        (
            (True, True, bad_shape),
            (TypeError, "slot_capacity must be an integer"),
        ),
        (
            (0, True, bad_shape),
            (TypeError, "field_count must be an integer"),
        ),
        (
            (0, 0, bad_shape),
            (ValueError, "field_count must be positive"),
        ),
        (
            (0, 1, bad_shape),
            (TypeError, "workspace_shape must be a NumPy array"),
        ),
    ]
    inverse_cases = [
        (
            (True, True, True, bad_shape),
            (TypeError, "budget_bytes must be an integer"),
        ),
        (
            (0, True, True, bad_shape),
            (TypeError, "block_count must be an integer"),
        ),
        (
            (0, 0, True, bad_shape),
            (TypeError, "field_count must be an integer"),
        ),
        (
            (0, 0, 0, bad_shape),
            (ValueError, "field_count must be positive"),
        ),
        (
            (0, 0, 1, bad_shape),
            (TypeError, "workspace_shape must be a NumPy array"),
        ),
    ]
    for arguments, expected in forward_cases:
        assert_forward_conformance(arguments, expected)
    for arguments, expected in inverse_cases:
        assert_inverse_conformance(arguments, expected)


def test_scalar_spatial_per_slot_and_total_overflow_stages_are_exact() -> None:
    unit = i3(1, 1, 1)
    assert_forward_conformance(
        (INT64_MAX + 1, 1, unit),
        (OverflowError, "slot_capacity does not fit in int64"),
    )
    assert_forward_conformance(
        (0, INT64_MAX + 1, unit),
        (OverflowError, "field_count does not fit in int64"),
    )
    assert_inverse_conformance(
        (INT64_MAX + 1, 1, 1, unit),
        (OverflowError, "budget_bytes does not fit in int64"),
    )
    assert_inverse_conformance(
        (0, INT64_MAX + 1, 1, unit),
        (OverflowError, "block_count does not fit in int64"),
    )

    volume_overflow = i3(INT64_MAX, 2, 1)
    assert_forward_conformance(
        (0, 1, volume_overflow),
        (OverflowError, "workspace spatial volume does not fit in int64"),
    )
    assert_inverse_conformance(
        (0, 0, 1, volume_overflow),
        (OverflowError, "workspace spatial volume does not fit in int64"),
    )

    per_slot_overflow = i3(INT64_MAX // 8, 1, 1)
    assert_forward_conformance(
        (0, 1, per_slot_overflow),
        (OverflowError, "workspace bytes per slot do not fit in int64"),
    )
    assert_inverse_conformance(
        (0, 0, 1, per_slot_overflow),
        (OverflowError, "workspace bytes per slot do not fit in int64"),
    )

    first_total_overflow = INT64_MAX // 16 + 1
    assert_forward_conformance(
        (first_total_overflow, 1, unit),
        (OverflowError, "managed workspace bytes do not fit in int64"),
    )


def test_read_only_shape_is_accepted_and_never_mutated() -> None:
    shape = i3(7, 5, 3)
    before = shape.copy()
    shape.setflags(write=False)
    expected_bytes = reference_nbytes(11, 4, shape)
    expected_capacity = reference_capacity(expected_bytes, 99, 4, shape)
    assert workspace_nbytes(11, 4, shape) == expected_bytes
    assert workspace_slot_capacity(expected_bytes, 99, 4, shape) == (
        expected_capacity
    )
    assert np.array_equal(shape, before)
    assert not shape.flags.writeable


@pytest.mark.parametrize(
    ("slot_capacity", "field_count", "shape"),
    [
        (0, 2, i3(3, 4, 5)),
        (1, 1, i3(1, 1, 1)),
        (7, 3, i3(4, 5, 6)),
        (31, 2, i3(8, 7, 5)),
    ],
)
def test_formula_equals_actual_numpy_payload_and_id_nbytes(
    slot_capacity, field_count, shape
) -> None:
    payload = np.empty(
        (slot_capacity, field_count, *(int(value) for value in shape)),
        dtype=np.float64,
    )
    block_ids = np.empty(slot_capacity, dtype=np.int64)
    actual = payload.nbytes + block_ids.nbytes
    assert workspace_nbytes(slot_capacity, field_count, shape) == actual
    capacity = workspace_slot_capacity(actual, 10_000, field_count, shape)
    assert capacity == slot_capacity
    if slot_capacity < 10_000:
        assert workspace_nbytes(capacity + 1, field_count, shape) > actual


def test_randomized_canonical_legacy_and_arbitrary_integer_reference_agree() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(250):
        shape = rng.integers(1, 65, size=3, dtype=np.int64)
        fields = int(rng.integers(1, 17))
        slots = int(rng.integers(0, 513))
        blocks = int(rng.integers(0, 1025))
        per_slot = reference_nbytes(1, fields, shape)
        budget = int(rng.integers(0, 2049)) * per_slot + int(
            rng.integers(0, per_slot)
        )
        expected_bytes = reference_nbytes(slots, fields, shape)
        expected_capacity = reference_capacity(
            budget, blocks, fields, shape
        )
        assert_forward_conformance(
            (slots, fields, shape), ("return", expected_bytes)
        )
        assert_inverse_conformance(
            (budget, blocks, fields, shape),
            ("return", expected_capacity),
        )


def test_legacy_wrappers_match_canonical_error_type_and_message() -> None:
    shape = i3(1, 1, 1)
    forward_errors = [
        (-1, 1, shape),
        (0, 0, shape),
        (0, 1, i3(0, 1, 1)),
        (INT64_MAX // 16 + 1, 1, shape),
    ]
    inverse_errors = [
        (-1, 1, 1, shape),
        (0, -1, 1, shape),
        (0, 0, 0, shape),
        (0, 0, 1, i3(INT64_MAX, 2, 1)),
    ]
    for arguments in forward_errors:
        expected = outcome(workspace_nbytes, *arguments)
        assert expected[0] != "return"
        assert_forward_conformance(arguments, expected)
    for arguments in inverse_errors:
        expected = outcome(workspace_slot_capacity, *arguments)
        assert expected[0] != "return"
        assert_inverse_conformance(arguments, expected)


def test_int001_richer_workspace_formula_remains_explicitly_separate() -> None:
    slot_capacity = 7
    field_count = 3
    padded_shape = i3(10, 12, 14)
    block_shape = i3(8, 10, 12)
    payload = np.empty(
        (
            slot_capacity,
            field_count,
            *(int(value) for value in padded_shape),
        ),
        dtype=np.float64,
    )
    block_ids = np.empty(slot_capacity, dtype=np.int64)
    output_workspace = np.empty(
        (slot_capacity, 1, *(int(value) for value in block_shape)),
        dtype=np.float64,
    )
    reduction_state = np.empty(1, dtype=np.float64)

    wsp_bytes = workspace_nbytes(slot_capacity, field_count, padded_shape)
    int_bytes = (
        payload.nbytes
        + block_ids.nbytes
        + output_workspace.nbytes
        + reduction_state.nbytes
    )
    block_volume = int(np.prod(block_shape, dtype=np.int64))
    padded_volume = int(np.prod(padded_shape, dtype=np.int64))
    expected_int = 8 * slot_capacity * (
        field_count * padded_volume + block_volume + 1
    ) + 8
    assert wsp_bytes == payload.nbytes + block_ids.nbytes
    assert int_bytes == expected_int
    assert int_bytes == wsp_bytes + output_workspace.nbytes + 8
    assert int_bytes > wsp_bytes
