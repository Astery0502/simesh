from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.format import open_memmap

from simesh_rewrite.access import (
    AccessPattern,
    required_input_region,
    validate_access_requirement,
)
from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.reductions import (
    accumulate_field_sum,
    finalize_field_sum,
    merge_field_sums,
)
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def bits(value: np.float64 | float) -> np.uint64:
    return np.float64(value).view(np.uint64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def manual_sum(
    payload: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    field: int,
    initial: float = 0.0,
) -> np.float64:
    total = np.float64(initial)
    with np.errstate(all="ignore"):
        for slot in range(payload.shape[0]):
            for i in range(int(lower[0]), int(upper[0])):
                for j in range(int(lower[1]), int(upper[1])):
                    for k in range(int(lower[2]), int(upper[2])):
                        total = np.float64(total + payload[slot, field, i, j, k])
    return total


def test_manual_slot_xyz_order_subregion_and_field_match_bitwise() -> None:
    payload = np.full((2, 2, 3, 2, 4), -77.0)
    lower = i3(1, 0, 1)
    upper = i3(3, 2, 4)
    stream = np.asarray(
        [1.0e16, 1.0, -1.0e16, 1.0] * 6,
        dtype=np.float64,
    )
    position = 0
    for slot in range(2):
        for i in range(1, 3):
            for j in range(2):
                for k in range(1, 4):
                    payload[slot, 1, i, j, k] = stream[position]
                    position += 1
    before = payload.copy()
    accumulator = np.array([2.5], dtype=np.float64)
    expected = manual_sum(payload, lower, upper, 1, initial=2.5)
    accumulate_field_sum(payload, lower, upper, 1, accumulator)
    assert bits(accumulator[0]) == bits(expected)
    assert np.array_equal(payload.view(np.uint64), before.view(np.uint64))


def test_persistent_state_is_chunk_boundary_invariant() -> None:
    payload = np.resize(
        np.asarray([1.0e16, 1.0, -1.0e16, 1.0]),
        16 * 2,
    ).reshape(16, 1, 2, 1, 1)
    expected = manual_sum(payload, i3(0, 0, 0), i3(2, 1, 1), 0)
    results = []
    for capacity in (1, 3, 7, 16):
        accumulator = np.array([0.0])
        first = 0
        while first < payload.shape[0]:
            stop = min(first + capacity, payload.shape[0])
            accumulate_field_sum(
                payload[first:stop],
                i3(0, 0, 0),
                i3(2, 1, 1),
                0,
                accumulator,
            )
            first = stop
        results.append(accumulator[0])
    assert all(bits(result) == bits(expected) for result in results)


def test_chunk_partial_merges_are_capacity_sensitive() -> None:
    values = np.asarray([1.0e16, 1.0, -1.0e16, 1.0]).reshape(4, 1, 1, 1, 1)
    persistent = np.array([0.0])
    accumulate_field_sum(values, i3(0, 0, 0), i3(1, 1, 1), 0, persistent)
    assert persistent[0] == 1.0

    merged = np.array([0.0])
    for chunk in (values[:2], values[2:]):
        partial = np.array([0.0])
        accumulate_field_sum(chunk, i3(0, 0, 0), i3(1, 1, 1), 0, partial)
        merge_field_sums(merged, partial)
    assert merged[0] == 0.0


def test_merge_tree_is_not_associative() -> None:
    left_tree = np.array([1.0e16])
    merge_field_sums(left_tree, np.array([-1.0e16]))
    merge_field_sums(left_tree, np.array([1.0]))

    right_partial = np.array([-1.0e16])
    merge_field_sums(right_partial, np.array([1.0]))
    right_tree = np.array([1.0e16])
    merge_field_sums(right_tree, right_partial)
    assert left_tree[0] == 1.0
    assert right_tree[0] == 0.0


def test_empty_nonzero_initial_and_signed_zero_states() -> None:
    payload = np.empty((0, 1, 2, 2, 2), dtype=np.float64)
    negative_zero = np.asarray([0x8000000000000000], dtype=np.uint64).view(np.float64)
    accumulate_field_sum(
        payload,
        i3(0, 0, 0),
        i3(2, 2, 2),
        0,
        negative_zero,
    )
    assert bits(negative_zero[0]) == np.uint64(0x8000000000000000)

    empty_region_payload = np.full((1, 1, 2, 2, 2), np.nan)
    state = np.array([3.25])
    accumulate_field_sum(
        empty_region_payload,
        i3(1, 0, 0),
        i3(1, 2, 2),
        0,
        state,
    )
    assert state[0] == 3.25

    negative_zeros = np.full((1, 1, 1, 1, 2), -0.0)
    positive_identity = np.array([0.0])
    accumulate_field_sum(
        negative_zeros,
        i3(0, 0, 0),
        i3(1, 1, 2),
        0,
        positive_identity,
    )
    assert bits(positive_identity[0]) == np.uint64(0)

    negative_identity = np.asarray(
        [0x8000000000000000], dtype=np.uint64
    ).view(np.float64)
    accumulate_field_sum(
        negative_zeros,
        i3(0, 0, 0),
        i3(1, 1, 2),
        0,
        negative_identity,
    )
    assert bits(negative_identity[0]) == np.uint64(0x8000000000000000)
    merge_field_sums(negative_identity, np.array([0.0]))
    assert bits(negative_identity[0]) == np.uint64(0)


def test_nonfinite_overflow_and_fixed_nan_sequence() -> None:
    sequences = [
        ([np.inf, -np.inf], "nan"),
        ([np.finfo(float).max, np.finfo(float).max, -np.inf], "nan"),
        ([np.inf, 1.0], "+inf"),
        ([-np.inf, 1.0], "-inf"),
        ([np.nan, 1.0, 2.0], "nan"),
    ]
    for raw, classification in sequences:
        payload = np.asarray(raw, dtype=np.float64).reshape(1, 1, 1, 1, -1)
        accumulator = np.array([0.0])
        with np.errstate(all="ignore"):
            accumulate_field_sum(
                payload,
                i3(0, 0, 0),
                i3(1, 1, len(raw)),
                0,
                accumulator,
            )
        expected = manual_sum(payload, i3(0, 0, 0), i3(1, 1, len(raw)), 0)
        if classification == "nan":
            assert np.isnan(accumulator[0])
            assert np.isnan(expected)
        elif classification == "+inf":
            assert np.isposinf(accumulator[0])
        else:
            assert np.isneginf(accumulator[0])


def test_invalid_field_region_state_writability_and_overlap_are_atomic() -> None:
    payload = np.arange(8, dtype=np.float64).reshape(1, 1, 2, 2, 2)
    state = np.array([7.0])
    before = state.copy()
    with pytest.raises(ValueError, match="field axis"):
        accumulate_field_sum(payload, i3(0, 0, 0), i3(2, 2, 2), 1, state)
    assert_bits_equal(state, before)
    with pytest.raises(ValueError, match="valid region"):
        accumulate_field_sum(payload, i3(1, 0, 0), i3(0, 2, 2), 0, state)
    assert_bits_equal(state, before)

    bad_shape = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="one-value"):
        accumulate_field_sum(
            payload,
            i3(0, 0, 0),
            i3(2, 2, 2),
            0,
            bad_shape,
        )
    readonly = state.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        accumulate_field_sum(
            payload,
            i3(0, 0, 0),
            i3(2, 2, 2),
            0,
            readonly,
        )

    overlap_payload = np.zeros((1, 1, 1, 1, 1), dtype=np.float64)
    overlap_state = overlap_payload.reshape(-1)
    overlap_before = overlap_payload.copy()
    with pytest.raises(ValueError, match="overlap"):
        accumulate_field_sum(
            overlap_payload,
            i3(0, 0, 0),
            i3(1, 1, 1),
            0,
            overlap_state,
        )
    assert_bits_equal(overlap_payload, overlap_before)

    metadata_base = np.zeros(3, dtype=np.float64)
    metadata_state = metadata_base[:1]
    metadata_lower = metadata_base.view(np.int64)
    with pytest.raises(ValueError, match="overlap"):
        accumulate_field_sum(
            overlap_payload,
            metadata_lower,
            i3(1, 1, 1),
            0,
            metadata_state,
        )

    merge_state = np.array([2.0])
    merge_before = merge_state.copy()
    with pytest.raises(ValueError, match="overlap"):
        merge_field_sums(merge_state, merge_state)
    assert_bits_equal(merge_state, merge_before)
    with pytest.raises(ValueError, match="one-value"):
        merge_field_sums(merge_state, np.array([1.0, 2.0]))


def test_readonly_finalize_preserves_state_and_python_value() -> None:
    state = np.asarray([0x8000000000000000], dtype=np.uint64).view(np.float64)
    state.setflags(write=False)
    before = state.view(np.uint64).copy()
    result = finalize_field_sum(state)
    assert bits(result) == np.uint64(0x8000000000000000)
    assert np.array_equal(state.view(np.uint64), before)


def test_fnd_streaming_reduction_has_zero_reach() -> None:
    zero = i3(0, 0, 0)
    validate_access_requirement(AccessPattern.STREAMING_REDUCTION, zero, zero)
    lower, upper = required_input_region(
        i3(2, 3, 4),
        i3(7, 9, 11),
        zero,
        zero,
    )
    assert np.array_equal(lower, i3(2, 3, 4))
    assert np.array_equal(upper, i3(7, 9, 11))
    with pytest.raises(ValueError, match="zero reach"):
        validate_access_requirement(
            AccessPattern.STREAMING_REDUCTION,
            i3(1, 0, 0),
            zero,
        )


def test_math_fsum_and_numpy_are_descriptive_accuracy_references() -> None:
    values = np.asarray([1.0e16] + [1.0] * 1000 + [-1.0e16])
    payload = values.reshape(1, 1, 1, 1, -1)
    accumulator = np.array([0.0])
    accumulate_field_sum(
        payload,
        i3(0, 0, 0),
        i3(1, 1, len(values)),
        0,
        accumulator,
    )
    accurate = math.fsum(values.tolist())
    numpy_result = float(np.sum(values, dtype=np.float64))
    assert accumulator[0] == 0.0
    assert accurate == 1000.0
    assert np.isfinite(numpy_result)
    assert abs(numpy_result - accurate) < abs(float(accumulator[0]) - accurate)


def run_memmap_reduction(
    backing: np.ndarray,
    faces: np.ndarray,
    capacity: int,
    *,
    merge_partials: bool,
) -> np.float64:
    block_shape = i3(*backing.shape[2:])
    block_ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty((capacity, 1, *backing.shape[2:]), dtype=np.float64)
    accumulator = np.array([0.0])
    first = 0
    while first < backing.shape[0]:
        primary_count, selected_count = plan_level1_chunk(
            first,
            faces,
            False,
            block_ids,
        )
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            block_shape,
            block_ids[:selected_count],
            np.array([0], dtype=np.int64),
            payload[:selected_count],
            i3(0, 0, 0),
        )
        if merge_partials:
            partial = np.array([0.0])
            accumulate_field_sum(
                payload[:primary_count],
                i3(0, 0, 0),
                block_shape,
                0,
                partial,
            )
            merge_field_sums(accumulator, partial)
        else:
            accumulate_field_sum(
                payload[:primary_count],
                i3(0, 0, 0),
                block_shape,
                0,
                accumulator,
            )
        first += primary_count
    return accumulator[0]


def test_readonly_memmap_bounded_composition_across_capacities(tmp_path: Path) -> None:
    block_count = 257
    block_shape = (2, 2, 2)
    value_count = block_count * int(np.prod(block_shape))
    values = np.resize(
        np.asarray([1.0e16, 1.0, -1.0e16, 1.0]),
        value_count,
    )
    path = tmp_path / "reduction.npy"
    writable = open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(block_count, 1, *block_shape),
    )
    writable[:, 0] = values.reshape(block_count, *block_shape)
    writable.flush()
    del writable
    backing = np.load(path, mmap_mode="r")
    assert isinstance(backing, np.memmap)
    assert not backing.flags.writeable

    root_shape = i3(block_count, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    persistent = []
    partial = []
    for capacity in (1, 7, 31, 257):
        persistent.append(
            run_memmap_reduction(
                backing,
                faces,
                capacity,
                merge_partials=False,
            )
        )
        partial.append(
            run_memmap_reduction(
                backing,
                faces,
                capacity,
                merge_partials=True,
            )
        )
    assert [float(value) for value in persistent] == [1.0, 1.0, 1.0, 1.0]
    assert [float(value) for value in partial] == [257.0, 37.0, 9.0, 1.0]
    assert math.fsum(values.tolist()) == 1028.0
    assert np.isfinite(np.sum(backing[:, 0], dtype=np.float64))
