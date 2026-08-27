from __future__ import annotations

import numpy as np
import pytest

from simesh.amrvac import load_from_uniform
from simesh_rewrite.foundation import (
    copy_region_into,
    interior_region,
    ravel_cell,
    unravel_cell,
)
from simesh_rewrite.foundation_reference import (
    copy_region_into as copy_region_into_reference,
)
from simesh_rewrite.foundation_reference import (
    interior_region as interior_region_reference,
)
from simesh_rewrite.foundation_reference import ravel_cell as ravel_cell_reference
from simesh_rewrite.foundation_reference import unravel_cell as unravel_cell_reference


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def test_axis_order_and_exact_c_offset() -> None:
    shape = (2, 3, 4, 5, 6)
    payload = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    slot, field, i, j, k = 1, 2, 3, 4, 5
    offset = ((((slot * shape[1] + field) * shape[2] + i) * shape[3] + j) * shape[4] + k)
    assert payload[slot, field, i, j, k] == payload.ravel(order="C")[offset]


def test_ravel_and_unravel_match_reference_exactly() -> None:
    shape = i3(4, 5, 7)
    for index_tuple in np.ndindex(tuple(int(value) for value in shape)):
        index = i3(*index_tuple)
        expected = ravel_cell_reference(index, shape)
        assert ravel_cell(index, shape) == expected
        assert unravel_cell(expected, shape) == index_tuple
        assert unravel_cell_reference(expected, shape) == index_tuple


def test_asymmetric_interior_region_matches_reference() -> None:
    interior_shape = i3(4, 6, 8)
    lower_halo = i3(1, 3, 2)
    actual = interior_region(interior_shape, lower_halo)
    expected = interior_region_reference(interior_shape, lower_halo)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert tuple(actual[0]) == (1, 3, 2)
    assert tuple(actual[1]) == (5, 9, 10)


def test_index_arithmetic_rejects_int64_overflow_in_both_paths() -> None:
    maximum = np.iinfo(np.int64).max
    with pytest.raises(OverflowError):
        interior_region(i3(1, 0, 0), i3(maximum, 0, 0))
    with pytest.raises(OverflowError):
        interior_region_reference(i3(1, 0, 0), i3(maximum, 0, 0))

    oversized_shape = i3(maximum, 2, 1)
    with pytest.raises(OverflowError):
        ravel_cell(i3(0, 0, 0), oversized_shape)
    with pytest.raises(OverflowError):
        ravel_cell_reference(i3(0, 0, 0), oversized_shape)
    with pytest.raises(OverflowError):
        unravel_cell(0, oversized_shape)
    with pytest.raises(OverflowError):
        unravel_cell_reference(0, oversized_shape)


def test_copy_region_matches_reference_without_invalid_reads() -> None:
    source = np.full((2, 3, 8, 10, 12), np.nan, dtype=np.float64)
    source_lower = i3(1, 3, 2)
    destination_lower = i3(2, 1, 4)
    extent = i3(4, 5, 6)

    source_region = (
        slice(None),
        slice(None),
        slice(1, 5),
        slice(3, 8),
        slice(2, 8),
    )
    values = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float64).reshape(2, 3, 4, 5, 6)
    source[source_region] = values
    source_before = source.copy()

    destination = np.full((2, 3, 9, 9, 13), -17.0, dtype=np.float64)
    expected = destination.copy()
    copy_region_into_reference(
        source,
        source_lower,
        expected,
        destination_lower,
        extent,
    )
    copy_region_into(
        source,
        source_lower,
        destination,
        destination_lower,
        extent,
    )

    assert np.array_equal(destination, expected)
    assert np.array_equal(source, source_before, equal_nan=True)
    assert np.all(np.isfinite(destination[:, :, 2:6, 1:6, 4:10]))
    assert np.all(destination[:, :, :2, :, :] == -17.0)


def test_slot_is_distinct_from_global_block_id() -> None:
    block_ids = np.array([7, 0, 12], dtype=np.int64)
    payload = np.zeros((3, 1, 1, 1, 1), dtype=np.float64)
    payload[:, 0, 0, 0, 0] = [70.0, 0.0, 120.0]
    by_global_id = {
        int(block_id): payload[slot, 0, 0, 0, 0]
        for slot, block_id in enumerate(block_ids)
    }
    assert by_global_id == {7: 70.0, 0: 0.0, 12: 120.0}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (np.zeros((1, 1, 2, 2, 2), dtype=np.float32), "dtype float64"),
        (np.zeros((1, 1, 2, 2), dtype=np.float64), "rank 4"),
        (np.asfortranarray(np.zeros((1, 1, 2, 2, 2))), "C-contiguous"),
    ],
)
def test_copy_rejects_noncanonical_payloads(source: np.ndarray, message: str) -> None:
    destination = np.zeros((1, 1, 2, 2, 2), dtype=np.float64)
    with pytest.raises((TypeError, ValueError), match=message):
        copy_region_into(source, i3(0, 0, 0), destination, i3(0, 0, 0), i3(1, 1, 1))


def test_copy_rejects_invalid_bounds_and_overlap() -> None:
    source = np.zeros((1, 1, 3, 4, 5), dtype=np.float64)
    destination = np.zeros_like(source)
    with pytest.raises(ValueError, match="source region"):
        copy_region_into(source, i3(2, 3, 4), destination, i3(0, 0, 0), i3(2, 2, 2))
    with pytest.raises(ValueError, match="must not overlap"):
        copy_region_into(source, i3(0, 0, 0), source, i3(0, 0, 0), i3(1, 1, 1))


def test_copy_accepts_read_only_input_and_zero_extent() -> None:
    source = np.arange(24, dtype=np.float64).reshape(1, 1, 2, 3, 4)
    source.setflags(write=False)
    destination = np.full_like(source, -1.0)
    copy_region_into(source, i3(0, 0, 0), destination, i3(0, 0, 0), i3(2, 3, 4))
    assert np.array_equal(destination, source)

    unchanged = destination.copy()
    copy_region_into(source, i3(2, 3, 4), destination, i3(2, 3, 4), i3(0, 0, 0))
    assert np.array_equal(destination, unchanged)


def test_current_level1_blocks_feed_canonical_payload_exactly() -> None:
    nx, ny, nz, nfield = 4, 8, 8, 2
    x = np.arange(nx, dtype=np.float64)[:, None, None, None]
    y = np.arange(ny, dtype=np.float64)[None, :, None, None]
    z = np.arange(nz, dtype=np.float64)[None, None, :, None]
    field = np.arange(nfield, dtype=np.float64)[None, None, None, :]
    uniform = 1000.0 * x + 100.0 * y + 10.0 * z + field
    dataset = load_from_uniform(
        uniform,
        ["a", "b"],
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([2, 4, 4], dtype=np.int32),
    )
    current_blocks = np.asarray(dataset.blocks())

    assert current_blocks.shape == (8, 2, 2, 4, 4)
    assert current_blocks.dtype == np.float64
    assert current_blocks.flags.c_contiguous
    assert np.all(np.diff(current_blocks, axis=1) == 1.0)
    assert np.all(np.diff(current_blocks, axis=2) == 1000.0)
    assert np.all(np.diff(current_blocks, axis=3) == 100.0)
    assert np.all(np.diff(current_blocks, axis=4) == 10.0)

    padded = np.full((8, 2, 5, 7, 9), np.nan, dtype=np.float64)
    copy_region_into(
        current_blocks,
        i3(0, 0, 0),
        padded,
        i3(1, 2, 3),
        i3(2, 4, 4),
    )
    recovered = np.empty_like(current_blocks)
    copy_region_into_reference(
        padded,
        i3(1, 2, 3),
        recovered,
        i3(0, 0, 0),
        i3(2, 4, 4),
    )
    assert np.array_equal(recovered, current_blocks)
