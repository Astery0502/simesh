from __future__ import annotations

import numpy as np
import pytest

from simesh.amrvac import load_from_uniform
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from
from simesh_rewrite.storage_reference import (
    gather_blocks_into_reference,
    scatter_blocks_from_reference,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def patterned_storage(shape=(5, 4, 4, 5, 6)) -> np.ndarray:
    block, field, x, y, z = np.indices(shape, dtype=np.float64)
    return 10000.0 * block + 1000.0 * field + 100.0 * x + 10.0 * y + z


def test_gather_matches_reference_with_reordered_duplicate_selection() -> None:
    backing = patterned_storage()
    block_ids = np.array([3, 1, 3], dtype=np.int64)
    field_ids = np.array([2, 0, 2], dtype=np.int64)
    source_lower = i3(1, 1, 2)
    source_upper = i3(4, 5, 6)
    destination_lower = i3(2, 1, 3)
    destination = np.full((3, 3, 7, 7, 9), -17.0)
    expected = destination.copy()
    backing_before = backing.copy()
    gather_blocks_into_reference(
        backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        expected,
        destination_lower,
    )
    gather_blocks_into(
        backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )
    assert np.array_equal(destination.view(np.uint64), expected.view(np.uint64))
    assert np.array_equal(backing, backing_before)
    assert np.all(destination[:, :, :2] == -17.0)
    assert np.array_equal(destination[0, 0], destination[2, 2])


def test_scatter_matches_reference_and_preserves_unselected_cells() -> None:
    backing = np.full((5, 4, 6, 7, 8), -11.0)
    expected = backing.copy()
    source = patterned_storage((3, 2, 5, 6, 7))
    source_before = source.copy()
    block_ids = np.array([4, 1, 3], dtype=np.int64)
    field_ids = np.array([2, 0], dtype=np.int64)
    source_lower = i3(1, 2, 1)
    source_upper = i3(5, 6, 7)
    backing_lower = i3(2, 1, 2)
    scatter_blocks_from_reference(
        source,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        expected,
        backing_lower,
    )
    scatter_blocks_from(
        source,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        backing,
        backing_lower,
    )
    assert np.array_equal(backing.view(np.uint64), expected.view(np.uint64))
    assert np.array_equal(source, source_before)
    assert np.all(backing[0] == -11.0)
    assert np.all(backing[:, 1] == -11.0)


def test_scatter_duplicates_use_lexicographic_last_write() -> None:
    backing = np.zeros((2, 2, 1, 1, 1), dtype=np.float64)
    source = np.empty((2, 2, 1, 1, 1), dtype=np.float64)
    source[:, :, 0, 0, 0] = [[10.0, 11.0], [20.0, 21.0]]
    block_ids = np.array([1, 1], dtype=np.int64)
    field_ids = np.array([0, 0], dtype=np.int64)
    scatter_blocks_from(
        source,
        i3(0, 0, 0),
        i3(1, 1, 1),
        block_ids,
        field_ids,
        backing,
        i3(0, 0, 0),
    )
    assert backing[1, 0, 0, 0, 0] == 21.0


def test_empty_requests_still_validate_selectors_then_noop() -> None:
    backing = np.full((2, 2, 2, 2, 2), 3.0)
    destination = np.full((0, 1, 2, 2, 2), -5.0)
    gather_blocks_into(
        backing,
        i3(0, 0, 0),
        i3(2, 2, 2),
        np.empty(0, dtype=np.int64),
        np.array([1], dtype=np.int64),
        destination,
        i3(0, 0, 0),
    )
    assert destination.size == 0

    unchanged = backing.copy()
    empty_source = np.empty((1, 1, 2, 2, 2), dtype=np.float64)
    scatter_blocks_from(
        empty_source,
        i3(1, 0, 0),
        i3(1, 2, 2),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        backing,
        i3(0, 0, 0),
    )
    assert np.array_equal(backing, unchanged)

    with pytest.raises(ValueError, match="field_ids entry 0"):
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            i3(2, 2, 2),
            np.empty(0, dtype=np.int64),
            np.array([2], dtype=np.int64),
            destination,
            i3(0, 0, 0),
        )


def test_binary64_payload_bits_copy_exactly() -> None:
    patterns = np.array(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF8000000000001,
            0x7FF0000000000000,
        ],
        dtype=np.uint64,
    )
    backing = patterns.view(np.float64).reshape(1, 1, 1, 1, 4).copy()
    destination = np.empty_like(backing)
    gather_blocks_into(
        backing,
        i3(0, 0, 0),
        i3(1, 1, 4),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        destination,
        i3(0, 0, 0),
    )
    assert np.array_equal(destination.view(np.uint64).ravel(), patterns)

    sink = np.zeros_like(backing)
    scatter_blocks_from(
        destination,
        i3(0, 0, 0),
        i3(1, 1, 4),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        sink,
        i3(0, 0, 0),
    )
    assert np.array_equal(sink.view(np.uint64).ravel(), patterns)


def test_z_full_plane_fast_path_matches_reference() -> None:
    backing = patterned_storage((2, 2, 5, 6, 7))
    block_ids = np.array([1], dtype=np.int64)
    field_ids = np.array([1], dtype=np.int64)
    source_lower = i3(1, 2, 0)
    source_upper = i3(5, 5, 7)
    destination_lower = i3(0, 1, 0)
    destination = np.full((1, 1, 5, 5, 7), -3.0)
    expected = destination.copy()
    gather_blocks_into_reference(
        backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        expected,
        destination_lower,
    )
    gather_blocks_into(
        backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )
    assert np.array_equal(destination, expected)

    sink = np.full_like(backing, -9.0)
    expected_sink = sink.copy()
    scatter_blocks_from_reference(
        destination,
        destination_lower,
        destination_lower + (source_upper - source_lower),
        block_ids,
        field_ids,
        expected_sink,
        source_lower,
    )
    scatter_blocks_from(
        destination,
        destination_lower,
        destination_lower + (source_upper - source_lower),
        block_ids,
        field_ids,
        sink,
        source_lower,
    )
    assert np.array_equal(sink, expected_sink)


def test_current_level1_blocks_are_directly_compatible_backing() -> None:
    nx, ny, nz, nfield = 4, 8, 8, 3
    x = np.arange(nx, dtype=np.float64)[:, None, None, None]
    y = np.arange(ny, dtype=np.float64)[None, :, None, None]
    z = np.arange(nz, dtype=np.float64)[None, None, :, None]
    field = np.arange(nfield, dtype=np.float64)[None, None, None, :]
    uniform = 1000.0 * x + 100.0 * y + 10.0 * z + field
    dataset = load_from_uniform(
        uniform,
        ["a", "b", "c"],
        np.zeros(3),
        np.ones(3),
        np.array([2, 4, 4], dtype=np.int32),
    )
    backing = np.asarray(dataset.blocks())
    block_ids = np.array([7, 0, 3], dtype=np.int64)
    field_ids = np.array([2, 0], dtype=np.int64)
    gathered = np.empty((3, 2, 2, 4, 4), dtype=np.float64)
    gather_blocks_into(
        backing,
        i3(0, 0, 0),
        i3(2, 4, 4),
        block_ids,
        field_ids,
        gathered,
        i3(0, 0, 0),
    )
    assert np.array_equal(gathered, backing[block_ids[:, None], field_ids[None, :]])

    expected = backing.copy()
    replacement = gathered + 0.25
    for slot, block_id in enumerate(block_ids):
        for field_slot, field_id in enumerate(field_ids):
            expected[block_id, field_id] = replacement[slot, field_slot]
    scatter_blocks_from(
        replacement,
        i3(0, 0, 0),
        i3(2, 4, 4),
        block_ids,
        field_ids,
        backing,
        i3(0, 0, 0),
    )
    assert np.array_equal(backing, expected)


def test_invalid_request_is_atomic() -> None:
    backing = np.zeros((2, 2, 3, 3, 3), dtype=np.float64)
    destination = np.full((2, 1, 3, 3, 3), -1.0)
    before_destination = destination.copy()
    with pytest.raises(ValueError, match="block_ids entry 1"):
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            i3(3, 3, 3),
            np.array([0, 2], dtype=np.int64),
            np.array([0], dtype=np.int64),
            destination,
            i3(0, 0, 0),
        )
    assert np.array_equal(destination, before_destination)

    source = np.ones((1, 1, 2, 2, 2), dtype=np.float64)
    before_backing = backing.copy()
    with pytest.raises(ValueError, match="output spatial shape"):
        scatter_blocks_from(
            source,
            i3(0, 0, 0),
            i3(2, 2, 2),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            backing,
            i3(2, 2, 2),
        )
    assert np.array_equal(backing, before_backing)

    with pytest.raises(OverflowError, match="int64"):
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            i3(1, 1, 1),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.empty((1, 1, 2, 2, 2), dtype=np.float64),
            i3(np.iinfo(np.int64).max, 0, 0),
        )
    with pytest.raises(OverflowError, match="int64"):
        scatter_blocks_from(
            source,
            i3(0, 0, 0),
            i3(1, 1, 1),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            backing,
            i3(np.iinfo(np.int64).max, 0, 0),
        )
    assert np.array_equal(backing, before_backing)

    with pytest.raises(ValueError, match="must not overlap"):
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            i3(2, 2, 2),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            backing,
            i3(0, 0, 0),
        )

    with pytest.raises(ValueError, match="must not overlap"):
        scatter_blocks_from(
            backing,
            i3(0, 0, 0),
            i3(2, 2, 2),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            backing,
            i3(0, 0, 0),
        )
