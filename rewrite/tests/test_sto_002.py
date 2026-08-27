from __future__ import annotations

from numpy.lib.format import open_memmap
import numpy as np
import pytest

from simesh_rewrite.chunking import (
    minimum_face_closed_slots,
    plan_level1_chunk,
    workspace_nbytes,
    workspace_slot_capacity,
)
from simesh_rewrite.chunking_reference import plan_level1_chunk_reference
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def topology(shape: tuple[int, int, int]):
    root_shape = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    return level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)


def test_workspace_budget_is_exact_at_byte_boundary() -> None:
    shape = i3(10, 12, 14)
    per_slot = workspace_nbytes(1, 3, shape)
    assert per_slot == 8 * (3 * 10 * 12 * 14 + 1)
    assert workspace_slot_capacity(per_slot * 7 - 1, 100, 3, shape) == 6
    assert workspace_slot_capacity(per_slot * 7, 100, 3, shape) == 7
    assert workspace_nbytes(7, 3, shape) == per_slot * 7
    assert workspace_slot_capacity(per_slot * 200, 100, 3, shape) == 100


def test_no_closure_chunks_cover_every_primary_once() -> None:
    neighbors = topology((5, 3, 2))
    ids = np.full(7, -99, dtype=np.int64)
    first = 0
    covered = []
    while first < neighbors.shape[0]:
        primary_count, selected_count = plan_level1_chunk(
            first,
            neighbors,
            False,
            ids,
        )
        assert selected_count == primary_count
        assert np.array_equal(
            ids[:primary_count],
            np.arange(first, first + primary_count),
        )
        covered.extend(ids[:primary_count].tolist())
        first += primary_count
    assert covered == list(range(neighbors.shape[0]))
    tail_before = ids.copy()
    assert plan_level1_chunk(first, neighbors, False, ids) == (0, 0)
    assert np.array_equal(ids, tail_before)


def test_face_closed_plan_matches_reference_and_is_maximal() -> None:
    neighbors = topology((4, 3, 2))
    for capacity in (4, 7, 10, 16):
        ids = np.full(capacity, -77, dtype=np.int64)
        expected_error = None
        try:
            values, expected_primary, expected_selected = plan_level1_chunk_reference(
                0,
                neighbors,
                True,
                capacity,
            )
        except ValueError as exc:
            expected_error = exc
        if expected_error is not None:
            before = ids.copy()
            with pytest.raises(ValueError, match="first primary closure"):
                plan_level1_chunk(0, neighbors, True, ids)
            assert np.array_equal(ids, before)
            continue
        actual = plan_level1_chunk(0, neighbors, True, ids)
        assert actual == (expected_primary, expected_selected)
        assert ids[:expected_selected].tolist() == values
        assert np.all(ids[expected_selected:] == -77)
        assert len(np.unique(ids[:expected_selected])) == expected_selected


def test_support_is_promoted_when_it_becomes_primary() -> None:
    neighbors = topology((4, 1, 1))
    ids = np.full(3, -1, dtype=np.int64)
    primary_count, selected_count = plan_level1_chunk(0, neighbors, True, ids)
    assert (primary_count, selected_count) == (2, 3)
    assert ids.tolist() == [0, 1, 2]


def test_face_closed_chunks_cover_all_primaries_exactly_once() -> None:
    neighbors = topology((6, 4, 3))
    capacity = 14
    ids = np.empty(capacity, dtype=np.int64)
    first = 0
    covered = []
    while first < neighbors.shape[0]:
        expected, expected_primary, expected_selected = plan_level1_chunk_reference(
            first,
            neighbors,
            True,
            capacity,
        )
        primary_count, selected_count = plan_level1_chunk(
            first,
            neighbors,
            True,
            ids,
        )
        assert (primary_count, selected_count) == (
            expected_primary,
            expected_selected,
        )
        assert ids[:selected_count].tolist() == expected
        covered.extend(ids[:primary_count].tolist())
        first += primary_count
    assert covered == list(range(neighbors.shape[0]))


def test_minimum_closed_capacity_and_invalid_requests_are_atomic() -> None:
    neighbors = topology((3, 3, 3))
    assert minimum_face_closed_slots(neighbors) == 7
    center_maps = level1_morton(i3(3, 3, 3))
    center = int(center_maps[0][1, 1, 1])
    too_small = np.full(6, -8, dtype=np.int64)
    before = too_small.copy()
    with pytest.raises(ValueError, match="first primary closure"):
        plan_level1_chunk(center, neighbors, True, too_small)
    assert np.array_equal(too_small, before)

    with pytest.raises(TypeError, match="bool"):
        plan_level1_chunk(0, neighbors, np.bool_(True), np.empty(7, dtype=np.int64))
    readonly = np.empty(7, dtype=np.int64)
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        plan_level1_chunk(0, neighbors, True, readonly)
    with pytest.raises(ValueError, match="exceeds"):
        plan_level1_chunk(neighbors.shape[0] + 1, neighbors, False, np.empty(7, dtype=np.int64))


def test_memmap_stream_reuses_budgeted_workspace_without_resident_full_copy(tmp_path) -> None:
    block_count, field_count = 257, 2
    block_shape = i3(8, 8, 8)
    path = tmp_path / "blocks.npy"
    writable = open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(block_count, field_count, *(int(value) for value in block_shape)),
    )
    for block_id in range(block_count):
        for field_id in range(field_count):
            writable[block_id, field_id].fill(10.0 * block_id + field_id)
    writable.flush()
    del writable
    backing = np.load(path, mmap_mode="r")
    assert isinstance(backing, np.memmap)

    per_slot = workspace_nbytes(1, field_count, block_shape)
    budget = 7 * per_slot
    capacity = workspace_slot_capacity(
        budget,
        block_count,
        field_count,
        block_shape,
    )
    assert capacity == 7
    ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty(
        (capacity, field_count, *(int(value) for value in block_shape)),
        dtype=np.float64,
    )
    assert payload.nbytes + ids.nbytes == budget
    assert backing.nbytes > budget
    payload_identity = id(payload)
    field_ids = np.arange(field_count, dtype=np.int64)
    faces = np.full((block_count, 6), -1, dtype=np.int64)
    lower = i3(0, 0, 0)
    total = 0.0
    first = 0
    while first < block_count:
        primary_count, selected_count = plan_level1_chunk(
            first,
            faces,
            False,
            ids,
        )
        gather_blocks_into(
            backing,
            lower,
            block_shape,
            ids[:selected_count],
            field_ids,
            payload[:selected_count],
            lower,
        )
        total += float(np.sum(payload[:primary_count]))
        first += primary_count
        assert id(payload) == payload_identity

    cells = int(np.prod(block_shape))
    expected = cells * sum(
        10.0 * block_id + field_id
        for block_id in range(block_count)
        for field_id in range(field_count)
    )
    assert total == expected
