from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.sampling import place_level1_blocks
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def maps(
    root_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    return level1_morton(i3(*root_shape))


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def place_reference(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    block_ids: np.ndarray,
    block_cell_counts: np.ndarray,
    rank_to_coord: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    source = tuple(
        slice(int(lower), int(lower + extent))
        for lower, extent in zip(
            payload_valid_lower,
            block_cell_counts,
            strict=True,
        )
    )
    for slot, block_id in enumerate(block_ids):
        coordinate = rank_to_coord[int(block_id)]
        global_lower = coordinate * block_cell_counts
        destination = tuple(
            slice(int(lower), int(lower + extent))
            for lower, extent in zip(
                global_lower,
                block_cell_counts,
                strict=True,
            )
        )
        uniform_grid[(slice(None), *destination)] = payload[
            (slot, slice(None), *source)
        ]


def patterned_blocks(
    block_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    blocks = np.empty(
        (block_count, field_count, *block_shape),
        dtype=np.float64,
    )
    for block in range(block_count):
        for field in range(field_count):
            blocks[block, field] = (
                100000.0 * block
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
            )
    return blocks


def test_padded_reordered_repeated_partial_placement_matches_reference() -> None:
    root_shape = (3, 2, 2)
    block_counts = i3(2, 3, 4)
    domain_counts = i3(6, 6, 8)
    coord_to_rank, rank_to_coord = maps(root_shape)
    block_ids = np.array([9, 0, 9, 5], dtype=np.int64)
    lower = i3(1, 2, 3)
    upper = lower + block_counts
    spatial_shape = tuple(int(value) for value in upper + i3(2, 1, 2))
    halo_nan = np.asarray([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]
    payload = np.full((4, 2, *spatial_shape), halo_nan, dtype=np.float64)
    source = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper, strict=True)
    )
    x, y, z = np.indices(tuple(int(value) for value in block_counts), dtype=float)
    for slot in range(payload.shape[0]):
        for field in range(payload.shape[1]):
            payload[(slot, field, *source)] = (
                10000.0 * slot + 1000.0 * field + 100.0 * x + 10.0 * y + z
            )
    payload_before = payload.copy()
    untouched = np.asarray([0xFFF8000000005678], dtype=np.uint64).view(np.float64)[0]
    uniform = np.full(
        (2, *(int(value) for value in domain_counts)),
        untouched,
        dtype=np.float64,
    )
    expected = uniform.copy()
    place_reference(
        payload,
        lower,
        block_ids,
        block_counts,
        rank_to_coord,
        expected,
    )
    place_level1_blocks(
        payload,
        lower,
        upper,
        block_ids,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        uniform,
    )

    assert_bits_equal(uniform, expected)
    assert_bits_equal(payload, payload_before)

    repeated_coordinate = rank_to_coord[9]
    repeated_lower = repeated_coordinate * block_counts
    repeated_region = tuple(
        slice(int(start), int(start + extent))
        for start, extent in zip(repeated_lower, block_counts, strict=True)
    )
    assert_bits_equal(
        uniform[(slice(None), *repeated_region)],
        payload[(2, slice(None), *source)],
    )
    unselected_coordinate = rank_to_coord[1]
    unselected_lower = unselected_coordinate * block_counts
    unselected_region = tuple(
        slice(int(start), int(start + extent))
        for start, extent in zip(unselected_lower, block_counts, strict=True)
    )
    assert np.all(
        uniform[(slice(None), *unselected_region)].view(np.uint64)
        == np.uint64(0xFFF8000000005678)
    )


def test_special_float_representations_copy_exactly() -> None:
    root_shape = (2, 1, 1)
    block_counts = i3(1, 1, 4)
    domain_counts = i3(2, 1, 4)
    coord_to_rank, rank_to_coord = maps(root_shape)
    patterns = np.array(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF8000000000001,
            0x7FF0000000000001,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x0000000000000001,
            0xFFF8000000001234,
        ],
        dtype=np.uint64,
    )
    payload = np.empty((2, 1, 1, 1, 4), dtype=np.float64)
    payload[0] = patterns[4:].view(np.float64).reshape(1, 1, 1, 4)
    payload[1] = patterns[:4].view(np.float64).reshape(1, 1, 1, 4)
    uniform = np.zeros((1, 2, 1, 4), dtype=np.float64)
    place_level1_blocks(
        payload,
        i3(0, 0, 0),
        block_counts,
        np.array([1, 0], dtype=np.int64),
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        uniform,
    )
    assert np.array_equal(uniform.view(np.uint64).ravel(), patterns)


def test_non_cubic_full_placement_matches_current_bitwise() -> None:
    root_shape = (3, 2, 4)
    block_counts = i3(2, 4, 2)
    domain_counts = i3(6, 8, 8)
    coord_to_rank, rank_to_coord = maps(root_shape)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    payload = patterned_blocks(
        len(block_ids),
        3,
        tuple(int(value) for value in block_counts),
    )
    actual = np.full(
        (3, *(int(value) for value in domain_counts)),
        np.nan,
        dtype=np.float64,
    )
    place_level1_blocks(
        payload,
        i3(0, 0, 0),
        block_counts,
        block_ids,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        actual,
    )

    forest = AMRForest(
        3,
        *root_shape,
        np.ones(len(block_ids), dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block_counts.astype(np.uint32),
        domain_counts.astype(np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(0),
        np.uint32(payload.shape[1]),
        forest,
    )
    current = np.full_like(actual, np.nan)
    mesh.uniform_full_level1(payload, current)
    assert_bits_equal(actual, current)


def test_empty_slot_selection_is_a_valid_noop() -> None:
    root_shape = (2, 2, 1)
    block_counts = i3(2, 3, 4)
    domain_counts = i3(4, 6, 4)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = np.empty((0, 2, 2, 3, 4), dtype=np.float64)
    uniform = np.full(
        (2, *(int(value) for value in domain_counts)),
        -17.0,
        dtype=np.float64,
    )
    before = uniform.copy()
    place_level1_blocks(
        payload,
        i3(0, 0, 0),
        block_counts,
        np.empty(0, dtype=np.int64),
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        uniform,
    )
    assert_bits_equal(uniform, before)


def test_late_id_and_selected_map_failures_are_atomic() -> None:
    root_shape = (2, 2, 2)
    block_counts = i3(2, 2, 2)
    domain_counts = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = patterned_blocks(3, 1, (2, 2, 2))
    uniform = np.full((1, 4, 4, 4), -31.0)

    invalid_ids = np.array([0, 1, len(rank_to_coord)], dtype=np.int64)
    before = uniform.copy()
    with pytest.raises(ValueError, match="slot 2"):
        place_level1_blocks(
            payload,
            i3(0, 0, 0),
            block_counts,
            invalid_ids,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            uniform,
        )
    assert_bits_equal(uniform, before)

    selected_ids = np.array([0, 1, 2], dtype=np.int64)
    inconsistent_forward = coord_to_rank.copy()
    coordinate = tuple(int(value) for value in rank_to_coord[2])
    inconsistent_forward[coordinate] = 0
    with pytest.raises(ValueError, match="slot 2"):
        place_level1_blocks(
            payload,
            i3(0, 0, 0),
            block_counts,
            selected_ids,
            domain_counts,
            block_counts,
            inconsistent_forward,
            rank_to_coord,
            uniform,
        )
    assert_bits_equal(uniform, before)


def test_shape_region_readonly_and_overlap_fail_atomically() -> None:
    block_counts = i3(2, 2, 2)
    domain_counts = i3(2, 2, 2)
    coord_to_rank, rank_to_coord = maps((1, 1, 1))
    block_ids = np.array([0], dtype=np.int64)
    payload = patterned_blocks(1, 1, (2, 2, 2))

    wrong_shape = np.full((1, 2, 2, 3), -9.0)
    wrong_before = wrong_shape.copy()
    with pytest.raises(ValueError, match="shape"):
        place_level1_blocks(
            payload,
            i3(0, 0, 0),
            block_counts,
            block_ids,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            wrong_shape,
        )
    assert_bits_equal(wrong_shape, wrong_before)

    uniform = np.full((1, 2, 2, 2), -11.0)
    before = uniform.copy()
    with pytest.raises(ValueError, match="extent"):
        place_level1_blocks(
            payload,
            i3(0, 0, 0),
            i3(1, 2, 2),
            block_ids,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            uniform,
        )
    assert_bits_equal(uniform, before)

    readonly = uniform.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        place_level1_blocks(
            payload,
            i3(0, 0, 0),
            block_counts,
            block_ids,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            readonly,
        )

    overlapping = patterned_blocks(1, 1, (2, 2, 2))
    overlapping_output = overlapping.reshape(1, 2, 2, 2)
    overlap_before = overlapping.copy()
    with pytest.raises(ValueError, match="overlap"):
        place_level1_blocks(
            overlapping,
            i3(0, 0, 0),
            block_counts,
            block_ids,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            overlapping_output,
        )
    assert_bits_equal(overlapping, overlap_before)


def test_bounded_plan_gather_place_traversal_matches_direct_grid() -> None:
    root_shape = (5, 3, 2)
    block_counts = i3(2, 3, 4)
    domain_counts = i3(10, 9, 8)
    coord_to_rank, rank_to_coord = maps(root_shape)
    faces = level1_face_neighbors(
        i3(*root_shape),
        coord_to_rank,
        rank_to_coord,
    )
    backing = patterned_blocks(
        rank_to_coord.shape[0],
        3,
        tuple(int(value) for value in block_counts),
    )
    field_ids = np.array([2, 0], dtype=np.int64)
    capacity = 7
    chunk_ids = np.empty(capacity, dtype=np.int64)
    workspace = np.empty(
        (capacity, len(field_ids), *(int(value) for value in block_counts)),
        dtype=np.float64,
    )
    workspace_identity = id(workspace)
    uniform = np.full(
        (len(field_ids), *(int(value) for value in domain_counts)),
        np.nan,
        dtype=np.float64,
    )
    first = 0
    covered = []
    while first < faces.shape[0]:
        primary_count, selected_count = plan_level1_chunk(
            first,
            faces,
            False,
            chunk_ids,
        )
        assert primary_count == selected_count
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            block_counts,
            chunk_ids[:selected_count],
            field_ids,
            workspace[:selected_count],
            i3(0, 0, 0),
        )
        place_level1_blocks(
            workspace[:primary_count],
            i3(0, 0, 0),
            block_counts,
            chunk_ids[:primary_count],
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            uniform,
        )
        covered.extend(chunk_ids[:primary_count].tolist())
        first += primary_count
        assert id(workspace) == workspace_identity

    expected = np.full_like(uniform, np.nan)
    place_reference(
        backing[:, field_ids],
        i3(0, 0, 0),
        np.arange(rank_to_coord.shape[0], dtype=np.int64),
        block_counts,
        rank_to_coord,
        expected,
    )
    assert covered == list(range(rank_to_coord.shape[0]))
    assert_bits_equal(uniform, expected)
