from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import (
    fill_level1_face_neighbors,
    level1_face_neighbors,
)
from simesh_rewrite.topology_reference import level1_face_neighbors_reference


CURRENT_FACE_COLUMNS = np.array([12, 14, 10, 16, 4, 22], dtype=np.int64)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_topology_invariants(
    root_shape: np.ndarray,
    rank_to_coord: np.ndarray,
    neighbors: np.ndarray,
) -> None:
    volume = rank_to_coord.shape[0]
    assert neighbors.shape == (volume, 6)
    assert np.count_nonzero(neighbors == -1) == 2 * (
        int(root_shape[1] * root_shape[2])
        + int(root_shape[0] * root_shape[2])
        + int(root_shape[0] * root_shape[1])
    )
    for rank, coordinate in enumerate(rank_to_coord):
        for face, neighbor in enumerate(neighbors[rank]):
            axis, side = divmod(face, 2)
            expected_boundary = coordinate[axis] == (
                0 if side == 0 else root_shape[axis] - 1
            )
            assert (neighbor == -1) == expected_boundary
            if neighbor >= 0:
                assert neighbor < volume
                assert neighbor != rank
                expected = coordinate.copy()
                expected[axis] += -1 if side == 0 else 1
                assert np.array_equal(rank_to_coord[neighbor], expected)
                assert neighbors[neighbor, face ^ 1] == rank


def test_golden_nonsymmetric_singleton_topology() -> None:
    root_shape = i3(3, 2, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    expected = np.array(
        [
            [-1, 1, -1, 2, -1, -1],
            [0, 4, -1, 3, -1, -1],
            [-1, 3, 0, -1, -1, -1],
            [2, 5, 1, -1, -1, -1],
            [1, -1, -1, 5, -1, -1],
            [3, -1, 4, -1, -1, -1],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(neighbors, expected)
    assert_topology_invariants(root_shape, rank_to_coord, neighbors)


def test_compiled_matches_reference_for_small_shapes() -> None:
    for nx in range(1, 7):
        for ny in range(1, 7):
            for nz in range(1, 7):
                root_shape = i3(nx, ny, nz)
                coord_to_rank, rank_to_coord = level1_morton(root_shape)
                actual = level1_face_neighbors(
                    root_shape,
                    coord_to_rank,
                    rank_to_coord,
                )
                expected = level1_face_neighbors_reference(
                    root_shape,
                    coord_to_rank,
                    rank_to_coord,
                )
                assert np.array_equal(actual, expected)
                assert_topology_invariants(root_shape, rank_to_coord, actual)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (1, 2, 3), (2, 2, 2), (3, 2, 4), (5, 3, 2)],
)
def test_face_ids_match_current_level1_forest(shape: tuple[int, int, int]) -> None:
    root_shape = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    actual = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    forest = AMRForest(3, *shape, np.ones(int(np.prod(shape)), dtype=np.int32))
    current_ids = np.asarray(forest.neighbor_index, dtype=np.int64)[
        :, CURRENT_FACE_COLUMNS
    ] - 1
    current_types = np.asarray(forest.neighbor_type)[:, CURRENT_FACE_COLUMNS]
    assert np.array_equal(actual, current_ids)
    assert np.array_equal(actual == -1, current_types == 1)
    assert np.all(current_types[actual >= 0] == 3)


def walk_faces(neighbors: np.ndarray, start: int, delta: tuple[int, int, int], order) -> int:
    block = int(start)
    for axis in order:
        side = 0 if delta[axis] < 0 else 1
        block = int(neighbors[block, 2 * axis + side])
        if block < 0:
            return -1
    return block


def test_six_faces_reconstruct_all_current_level1_directions() -> None:
    shape = (3, 2, 4)
    root_shape = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    forest = AMRForest(3, *shape, np.ones(int(np.prod(shape)), dtype=np.int32))
    current_ids = np.asarray(forest.neighbor_index, dtype=np.int64) - 1
    current_types = np.asarray(forest.neighbor_type)

    for rank in range(rank_to_coord.shape[0]):
        for dx, dy, dz in np.ndindex(3, 3, 3):
            delta = (dx - 1, dy - 1, dz - 1)
            axes = tuple(axis for axis, value in enumerate(delta) if value)
            if not axes:
                assert current_ids[rank, 13] == rank
                assert current_types[rank, 13] == 0
                continue
            current_column = dx + 3 * dy + 9 * dz
            results = {
                walk_faces(neighbors, rank, delta, order)
                for order in permutations(axes)
            }
            assert results == {int(current_ids[rank, current_column])}


def test_distinct_axis_face_steps_commute() -> None:
    root_shape = i3(4, 3, 5)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    for rank in range(rank_to_coord.shape[0]):
        for first_axis in range(3):
            for second_axis in range(first_axis + 1, 3):
                for first_side in range(2):
                    for second_side in range(2):
                        delta = [0, 0, 0]
                        delta[first_axis] = -1 if first_side == 0 else 1
                        delta[second_axis] = -1 if second_side == 0 else 1
                        left = walk_faces(
                            neighbors,
                            rank,
                            tuple(delta),
                            (first_axis, second_axis),
                        )
                        right = walk_faces(
                            neighbors,
                            rank,
                            tuple(delta),
                            (second_axis, first_axis),
                        )
                        assert left == right


def test_invalid_maps_and_outputs_fail_before_mutation() -> None:
    root_shape = i3(2, 2, 2)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = np.full((8, 6), -7, dtype=np.int64)
    before = neighbors.copy()

    broken_inverse = rank_to_coord.copy()
    broken_inverse[3] = broken_inverse[2]
    with pytest.raises(ValueError, match="rank 3"):
        fill_level1_face_neighbors(
            root_shape,
            coord_to_rank,
            broken_inverse,
            neighbors,
        )
    assert np.array_equal(neighbors, before)

    broken_forward = coord_to_rank.copy()
    broken_forward[0, 0, 0] = 7
    with pytest.raises(ValueError, match="rank 0"):
        fill_level1_face_neighbors(
            root_shape,
            broken_forward,
            rank_to_coord,
            neighbors,
        )
    assert np.array_equal(neighbors, before)

    out_of_bounds_inverse = rank_to_coord.copy()
    out_of_bounds_inverse[0, 0] = -1
    with pytest.raises(ValueError, match="rank 0"):
        fill_level1_face_neighbors(
            root_shape,
            coord_to_rank,
            out_of_bounds_inverse,
            neighbors,
        )
    assert np.array_equal(neighbors, before)

    with pytest.raises(ValueError, match="shape"):
        fill_level1_face_neighbors(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            neighbors[:, :5],
        )
    with pytest.raises(TypeError, match="int64"):
        fill_level1_face_neighbors(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            np.empty((8, 6), dtype=np.int32),
        )

    noncontiguous_forward = np.empty((2, 2, 4), dtype=np.int64)[:, :, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        fill_level1_face_neighbors(
            root_shape,
            noncontiguous_forward,
            rank_to_coord,
            neighbors,
        )

    readonly = neighbors.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_level1_face_neighbors(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            readonly,
        )

    base = np.empty(48, dtype=np.int64)
    overlapping_forward = base[:8].reshape(2, 2, 2)
    overlapping_output = base.reshape(8, 6)
    with pytest.raises(ValueError, match="must not overlap"):
        fill_level1_face_neighbors(
            root_shape,
            overlapping_forward,
            rank_to_coord,
            overlapping_output,
        )

    with pytest.raises(ValueError, match="positive"):
        level1_face_neighbors(
            i3(2, 0, 2),
            coord_to_rank,
            rank_to_coord,
        )
    with pytest.raises(OverflowError, match="int64"):
        fill_level1_face_neighbors(
            i3(np.iinfo(np.int64).max, 2, 1),
            coord_to_rank,
            rank_to_coord,
            neighbors,
        )


def test_read_only_maps_are_accepted_and_preserved() -> None:
    root_shape = i3(3, 2, 2)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    expected = level1_face_neighbors_reference(
        root_shape,
        coord_to_rank,
        rank_to_coord,
    )
    coord_to_rank.setflags(write=False)
    rank_to_coord.setflags(write=False)
    actual = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    assert np.array_equal(actual, expected)
