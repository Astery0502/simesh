from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.morton import fill_morton_mapping3D
from simesh_rewrite.morton import fill_level1_morton, level1_morton
from simesh_rewrite.morton_reference import (
    level1_morton_reference,
    morton_key,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_mapping_invariants(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
) -> None:
    volume = int(np.prod(root_shape, dtype=np.int64))
    assert np.array_equal(np.sort(coord_to_rank.ravel()), np.arange(volume))
    assert np.all(rank_to_coord >= 0)
    assert np.all(rank_to_coord < root_shape)
    assert np.unique(rank_to_coord, axis=0).shape[0] == volume
    ranks = np.arange(volume, dtype=np.int64)
    assert np.array_equal(
        coord_to_rank[
            rank_to_coord[:, 0],
            rank_to_coord[:, 1],
            rank_to_coord[:, 2],
        ],
        ranks,
    )
    keys = [morton_key(*coordinate) for coordinate in rank_to_coord]
    assert all(left < right for left, right in zip(keys, keys[1:]))


def test_power_of_two_cube_freezes_bit_and_axis_order() -> None:
    coord_to_rank, rank_to_coord = level1_morton(i3(2, 2, 2))
    expected_inverse = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(rank_to_coord, expected_inverse)
    assert np.array_equal(coord_to_rank.ravel(order="C"), [0, 4, 2, 6, 1, 5, 3, 7])

    forward4, inverse4 = level1_morton(i3(4, 4, 4))
    assert np.array_equal(
        forward4,
        np.fromfunction(
            np.vectorize(lambda x, y, z: morton_key(int(x), int(y), int(z))),
            (4, 4, 4),
            dtype=int,
        ),
    )
    assert_mapping_invariants(i3(4, 4, 4), forward4, inverse4)


def test_clipped_box_has_exact_dense_maps() -> None:
    coord_to_rank, rank_to_coord = level1_morton(i3(3, 2, 2))
    expected_inverse = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
            [2, 1, 1],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(rank_to_coord, expected_inverse)
    assert np.array_equal(
        coord_to_rank.ravel(order="C"),
        [0, 4, 2, 6, 1, 5, 3, 7, 8, 10, 9, 11],
    )

    unequal_power_of_two, _ = level1_morton(i3(4, 2, 8))
    assert morton_key(0, 0, 2) == 32
    assert unequal_power_of_two[0, 0, 2] == 16


def test_compiled_matches_independent_reference_for_small_shapes() -> None:
    for nx in range(1, 9):
        for ny in range(1, 9):
            for nz in range(1, 9):
                root_shape = i3(nx, ny, nz)
                actual = level1_morton(root_shape)
                expected = level1_morton_reference(root_shape)
                assert np.array_equal(actual[0], expected[0])
                assert np.array_equal(actual[1], expected[1])
                assert_mapping_invariants(root_shape, *actual)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 4),
        (3, 2, 2),
        (2, 3, 4),
        (1, 3, 5),
        (5, 2, 7),
        (4, 2, 8),
        (6, 5, 3),
    ],
)
def test_compiled_matches_current_dense_mapping(shape: tuple[int, int, int]) -> None:
    root_shape = i3(*shape)
    actual = level1_morton(root_shape)
    current_forward = np.empty(shape, dtype=np.uint32)
    current_inverse = np.empty((int(np.prod(shape)), 3), dtype=np.uint32)
    fill_morton_mapping3D(current_forward, current_inverse, *shape)
    assert np.array_equal(actual[0], current_forward)
    assert np.array_equal(actual[1], current_inverse)


def test_mapping_has_no_current_ten_bit_coordinate_cap() -> None:
    root_shape = i3(2049, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    assert np.array_equal(coord_to_rank[:, 0, 0], np.arange(2049))
    assert np.array_equal(rank_to_coord[:, 0], np.arange(2049))
    assert np.all(rank_to_coord[:, 1:] == 0)


def test_reference_key_is_arbitrary_precision() -> None:
    assert morton_key(np.int64(1 << 21), 0, 0) == 1 << 63
    assert morton_key(np.int64(1 << 22), 0, 0) == 1 << 66
    with pytest.raises(ValueError, match="non-negative"):
        morton_key(-1, 0, 0)


def test_caller_owned_fill_and_topology_lookup_composition() -> None:
    root_shape = i3(3, 2, 2)
    root_shape_before = root_shape.copy()
    coord_to_rank = np.full((3, 2, 2), -1, dtype=np.int64)
    rank_to_coord = np.full((12, 3), -1, dtype=np.int64)
    fill_level1_morton(root_shape, coord_to_rank, rank_to_coord)
    assert np.array_equal(root_shape, root_shape_before)
    assert_mapping_invariants(root_shape, coord_to_rank, rank_to_coord)

    for rank, coordinate in enumerate(rank_to_coord):
        x, y, z = (int(value) for value in coordinate)
        if x + 1 < root_shape[0]:
            neighbor_rank = coord_to_rank[x + 1, y, z]
            assert np.array_equal(rank_to_coord[neighbor_rank], [x + 1, y, z])
        assert coord_to_rank[x, y, z] == rank


def test_invalid_calls_do_not_mutate_outputs() -> None:
    root_shape = i3(2, 2, 2)
    coord_to_rank = np.full((2, 2, 2), -7, dtype=np.int64)
    rank_to_coord = np.full((8, 3), -9, dtype=np.int64)
    before_forward = coord_to_rank.copy()
    before_inverse = rank_to_coord.copy()

    with pytest.raises(ValueError, match="shape"):
        fill_level1_morton(root_shape, coord_to_rank, rank_to_coord[:, :2])
    assert np.array_equal(coord_to_rank, before_forward)
    assert np.array_equal(rank_to_coord, before_inverse)

    noncontiguous = np.empty((2, 2, 4), dtype=np.int64)[:, :, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        fill_level1_morton(root_shape, noncontiguous, rank_to_coord)
    with pytest.raises(TypeError, match="int64"):
        fill_level1_morton(
            root_shape,
            np.empty((2, 2, 2), dtype=np.uint64),
            rank_to_coord,
        )

    readonly_inverse = rank_to_coord.copy()
    readonly_inverse.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_level1_morton(root_shape, coord_to_rank, readonly_inverse)

    base = np.empty(24, dtype=np.int64)
    overlapping_forward = base[:8].reshape(2, 2, 2)
    overlapping_inverse = base[:24].reshape(8, 3)
    with pytest.raises(ValueError, match="must not overlap"):
        fill_level1_morton(
            root_shape,
            overlapping_forward,
            overlapping_inverse,
        )

    with pytest.raises(ValueError, match="positive"):
        level1_morton(i3(2, 0, 2))
    with pytest.raises(OverflowError, match="int64"):
        fill_level1_morton(
            i3(np.iinfo(np.int64).max, 2, 1),
            coord_to_rank,
            rank_to_coord,
        )
