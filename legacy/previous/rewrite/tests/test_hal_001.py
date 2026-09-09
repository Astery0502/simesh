from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.halos import (
    BoundaryMode,
    common_physical_valid_region,
    fill_physical_halos,
)
from simesh_rewrite.halos_reference import fill_physical_halos_reference
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def test_exact_face_layer_mappings() -> None:
    lower = i3(3, 0, 0)
    upper = i3(9, 1, 1)
    faces = np.full((1, 6), -1, dtype=np.int64)
    ids = np.array([0], dtype=np.int64)
    normals = i3(-1, -1, -1)
    interior = np.arange(6, dtype=np.float64)
    expected = {
        BoundaryMode.CONTINUOUS: ([0, 0, 0], [5, 5, 5]),
        BoundaryMode.SYMMETRIC: ([2, 1, 0], [5, 4, 3]),
        BoundaryMode.ANTISYMMETRIC: ([-2, -1, -0.0], [-5, -4, -3]),
    }
    for mode, (low_expected, high_expected) in expected.items():
        payload = np.full((1, 1, 12, 1, 1), 99.0)
        payload[0, 0, 3:9, 0, 0] = interior
        modes = np.zeros((1, 6), dtype=np.uint8)
        modes[0, 0:2] = mode
        fill_physical_halos(payload, lower, upper, ids, faces, modes, normals)
        assert np.array_equal(payload[0, 0, :3, 0, 0], low_expected)
        assert np.array_equal(payload[0, 0, 9:, 0, 0], high_expected)


def test_mixed_modes_match_reference_and_current_bitwise() -> None:
    block_shape = (4, 6, 8)
    halo = 3
    field_count = 4
    lower = i3(halo, halo, halo)
    upper = lower + i3(*block_shape)
    spatial_shape = tuple(int(value) for value in upper + halo)
    interior = np.empty((1, field_count, *block_shape), dtype=np.float64)
    x, y, z = np.indices(block_shape, dtype=np.float64)
    for field in range(field_count):
        interior[0, field] = (
            1000.0 * field + 100.0 * x + 10.0 * y + z + 1.0
        )
    interior[0, 1] = 2.0 - x
    interior[0, 2] = 3.0 - y
    interior[0, 3] = 4.0 - z

    modes = np.array(
        [
            [1, 2, 0, 1, 2, 0],
            [3, 3, 2, 0, 1, 2],
            [1, 2, 3, 3, 2, 0],
            [2, 0, 1, 2, 3, 3],
        ],
        dtype=np.uint8,
    )
    normals = i3(1, 2, 3)
    faces = np.full((1, 6), -1, dtype=np.int64)
    ids = np.array([0], dtype=np.int64)
    payload = np.full((1, field_count, *spatial_shape), np.nan)
    payload[
        :,
        :,
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ] = interior
    expected = payload.copy()
    fill_physical_halos_reference(
        expected,
        lower,
        upper,
        ids,
        faces,
        modes,
        normals,
    )
    fill_physical_halos(payload, lower, upper, ids, faces, modes, normals)
    assert np.array_equal(payload.view(np.uint64), expected.view(np.uint64))

    forest = AMRForest(3, 1, 1, 1, np.ones(1, dtype=np.int32))
    mesh = AMRMesh(
        3,
        np.asarray(block_shape, dtype=np.uint32),
        np.asarray(block_shape, dtype=np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(halo),
        np.uint32(field_count),
        forest,
        modes.astype(np.int32),
        normals.astype(np.int32),
    )
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()
    current = np.transpose(mesh.padded_view(), (0, 4, 1, 2, 3))
    assert np.array_equal(payload.view(np.uint64), current.view(np.uint64))
    assert np.array_equal(
        payload[
            :,
            :,
            lower[0] : upper[0],
            lower[1] : upper[1],
            lower[2] : upper[2],
        ].view(np.uint64),
        interior.view(np.uint64),
    )


def test_noinflow_non_normal_rows_are_continuous_and_axis_order_preserves_signed_zero() -> None:
    payload = np.full((1, 2, 4, 4, 1), 7.0)
    lower = i3(1, 1, 0)
    upper = i3(3, 3, 1)
    payload[0, 0, 1:3, 1:3, 0] = 5.0
    payload[0, 1, 1:3, 1:3, 0] = 2.0
    modes = np.zeros((2, 6), dtype=np.uint8)
    modes[0, 0] = BoundaryMode.NO_INFLOW
    modes[1, 0] = BoundaryMode.NO_INFLOW
    modes[1, 2] = BoundaryMode.ANTISYMMETRIC
    fill_physical_halos(
        payload,
        lower,
        upper,
        np.array([0], dtype=np.int64),
        np.full((1, 6), -1, dtype=np.int64),
        modes,
        i3(1, -1, -1),
    )
    assert payload[0, 0, 0, 1, 0] == 5.0
    assert payload[0, 1, 0, 1, 0] == 0.0
    assert payload[0, 1, 0, 0, 0].view(np.uint64) == 0x8000000000000000


def test_multislot_envelopes_and_common_valid_intersection() -> None:
    root_shape = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    ids = np.array([0, 1], dtype=np.int64)
    lower = i3(2, 2, 2)
    upper = i3(6, 6, 6)
    payload = np.full((2, 1, 8, 8, 8), -99.0)
    payload[:, :, 2:6, 2:6, 2:6] = 4.0
    fill_physical_halos(
        payload,
        lower,
        upper,
        ids,
        faces,
        np.zeros((1, 6), dtype=np.uint8),
        i3(-1, -1, -1),
    )
    assert np.all(payload[0, 0, :2, :, :] == 4.0)
    assert np.all(payload[0, 0, 6:, :, :] == -99.0)
    assert np.all(payload[1, 0, 6:, :, :] == 4.0)
    assert np.all(payload[1, 0, :2, :, :] == -99.0)
    common = common_physical_valid_region(i3(8, 8, 8), lower, upper, ids, faces)
    assert np.array_equal(common[0], i3(2, 0, 0))
    assert np.array_equal(common[1], i3(6, 8, 8))


def test_invalid_configuration_is_atomic() -> None:
    payload = np.full((1, 1, 6, 6, 6), 3.0)
    before = payload.copy()
    ids = np.array([0], dtype=np.int64)
    faces = np.full((1, 6), -1, dtype=np.int64)
    lower = i3(1, 1, 1)
    upper = i3(5, 5, 5)

    bad_modes = np.zeros((1, 6), dtype=np.uint8)
    bad_modes[0, 0] = 4
    with pytest.raises(ValueError, match="unknown mode"):
        fill_physical_halos(payload, lower, upper, ids, faces, bad_modes, i3(-1, -1, -1))
    assert np.array_equal(payload, before)

    noinflow = np.zeros((1, 6), dtype=np.uint8)
    noinflow[0, 0] = BoundaryMode.NO_INFLOW
    with pytest.raises(ValueError, match="normal field"):
        fill_physical_halos(payload, lower, upper, ids, faces, noinflow, i3(-1, -1, -1))

    deep_payload = np.zeros((1, 1, 10, 2, 2), dtype=np.float64)
    reflected = np.zeros((1, 6), dtype=np.uint8)
    reflected[0, 0] = BoundaryMode.SYMMETRIC
    with pytest.raises(ValueError, match="exceeds interior"):
        fill_physical_halos(
            deep_payload,
            i3(5, 0, 0),
            i3(9, 2, 2),
            ids,
            faces,
            reflected,
            i3(-1, -1, -1),
        )

    readonly = payload.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_physical_halos(
            readonly,
            lower,
            upper,
            ids,
            faces,
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
        )
