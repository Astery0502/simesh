from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halos import (
    BoundaryMode,
    fill_physical_halos,
    fill_same_level_halos,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def topology(root_shape: tuple[int, int, int]) -> np.ndarray:
    shape = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(shape)
    return level1_face_neighbors(shape, coord_to_rank, rank_to_coord)


def interior_slices(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper, strict=True)
    )


def axis_coded_backing(
    block_count: int,
    field_count: int,
    interior_shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(interior_shape, dtype=np.float64)
    backing = np.empty(
        (block_count, field_count, *interior_shape),
        dtype=np.float64,
    )
    for block in range(block_count):
        for field in range(field_count):
            backing[block, field] = (
                100000.0 * block
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return backing


def padded_from_backing(
    backing: np.ndarray,
    lower: np.ndarray,
    upper_width: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    interior_shape = np.asarray(backing.shape[2:], dtype=np.int64)
    upper = lower + interior_shape
    spatial_shape = tuple(int(value) for value in upper + upper_width)
    payload = np.full(
        (backing.shape[0], backing.shape[1], *spatial_shape),
        fill_value,
        dtype=np.float64,
    )
    payload[(slice(None), slice(None), *interior_slices(lower, upper))] = backing
    return payload, upper


def slot_for_id(block_ids: np.ndarray, block_id: int) -> int:
    for slot, candidate in enumerate(block_ids):
        if int(candidate) == block_id:
            return slot
    raise AssertionError(f"source block {block_id} is absent from the selected closure")


def fill_same_level_halos_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Independent direct-cell HAL-002 reference with linear slot lookup."""
    for primary_slot in range(primary_count):
        primary_id = int(block_ids[primary_slot])
        for field in range(payload.shape[1]):
            for target in np.ndindex(payload.shape[2:]):
                source_id = primary_id
                source = list(target)
                physical_operations: list[tuple[int, int, int]] = []
                has_sibling_crossing = False

                for axis in range(3):
                    if target[axis] < interior_lower[axis]:
                        face = 2 * axis
                        layer = int(interior_lower[axis]) - target[axis]
                    elif target[axis] >= interior_upper[axis]:
                        face = 2 * axis + 1
                        layer = target[axis] - int(interior_upper[axis]) + 1
                    else:
                        continue

                    neighbor_id = int(face_neighbor_ids[source_id, face])
                    if neighbor_id >= 0:
                        has_sibling_crossing = True
                        source_id = neighbor_id
                        if face % 2 == 0:
                            source[axis] = int(interior_upper[axis]) - layer
                        else:
                            source[axis] = int(interior_lower[axis]) + layer - 1
                        continue

                    mode = int(boundary_modes[field, face])
                    if mode in (
                        BoundaryMode.SYMMETRIC,
                        BoundaryMode.ANTISYMMETRIC,
                    ):
                        if face % 2 == 0:
                            source[axis] = int(interior_lower[axis]) + layer - 1
                        else:
                            source[axis] = int(interior_upper[axis]) - layer
                    elif face % 2 == 0:
                        source[axis] = int(interior_lower[axis])
                    else:
                        source[axis] = int(interior_upper[axis]) - 1
                    physical_operations.append((axis, face, mode))

                if not has_sibling_crossing:
                    continue

                source_slot = slot_for_id(block_ids, source_id)
                value = payload[(source_slot, field, *source)]
                for axis, face, mode in physical_operations:
                    if mode == BoundaryMode.ANTISYMMETRIC:
                        value = -value
                    elif (
                        mode == BoundaryMode.NO_INFLOW
                        and field == normal_field_slots[axis]
                    ):
                        if face % 2 == 0 and value > 0.0:
                            value = np.float64(0.0)
                        elif face % 2 == 1 and value < 0.0:
                            value = np.float64(0.0)
                payload[(primary_slot, field, *target)] = value


def current_level1_halos(
    interior: np.ndarray,
    root_shape: tuple[int, int, int],
    halo: int,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> np.ndarray:
    block_shape = np.asarray(interior.shape[2:], dtype=np.uint32)
    forest = AMRForest(
        3,
        *root_shape,
        np.ones(int(np.prod(root_shape)), dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block_shape,
        np.asarray(root_shape, dtype=np.uint32) * block_shape,
        np.zeros(3),
        np.ones(3),
        np.uint32(halo),
        np.uint32(interior.shape[1]),
        forest,
        boundary_modes.astype(np.int32),
        normal_field_slots.astype(np.int32),
    )
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()
    return np.transpose(mesh.padded_view(), (0, 4, 1, 2, 3)).copy()


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def pure_physical_mask(
    block_id: int,
    spatial_shape: tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    face_neighbor_ids: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(spatial_shape, dtype=bool)
    for target in np.ndindex(spatial_shape):
        outside_faces = []
        for axis in range(3):
            if target[axis] < lower[axis]:
                outside_faces.append(2 * axis)
            elif target[axis] >= upper[axis]:
                outside_faces.append(2 * axis + 1)
        if outside_faces and all(
            face_neighbor_ids[block_id, face] == -1 for face in outside_faces
        ):
            mask[target] = True
    return mask


def test_asymmetric_direct_mapping_matches_independent_reference_bitwise() -> None:
    root_shape = (3, 2, 2)
    faces = topology(root_shape)
    block_ids = np.arange(faces.shape[0], dtype=np.int64)
    backing = axis_coded_backing(faces.shape[0], 3, (4, 3, 5))
    backing[:, 1] = 1.5 - np.indices((4, 3, 5), dtype=np.float64)[0]
    backing[:, 2] = 1.5 - np.indices((4, 3, 5), dtype=np.float64)[1]
    lower = i3(1, 2, 3)
    payload, upper = padded_from_backing(backing, lower, i3(3, 1, 2))
    modes = np.array(
        [
            [1, 2, 0, 1, 2, 0],
            [3, 3, 2, 0, 1, 2],
            [1, 2, 3, 3, 2, 0],
        ],
        dtype=np.uint8,
    )
    normals = i3(1, 2, -1)
    fill_physical_halos(payload, lower, upper, block_ids, faces, modes, normals)
    expected = payload.copy()
    fill_same_level_halos_reference(
        expected,
        lower,
        upper,
        block_ids,
        len(block_ids),
        faces,
        modes,
        normals,
    )
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        len(block_ids),
        faces,
        modes,
        normals,
    )
    assert_bits_equal(payload, expected)


def test_sto_physical_same_level_composition_preserves_declared_regions() -> None:
    root_shape = (3, 3, 3)
    faces = topology(root_shape)
    planned_ids = np.full(27, -99, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_chunk(13, faces, planned_ids)
    block_ids = planned_ids[:selected_count].copy()
    assert primary_count < selected_count
    assert not np.array_equal(block_ids, np.arange(selected_count))

    backing = axis_coded_backing(faces.shape[0], 2, (4, 4, 4))
    lower = i3(2, 1, 2)
    upper = lower + i3(4, 4, 4)
    spatial_shape = tuple(int(value) for value in upper + i3(1, 2, 1))
    nan_value = np.asarray([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]
    payload = np.full(
        (selected_count, 2, *spatial_shape),
        nan_value,
        dtype=np.float64,
    )
    gather_blocks_into(
        backing,
        i3(0, 0, 0),
        i3(4, 4, 4),
        block_ids,
        np.arange(2, dtype=np.int64),
        payload,
        lower,
    )
    modes = np.array(
        [[1, 2, 0, 1, 2, 0], [0, 1, 2, 0, 1, 2]],
        dtype=np.uint8,
    )
    normals = i3(-1, -1, -1)
    fill_physical_halos(payload, lower, upper, block_ids, faces, modes, normals)

    halo_mask = np.ones(spatial_shape, dtype=bool)
    halo_mask[interior_slices(lower, upper)] = False
    for slot in range(primary_count, selected_count):
        for field in range(payload.shape[1]):
            payload[slot, field][halo_mask] = nan_value
    before = payload.copy()
    expected = payload.copy()
    fill_same_level_halos_reference(
        expected,
        lower,
        upper,
        block_ids,
        primary_count,
        faces,
        modes,
        normals,
    )
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        primary_count,
        faces,
        modes,
        normals,
    )

    assert_bits_equal(payload, expected)
    assert np.all(np.isfinite(payload[:primary_count]))
    assert_bits_equal(payload[primary_count:], before[primary_count:])
    interior = (slice(None), slice(None), *interior_slices(lower, upper))
    assert_bits_equal(payload[:primary_count][interior], before[:primary_count][interior])
    for slot in range(primary_count):
        mask = pure_physical_mask(
            int(block_ids[slot]),
            spatial_shape,
            lower,
            upper,
            faces,
        )
        assert_bits_equal(payload[slot][:, mask], before[slot][:, mask])


def test_commuting_modes_match_current_level1_mesh_bitwise() -> None:
    root_shape = (2, 2, 2)
    faces = topology(root_shape)
    block_ids = np.arange(faces.shape[0], dtype=np.int64)
    interior = axis_coded_backing(faces.shape[0], 3, (4, 6, 4))
    halo = 2
    lower = i3(halo, halo, halo)
    payload, upper = padded_from_backing(interior, lower, i3(halo, halo, halo))
    modes = np.array(
        [
            [0, 1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2, 0],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=np.uint8,
    )
    normals = i3(-1, -1, -1)
    fill_physical_halos(payload, lower, upper, block_ids, faces, modes, normals)
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        len(block_ids),
        faces,
        modes,
        normals,
    )
    current = current_level1_halos(interior, root_shape, halo, modes, normals)
    assert_bits_equal(payload, current)


def test_noncommuting_current_divergence_is_exact_and_bounded() -> None:
    root_shape = (2, 2, 2)
    faces = topology(root_shape)
    block_ids = np.arange(faces.shape[0], dtype=np.int64)
    interior = np.full((faces.shape[0], 1, 4, 4, 4), 611.0)
    halo = 2
    lower = i3(halo, halo, halo)
    payload, upper = padded_from_backing(interior, lower, i3(halo, halo, halo))
    modes = np.zeros((1, 6), dtype=np.uint8)
    modes[0, 0] = BoundaryMode.NO_INFLOW
    modes[0, 3] = BoundaryMode.ANTISYMMETRIC
    normals = i3(0, -1, -1)
    fill_physical_halos(payload, lower, upper, block_ids, faces, modes, normals)
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        len(block_ids),
        faces,
        modes,
        normals,
    )
    current = current_level1_halos(interior, root_shape, halo, modes, normals)

    rewrite_value = payload[2, 0, 0, 6, 6]
    current_value = current[2, 0, 0, 6, 6]
    assert rewrite_value.view(np.uint64) == 0x8000000000000000
    assert current_value == -611.0
    mismatch = payload.view(np.uint64) != current.view(np.uint64)
    expected_mismatch = np.zeros(payload.shape, dtype=bool)
    expected_mismatch[2, 0, 0:2, 6:8, 6:8] = True
    assert np.array_equal(mismatch, expected_mismatch)


def test_singleton_root_axes_match_direct_reference() -> None:
    root_shape = (1, 3, 1)
    faces = topology(root_shape)
    block_ids = np.arange(faces.shape[0], dtype=np.int64)
    assert np.all(faces[:, [0, 1, 4, 5]] == -1)
    backing = axis_coded_backing(faces.shape[0], 2, (3, 4, 2))
    lower = i3(2, 1, 1)
    payload, upper = padded_from_backing(backing, lower, i3(1, 2, 2))
    modes = np.array(
        [[2, 1, 0, 1, 2, 0], [3, 3, 1, 2, 0, 1]],
        dtype=np.uint8,
    )
    normals = i3(1, -1, -1)
    fill_physical_halos(payload, lower, upper, block_ids, faces, modes, normals)
    expected = payload.copy()
    fill_same_level_halos_reference(
        expected,
        lower,
        upper,
        block_ids,
        len(block_ids),
        faces,
        modes,
        normals,
    )
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        len(block_ids),
        faces,
        modes,
        normals,
    )
    assert_bits_equal(payload, expected)
    assert np.all(np.isfinite(payload))


def test_missing_closure_and_duplicate_ids_are_atomic() -> None:
    root_shape = (3, 3, 3)
    shape = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(shape)
    faces = level1_face_neighbors(shape, coord_to_rank, rank_to_coord)
    center = int(coord_to_rank[1, 1, 1])
    missing = int(coord_to_rank[0, 0, 0])
    incomplete_ids = np.asarray(
        [center]
        + [
            block
            for block in range(faces.shape[0])
            if block not in (center, missing)
        ],
        dtype=np.int64,
    )
    payload = np.arange(
        len(incomplete_ids) * 4**3,
        dtype=np.float64,
    ).reshape(len(incomplete_ids), 1, 4, 4, 4)
    before = payload.copy()
    with pytest.raises(ValueError):
        fill_same_level_halos(
            payload,
            i3(1, 1, 1),
            i3(3, 3, 3),
            incomplete_ids,
            1,
            faces,
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
        )
    assert_bits_equal(payload, before)

    duplicate_ids = np.arange(faces.shape[0], dtype=np.int64)
    duplicate_ids[-1] = duplicate_ids[-2]
    duplicate_payload = np.zeros((len(duplicate_ids), 1, 4, 4, 4))
    duplicate_before = duplicate_payload.copy()
    with pytest.raises(ValueError):
        fill_same_level_halos(
            duplicate_payload,
            i3(1, 1, 1),
            i3(3, 3, 3),
            duplicate_ids,
            1,
            faces,
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
        )
    assert_bits_equal(duplicate_payload, duplicate_before)


def test_deep_width_readonly_and_overlap_fail_atomically() -> None:
    faces = topology((1, 1, 1))
    ids = np.array([0], dtype=np.int64)
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)

    deep = np.arange(20, dtype=np.float64).reshape(1, 1, 5, 2, 2)
    deep_before = deep.copy()
    with pytest.raises(ValueError):
        fill_same_level_halos(
            deep,
            i3(3, 0, 0),
            i3(5, 2, 2),
            ids,
            1,
            faces,
            modes,
            normals,
        )
    assert_bits_equal(deep, deep_before)

    readonly = np.ones((1, 1, 4, 4, 4), dtype=np.float64)
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_same_level_halos(
            readonly,
            i3(1, 1, 1),
            i3(3, 3, 3),
            ids,
            1,
            faces,
            modes,
            normals,
        )

    overlapping = np.ones((1, 1, 2, 1, 1), dtype=np.float64)
    overlapping_ids = overlapping.view(np.int64).reshape(-1)[:1]
    overlapping_ids[0] = 0
    overlap_before = overlapping.copy()
    with pytest.raises(ValueError, match="overlap"):
        fill_same_level_halos(
            overlapping,
            i3(0, 0, 0),
            i3(2, 1, 1),
            overlapping_ids,
            1,
            faces,
            modes,
            normals,
        )
    assert_bits_equal(overlapping, overlap_before)


def test_zero_primary_and_end_plan_are_exact_noops() -> None:
    faces = topology((1, 1, 1))
    payload = np.arange(64, dtype=np.float64).reshape(1, 1, 4, 4, 4)
    before = payload.copy()
    fill_same_level_halos(
        payload,
        i3(1, 1, 1),
        i3(3, 3, 3),
        np.array([0], dtype=np.int64),
        0,
        faces,
        np.zeros((1, 6), dtype=np.uint8),
        i3(-1, -1, -1),
    )
    assert_bits_equal(payload, before)

    planned = np.full(1, -17, dtype=np.int64)
    assert plan_level1_halo_chunk(faces.shape[0], faces, planned) == (0, 0)
    empty_payload = np.empty((0, 1, 4, 4, 4), dtype=np.float64)
    fill_same_level_halos(
        empty_payload,
        i3(1, 1, 1),
        i3(3, 3, 3),
        planned[:0],
        0,
        faces,
        np.zeros((1, 6), dtype=np.uint8),
        i3(-1, -1, -1),
    )
    assert empty_payload.shape[0] == 0
