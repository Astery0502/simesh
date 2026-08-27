from __future__ import annotations

import math

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.geometry import level1_block_geometry
from simesh_rewrite.halos import (
    BoundaryMode,
    fill_physical_halos,
    fill_same_level_halos,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.sampling import sample_level1_trilinear
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


EPS = np.finfo(np.float64).eps


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def maps(root_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    return level1_morton(i3(*root_shape))


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def face(
    lower: np.float64,
    upper: np.float64,
    spacing: np.float64,
    index: int,
    count: int,
) -> np.float64:
    if index == 0:
        return lower
    if index == count:
        return upper
    return np.float64(lower + np.float64(index) * spacing)


def center(lower: np.float64, spacing: np.float64, index: int) -> np.float64:
    return np.float64(lower + (np.float64(index) + np.float64(0.5)) * spacing)


def source_cell(
    value: np.float64,
    lower: np.float64,
    upper: np.float64,
    spacing: np.float64,
    count: int,
) -> int:
    result = 0
    for index in range(1, count):
        if face(lower, upper, spacing, index, count) <= value:
            result = index
    return result


def axis_stencil(
    output_index: int,
    block_coordinate: int,
    block_cells: int,
    sample_lower: np.float64,
    output_spacing: np.float64,
    domain_lower: np.float64,
    domain_upper: np.float64,
    native_spacing: np.float64,
    domain_cells: int,
) -> tuple[int, int, np.float64, np.float64]:
    output_center = center(sample_lower, output_spacing, output_index)
    block_lower = face(
        domain_lower,
        domain_upper,
        native_spacing,
        block_coordinate * block_cells,
        domain_cells,
    )
    delta = np.float64(output_center - block_lower)
    ratio = np.float64(delta / native_spacing)
    normalized = np.float64(ratio - np.float64(0.5))
    left = math.floor(float(normalized))
    weight = np.float64(normalized - np.float64(left))
    return left, left + 1, weight, output_center


def lerp(left: np.float64, right: np.float64, weight: np.float64) -> np.float64:
    one_minus = np.float64(np.float64(1.0) - weight)
    left_term = np.float64(left * one_minus)
    right_term = np.float64(right * weight)
    return np.float64(left_term + right_term)


def sample_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    block_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    output: np.ndarray,
) -> None:
    native_spacing = (domain_upper - domain_lower) / domain_counts
    output_counts = np.asarray(output.shape[1:], dtype=np.int64)
    output_spacing = (sample_upper - sample_lower) / output_counts
    for slot, block_id_value in enumerate(block_ids):
        block_id = int(block_id_value)
        coordinate = rank_to_coord[block_id]
        for output_index in np.ndindex(output.shape[1:]):
            owners = []
            centers = []
            for axis in range(3):
                value = center(
                    np.float64(sample_lower[axis]),
                    np.float64(output_spacing[axis]),
                    output_index[axis],
                )
                centers.append(value)
                owners.append(
                    source_cell(
                        value,
                        np.float64(domain_lower[axis]),
                        np.float64(domain_upper[axis]),
                        np.float64(native_spacing[axis]),
                        int(domain_counts[axis]),
                    )
                )
            owner_coordinate = tuple(
                owners[axis] // int(block_counts[axis]) for axis in range(3)
            )
            if int(coord_to_rank[owner_coordinate]) != block_id:
                continue

            left = []
            weights = []
            for axis in range(3):
                left_index, _, weight, _ = axis_stencil(
                    output_index[axis],
                    int(coordinate[axis]),
                    int(block_counts[axis]),
                    np.float64(sample_lower[axis]),
                    np.float64(output_spacing[axis]),
                    np.float64(domain_lower[axis]),
                    np.float64(domain_upper[axis]),
                    np.float64(native_spacing[axis]),
                    int(domain_counts[axis]),
                )
                left.append(int(interior_lower[axis]) + left_index)
                weights.append(weight)
            right = [index + 1 for index in left]
            for field in range(payload.shape[1]):
                c00 = lerp(
                    payload[slot, field, left[0], left[1], left[2]],
                    payload[slot, field, left[0], left[1], right[2]],
                    weights[2],
                )
                c01 = lerp(
                    payload[slot, field, left[0], right[1], left[2]],
                    payload[slot, field, left[0], right[1], right[2]],
                    weights[2],
                )
                c10 = lerp(
                    payload[slot, field, right[0], left[1], left[2]],
                    payload[slot, field, right[0], left[1], right[2]],
                    weights[2],
                )
                c11 = lerp(
                    payload[slot, field, right[0], right[1], left[2]],
                    payload[slot, field, right[0], right[1], right[2]],
                    weights[2],
                )
                c0 = lerp(c00, c01, weights[1])
                c1 = lerp(c10, c11, weights[1])
                output[(field, *output_index)] = lerp(c0, c1, weights[0])


def topology(root_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_to_rank, rank_to_coord = maps(root_shape)
    neighbors = level1_face_neighbors(
        i3(*root_shape),
        coord_to_rank,
        rank_to_coord,
    )
    return coord_to_rank, rank_to_coord, neighbors


def complete_payload(
    backing: np.ndarray,
    root_shape: tuple[int, int, int],
    modes: np.ndarray,
    normals: np.ndarray,
    *,
    halo: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coord_to_rank, rank_to_coord, neighbors = topology(root_shape)
    block_ids = np.arange(backing.shape[0], dtype=np.int64)
    block_shape = np.asarray(backing.shape[2:], dtype=np.int64)
    lower = np.full(3, halo, dtype=np.int64)
    upper = lower + block_shape
    spatial_shape = tuple(int(value) for value in upper + halo)
    payload = np.full(
        (backing.shape[0], backing.shape[1], *spatial_shape),
        np.nan,
    )
    interior = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper, strict=True)
    )
    payload[(slice(None), slice(None), *interior)] = backing
    fill_physical_halos(payload, lower, upper, block_ids, neighbors, modes, normals)
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        len(block_ids),
        neighbors,
        modes,
        normals,
    )
    return payload, lower, upper, coord_to_rank, rank_to_coord


def current_affine_mesh() -> tuple[AMRMesh, np.ndarray]:
    root_shape = (2, 2, 2)
    block_shape = np.array([4, 4, 4], dtype=np.uint32)
    forest = AMRForest(3, *root_shape, np.ones(8, dtype=np.int32))
    mesh = AMRMesh(
        3,
        block_shape,
        np.array([8, 8, 8], dtype=np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(1),
        np.uint32(3),
        forest,
    )
    data = np.empty((8, 3, 4, 4, 4), dtype=np.float64)
    for block in range(8):
        rnode = np.asarray(mesh.rnode)[block]
        x = rnode[0] + (np.arange(4)[:, None, None] + 0.5) * rnode[6]
        y = rnode[1] + (np.arange(4)[None, :, None] + 0.5) * rnode[7]
        z = rnode[2] + (np.arange(4)[None, None, :] + 0.5) * rnode[8]
        base = x + 2.0 * y + 3.0 * z
        data[block, 0] = base
        data[block, 1] = base + 10.0
        data[block, 2] = base + 20.0
    mesh.load_interior_data(data)
    mesh.apply_ghost_cells()
    return mesh, data


def test_scalar_reference_exact_owner_indices_and_face_weights() -> None:
    root_shape = (2, 1, 1)
    backing = np.empty((2, 1, 2, 2, 2), dtype=np.float64)
    for block in range(2):
        for local in range(2):
            backing[block, 0, local] = block * 2 + local
    modes = np.zeros((1, 6), dtype=np.uint8)
    payload, lower, upper, coord_to_rank, rank_to_coord = complete_payload(
        backing,
        root_shape,
        modes,
        i3(-1, -1, -1),
    )
    ids = np.arange(2, dtype=np.int64)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    domain_counts = i3(4, 2, 2)
    block_counts = i3(2, 2, 2)

    cases = [
        (1.0, 0, 0, np.float64(0.5), 0.5),
        (2.0, 1, -1, np.float64(0.5), 1.5),
    ]
    for x_value, owner_block, expected_left, expected_weight, expected_value in cases:
        sample_lower = f3(x_value - 0.5, 0.0, 0.0)
        sample_upper = f3(x_value + 0.5, 1.0, 1.0)
        left, right, weight, actual_center = axis_stencil(
            0,
            owner_block,
            2,
            sample_lower[0],
            np.float64(1.0),
            domain_lower[0],
            domain_upper[0],
            np.float64(1.0),
            4,
        )
        assert actual_center == x_value
        assert (left, right, weight) == (
            expected_left,
            expected_left + 1,
            expected_weight,
        )
        output = np.full((1, 1, 1, 1), -9.0)
        expected = output.copy()
        sample_reference(
            payload,
            lower,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            expected,
        )
        sample_level1_trilinear(
            payload,
            i3(0, 0, 0),
            np.asarray(payload.shape[2:], dtype=np.int64),
            lower,
            upper,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )
        assert output[0, 0, 0, 0] == expected_value
        assert_bits_equal(output, expected)


def test_affine_bound_and_reordered_fields_match_current() -> None:
    mesh, interior = current_affine_mesh()
    payload = np.ascontiguousarray(
        np.transpose(mesh.padded_view(), (0, 4, 1, 2, 3))[:, [2, 0]]
    )
    lower = i3(1, 1, 1)
    upper = i3(5, 5, 5)
    coord_to_rank, rank_to_coord = maps((2, 2, 2))
    ids = np.arange(8, dtype=np.int64)
    sample_lower = f3(0.25, 0.25, 0.25)
    sample_upper = f3(0.75, 0.75, 0.75)
    output = np.zeros((2, 6, 6, 6), dtype=np.float64)
    sample_level1_trilinear(
        payload,
        i3(0, 0, 0),
        i3(6, 6, 6),
        lower,
        upper,
        ids,
        f3(0.0, 0.0, 0.0),
        f3(1.0, 1.0, 1.0),
        i3(8, 8, 8),
        i3(4, 4, 4),
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        output,
    )
    current = np.zeros_like(output)
    mesh.uniform_grid_linear(
        current,
        np.array([6, 6, 6], dtype=np.uint32),
        sample_lower,
        sample_upper,
        np.array([2, 0], dtype=np.uint32),
    )
    spacing = (sample_upper - sample_lower) / i3(6, 6, 6)
    x = sample_lower[0] + (np.arange(6)[:, None, None] + 0.5) * spacing[0]
    y = sample_lower[1] + (np.arange(6)[None, :, None] + 0.5) * spacing[1]
    z = sample_lower[2] + (np.arange(6)[None, None, :] + 0.5) * spacing[2]
    analytic = x + 2.0 * y + 3.0 * z
    expected = np.stack((analytic + 20.0, analytic))
    scale = max(1.0, float(np.max(np.abs(expected))))
    bound = 64.0 * EPS * scale
    assert float(np.max(np.abs(output - expected))) <= bound
    assert float(np.max(np.abs(output - current))) <= bound
    assert_bits_equal(interior, interior.copy())


def test_one_layer_reach_and_zero_weight_still_reads_all_corners() -> None:
    payload = np.ones((1, 1, 4, 4, 4), dtype=np.float64)
    payload[0, 0, 2, 2, 2] = np.nan
    coord_to_rank, rank_to_coord = maps((1, 1, 1))
    output = np.zeros((1, 1, 1, 1), dtype=np.float64)
    sample_level1_trilinear(
        payload,
        i3(0, 0, 0),
        i3(4, 4, 4),
        i3(1, 1, 1),
        i3(3, 3, 3),
        np.array([0], dtype=np.int64),
        f3(0.0, 0.0, 0.0),
        f3(2.0, 2.0, 2.0),
        i3(2, 2, 2),
        i3(2, 2, 2),
        coord_to_rank,
        rank_to_coord,
        f3(0.0, 0.0, 0.0),
        f3(1.0, 1.0, 1.0),
        output,
    )
    assert np.isnan(output[0, 0, 0, 0])


def test_first_layer_physical_boundary_modes_are_interpolated() -> None:
    block_counts = i3(2, 2, 2)
    backing = np.empty((1, 4, 2, 2, 2), dtype=np.float64)
    backing[:, 0].fill(2.0)
    backing[:, 1].fill(2.0)
    backing[:, 2].fill(2.0)
    backing[:, 3].fill(2.0)
    modes = np.zeros((4, 6), dtype=np.uint8)
    modes[0, 0] = BoundaryMode.CONTINUOUS
    modes[1, 0] = BoundaryMode.SYMMETRIC
    modes[2, 0] = BoundaryMode.ANTISYMMETRIC
    modes[3, 0] = BoundaryMode.NO_INFLOW
    payload, lower, upper, coord_to_rank, rank_to_coord = complete_payload(
        backing,
        (1, 1, 1),
        modes,
        i3(3, -1, -1),
    )
    output = np.zeros((4, 1, 1, 1), dtype=np.float64)
    sample_level1_trilinear(
        payload,
        i3(0, 0, 0),
        i3(4, 4, 4),
        lower,
        upper,
        np.array([0], dtype=np.int64),
        f3(0.0, 0.0, 0.0),
        f3(2.0, 2.0, 2.0),
        block_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        f3(0.0, 0.0, 0.0),
        f3(0.5, 1.0, 1.0),
        output,
    )
    assert output[:, 0, 0, 0].tolist() == [2.0, 2.0, 1.0, 1.5]


def test_geo_valid_coincident_centers_interpolate_and_outer_collapse_rejects() -> None:
    x0 = np.float64((1 << 53) + 2)
    x1 = np.float64(x0 + 4.0)
    domain_lower = f3(x0, 0.0, 0.0)
    domain_upper = f3(x1, 1.0, 1.0)
    domain_counts = i3(2, 1, 1)
    block_counts = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = maps((1, 1, 1))
    level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        np.array([0], dtype=np.int64),
    )
    payload = np.empty((1, 1, 4, 3, 3), dtype=np.float64)
    payload.fill(10.0)
    payload[0, 0, 2].fill(20.0)
    payload[0, 0, 3].fill(20.0)
    output = np.zeros((1, 2, 1, 1), dtype=np.float64)
    sample_level1_trilinear(
        payload,
        i3(0, 0, 0),
        i3(4, 3, 3),
        i3(1, 1, 1),
        i3(3, 2, 2),
        np.array([0], dtype=np.int64),
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        domain_lower,
        domain_upper,
        output,
    )
    assert output[0, :, 0, 0].tolist() == [15.0, 15.0]

    collapsed_lower = f3(1.0e16, 0.0, 0.0)
    collapsed_upper = f3(1.0e16 + 8.0, 1.0, 1.0)
    collapsed_counts = i3(8, 1, 1)
    collapsed_payload = np.ones((1, 1, 10, 3, 3), dtype=np.float64)
    collapsed_output = np.full((1, 8, 1, 1), -7.0)
    before = collapsed_output.copy()
    with pytest.raises(ValueError, match="center|representable"):
        sample_level1_trilinear(
            collapsed_payload,
            i3(0, 0, 0),
            i3(10, 3, 3),
            i3(1, 1, 1),
            i3(9, 2, 2),
            np.array([0], dtype=np.int64),
            collapsed_lower,
            collapsed_upper,
            collapsed_counts,
            collapsed_counts,
            coord_to_rank,
            rank_to_coord,
            collapsed_lower,
            collapsed_upper,
            collapsed_output,
        )
    assert_bits_equal(collapsed_output, before)


def test_padded_reordered_repeated_partial_and_empty_preserve_output() -> None:
    root_shape = (3, 2, 1)
    block_counts = i3(2, 2, 2)
    domain_counts = i3(6, 4, 2)
    coord_to_rank, rank_to_coord = maps(root_shape)
    ids = np.array([4, 0, 4], dtype=np.int64)
    lower = i3(2, 1, 3)
    upper = lower + block_counts
    payload = np.empty((3, 1, 6, 5, 7), dtype=np.float64)
    for slot in range(3):
        x, y, z = np.indices((6, 5, 7), dtype=np.float64)
        payload[slot, 0] = 1000.0 * slot + 100.0 * x + 10.0 * y + z
    before_payload = payload.copy()
    sentinel = np.asarray([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]
    output = np.full((1, 9, 5, 3), sentinel)
    expected = output.copy()
    args = (
        f3(-3.0, 2.0, 10.0),
        f3(3.0, 6.0, 12.0),
        f3(-2.5, 2.25, 10.0),
        f3(2.5, 5.75, 12.0),
    )
    sample_reference(
        payload,
        lower,
        ids,
        args[0],
        args[1],
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        args[2],
        args[3],
        expected,
    )
    sample_level1_trilinear(
        payload,
        lower - 1,
        upper + 1,
        lower,
        upper,
        ids,
        args[0],
        args[1],
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        args[2],
        args[3],
        output,
    )
    assert_bits_equal(output, expected)
    assert_bits_equal(payload, before_payload)
    assert np.any(output.view(np.uint64) == np.uint64(0x7FF8000000001234))

    empty = np.full_like(output, sentinel)
    empty_before = empty.copy()
    sample_level1_trilinear(
        np.empty((0, 1, 6, 5, 7), dtype=np.float64),
        lower - 1,
        upper + 1,
        lower,
        upper,
        np.empty(0, dtype=np.int64),
        args[0],
        args[1],
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        args[2],
        args[3],
        empty,
    )
    assert_bits_equal(empty, empty_before)


def test_invalid_regions_reach_map_output_and_overlap_are_atomic() -> None:
    block_counts = i3(2, 2, 2)
    domain_counts = i3(4, 2, 2)
    coord_to_rank, rank_to_coord = maps((2, 1, 1))
    payload = np.ones((2, 1, 4, 4, 4), dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int64)
    common = (
        f3(0.0, 0.0, 0.0),
        f3(4.0, 2.0, 2.0),
        f3(0.0, 0.0, 0.0),
        f3(4.0, 2.0, 2.0),
    )
    output = np.full((1, 4, 2, 2), -13.0)
    before = output.copy()

    invalid_calls = [
        (i3(1, 0, 0), i3(4, 4, 4), i3(1, 1, 1), i3(3, 3, 3)),
        (i3(0, 0, 0), i3(4, 4, 4), i3(1, 1, 1), i3(2, 3, 3)),
        (i3(0, 0, 0), i3(3, 4, 4), i3(1, 1, 1), i3(3, 3, 3)),
    ]
    for valid_lower, valid_upper, interior_lower, interior_upper in invalid_calls:
        with pytest.raises(ValueError):
            sample_level1_trilinear(
                payload,
                valid_lower,
                valid_upper,
                interior_lower,
                interior_upper,
                ids,
                common[0],
                common[1],
                domain_counts,
                block_counts,
                coord_to_rank,
                rank_to_coord,
                common[2],
                common[3],
                output,
            )
        assert_bits_equal(output, before)

    broken = coord_to_rank.copy()
    coordinate = tuple(int(value) for value in rank_to_coord[1])
    broken[coordinate] = 0
    with pytest.raises(ValueError, match="slot 1"):
        sample_level1_trilinear(
            payload,
            i3(0, 0, 0),
            i3(4, 4, 4),
            i3(1, 1, 1),
            i3(3, 3, 3),
            ids,
            common[0],
            common[1],
            domain_counts,
            block_counts,
            broken,
            rank_to_coord,
            common[2],
            common[3],
            output,
        )
    assert_bits_equal(output, before)

    wrong_fields = np.full((2, 4, 2, 2), -5.0)
    wrong_before = wrong_fields.copy()
    with pytest.raises(ValueError, match="field counts"):
        sample_level1_trilinear(
            payload,
            i3(0, 0, 0),
            i3(4, 4, 4),
            i3(1, 1, 1),
            i3(3, 3, 3),
            ids,
            common[0],
            common[1],
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            common[2],
            common[3],
            wrong_fields,
        )
    assert_bits_equal(wrong_fields, wrong_before)

    overlapping = np.ones((1, 1, 4, 4, 4), dtype=np.float64)
    overlap_output = overlapping.reshape(1, 4, 4, 4)
    overlap_before = overlapping.copy()
    one_forward, one_inverse = maps((1, 1, 1))
    with pytest.raises(ValueError, match="overlap"):
        sample_level1_trilinear(
            overlapping,
            i3(0, 0, 0),
            i3(4, 4, 4),
            i3(1, 1, 1),
            i3(3, 3, 3),
            np.array([0], dtype=np.int64),
            f3(0.0, 0.0, 0.0),
            f3(2.0, 2.0, 2.0),
            block_counts,
            block_counts,
            one_forward,
            one_inverse,
            f3(0.0, 0.0, 0.0),
            f3(2.0, 2.0, 2.0),
            overlap_output,
        )
    assert_bits_equal(overlapping, overlap_before)


def test_smooth_field_has_second_order_linf_convergence() -> None:
    errors = []
    sample_lower = f3(0.17, 0.19, 0.23)
    sample_upper = f3(0.79, 0.83, 0.77)
    output_shape = (10, 9, 8)
    for cells in (12, 24, 48):
        spacing = 1.0 / cells
        coordinates = [
            (np.arange(-1, cells + 1, dtype=np.float64) + 0.5) * spacing
            for _ in range(3)
        ]
        x = coordinates[0][:, None, None]
        y = coordinates[1][None, :, None]
        z = coordinates[2][None, None, :]
        payload = (
            np.sin(2.0 * np.pi * x)
            + 0.5 * np.cos(2.0 * np.pi * y)
            + 0.25 * np.sin(2.0 * np.pi * z)
        )[None, None]
        coord_to_rank, rank_to_coord = maps((1, 1, 1))
        output = np.zeros((1, *output_shape), dtype=np.float64)
        sample_level1_trilinear(
            np.ascontiguousarray(payload),
            i3(0, 0, 0),
            i3(cells + 2, cells + 2, cells + 2),
            i3(1, 1, 1),
            i3(cells + 1, cells + 1, cells + 1),
            np.array([0], dtype=np.int64),
            f3(0.0, 0.0, 0.0),
            f3(1.0, 1.0, 1.0),
            i3(cells, cells, cells),
            i3(cells, cells, cells),
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )
        output_spacing = (sample_upper - sample_lower) / i3(*output_shape)
        xo = sample_lower[0] + (np.arange(output_shape[0])[:, None, None] + 0.5) * output_spacing[0]
        yo = sample_lower[1] + (np.arange(output_shape[1])[None, :, None] + 0.5) * output_spacing[1]
        zo = sample_lower[2] + (np.arange(output_shape[2])[None, None, :] + 0.5) * output_spacing[2]
        exact = (
            np.sin(2.0 * np.pi * xo)
            + 0.5 * np.cos(2.0 * np.pi * yo)
            + 0.25 * np.sin(2.0 * np.pi * zo)
        )
        errors.append(float(np.max(np.abs(output[0] - exact))))
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]
    assert min(orders) >= 1.8


def test_bounded_full_halo_composition_matches_scalar_reference() -> None:
    root_shape = (3, 3, 3)
    block_counts = i3(4, 4, 4)
    domain_counts = i3(12, 12, 12)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(1.0, 1.0, 1.0)
    sample_lower = f3(0.15, 0.2, 0.25)
    sample_upper = f3(0.85, 0.8, 0.75)
    coord_to_rank, rank_to_coord, neighbors = topology(root_shape)
    backing = np.empty((27, 2, 4, 4, 4), dtype=np.float64)
    x, y, z = np.indices((4, 4, 4), dtype=np.float64)
    for block in range(27):
        backing[block, 0] = 1000.0 * block + 100.0 * x + 10.0 * y + z
        backing[block, 1] = 5000.0 + backing[block, 0]
    field_ids = np.array([1, 0], dtype=np.int64)
    modes = np.zeros((2, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    capacity = 27
    chunk_ids = np.empty(capacity, dtype=np.int64)
    workspace = np.full((capacity, 2, 6, 6, 6), np.nan)
    output = np.full((2, 9, 8, 7), np.nan)
    first = 0
    while first < 27:
        primary_count, selected_count = plan_level1_halo_chunk(
            first,
            neighbors,
            chunk_ids,
        )
        workspace[:selected_count].fill(np.nan)
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            block_counts,
            chunk_ids[:selected_count],
            field_ids,
            workspace[:selected_count],
            i3(1, 1, 1),
        )
        fill_physical_halos(
            workspace[:selected_count],
            i3(1, 1, 1),
            i3(5, 5, 5),
            chunk_ids[:selected_count],
            neighbors,
            modes,
            normals,
        )
        fill_same_level_halos(
            workspace[:selected_count],
            i3(1, 1, 1),
            i3(5, 5, 5),
            chunk_ids[:selected_count],
            primary_count,
            neighbors,
            modes,
            normals,
        )
        sample_level1_trilinear(
            workspace[:primary_count],
            i3(0, 0, 0),
            i3(6, 6, 6),
            i3(1, 1, 1),
            i3(5, 5, 5),
            chunk_ids[:primary_count],
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )
        first += primary_count

    full_payload, lower, upper, _, _ = complete_payload(
        backing[:, field_ids],
        root_shape,
        modes,
        normals,
    )
    expected = np.full_like(output, np.nan)
    sample_reference(
        full_payload,
        lower,
        np.arange(27, dtype=np.int64),
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        expected,
    )
    assert_bits_equal(output, expected)
