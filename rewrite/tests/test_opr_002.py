from __future__ import annotations

import math

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.access import (
    AccessPattern,
    required_input_region,
    validate_access_requirement,
)
from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.operators import central_difference_into
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from
from simesh_rewrite.topology import level1_face_neighbors


EPS = np.finfo(np.float64).eps


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def patterned(shape: tuple[int, ...]) -> np.ndarray:
    slot, field, x, y, z = np.indices(shape, dtype=np.float64)
    return 10000.0 * slot + 1000.0 * field + 100.0 * x + 10.0 * y + z


def reference(
    source: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    source_field: int,
    axis: int,
    spacing: np.ndarray,
    destination: np.ndarray,
    destination_field: int,
    destination_lower: np.ndarray,
) -> None:
    inverse = np.float64(0.5) / np.float64(spacing[axis])
    extent = output_upper - output_lower
    for slot in range(source.shape[0]):
        for offset in np.ndindex(*(int(value) for value in extent)):
            point = [int(output_lower[d]) + offset[d] for d in range(3)]
            minus = point.copy()
            plus = point.copy()
            minus[axis] -= 1
            plus[axis] += 1
            difference = np.subtract(
                source[(slot, source_field, *plus)],
                source[(slot, source_field, *minus)],
            )
            result = np.multiply(difference, inverse)
            target = [int(destination_lower[d]) + offset[d] for d in range(3)]
            destination[(slot, destination_field, *target)] = result


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_translated_axes_fields_and_transverse_edges_match_scalar_reference(
    axis: int,
) -> None:
    source = patterned((2, 3, 6, 7, 8))
    source_before = source.copy()
    source_valid_lower = i3(0, 0, 0)
    source_valid_upper = i3(6, 7, 8)
    output_lower = i3(0, 0, 0)
    output_upper = i3(6, 7, 8)
    output_lower[axis] = 1
    output_upper[axis] -= 1
    extent = output_upper - output_lower
    destination_lower = i3(1, 2, 1)
    destination_shape = tuple(int(value) for value in destination_lower + extent + 1)
    destination = np.full((2, 4, *destination_shape), -17.0)
    expected = destination.copy()
    spacing = np.array([0.25, 0.5, 2.0])
    reference(
        source,
        output_lower,
        output_upper,
        2,
        axis,
        spacing,
        expected,
        3,
        destination_lower,
    )
    central_difference_into(
        source,
        source_valid_lower,
        source_valid_upper,
        output_lower,
        output_upper,
        2,
        axis,
        spacing,
        destination,
        3,
        destination_lower,
    )
    assert_bits_equal(destination, expected)
    assert_bits_equal(source, source_before)
    assert np.any(destination[:, 3] != -17.0)
    assert np.all(destination[:, :3] == -17.0)


def test_inverse_then_subtract_then_multiply_order_is_exact() -> None:
    spacing_x = np.float64(3.74961474737501e-15)
    difference = np.float64(-3.022634972346485e95)
    source = np.zeros((1, 1, 3, 1, 1), dtype=np.float64)
    source[0, 0, 2, 0, 0] = difference
    destination = np.empty((1, 1, 1, 1, 1), dtype=np.float64)
    central_difference_into(
        source,
        i3(0, 0, 0),
        i3(3, 1, 1),
        i3(1, 0, 0),
        i3(2, 1, 1),
        0,
        0,
        np.array([spacing_x, 1.0, 1.0]),
        destination,
        0,
        i3(0, 0, 0),
    )
    assert destination[0, 0, 0, 0, 0].view(np.uint64) == np.uint64(
        0xD6B1297FE19BD436
    )


def test_one_layer_is_sufficient_and_transverse_outer_cells_are_computed() -> None:
    source = np.empty((1, 1, 5, 3, 3), dtype=np.float64)
    for i in range(5):
        source[0, 0, i].fill(float(i))
    destination = np.full((1, 1, 3, 3, 3), -9.0)
    central_difference_into(
        source,
        i3(0, 0, 0),
        i3(5, 3, 3),
        i3(1, 0, 0),
        i3(4, 3, 3),
        0,
        0,
        np.ones(3),
        destination,
        0,
        i3(0, 0, 0),
    )
    assert np.all(destination == 1.0)


def current_affine_derivatives() -> tuple[np.ndarray, np.ndarray, AMRMesh]:
    root_shape = (2, 2, 2)
    block_shape = np.array([4, 4, 4], dtype=np.uint32)
    forest = AMRForest(3, *root_shape, np.ones(8, dtype=np.int32))
    mesh = AMRMesh(
        3,
        block_shape,
        np.array([8, 8, 8], dtype=np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(2),
        np.uint32(3),
        forest,
    )
    interior = np.empty((8, 3, 4, 4, 4), dtype=np.float64)
    for block in range(8):
        rnode = np.asarray(mesh.rnode)[block]
        x = rnode[0] + (np.arange(4)[:, None, None] + 0.5) * rnode[6]
        y = rnode[1] + (np.arange(4)[None, :, None] + 0.5) * rnode[7]
        z = rnode[2] + (np.arange(4)[None, None, :] + 0.5) * rnode[8]
        interior[block, 0] = x
        interior[block, 1] = 2.0 * y
        interior[block, 2] = 5.0 * z
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()
    current = np.empty((8, 8, 8, 8, 3), dtype=np.float64)
    mesh.first_derivative_fields(
        current,
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([0, 1, 2], dtype=np.uint32),
        np.ones(3),
    )
    return interior, current, mesh


def test_affine_xyz_matches_current_and_has_physical_half_slopes() -> None:
    _, current, mesh = current_affine_derivatives()
    source = np.transpose(mesh.padded_view(), (0, 4, 1, 2, 3)).copy()
    spacing = np.asarray(mesh.rnode)[0, 6:9].copy()
    destination = np.empty((8, 3, 4, 4, 4), dtype=np.float64)
    for axis in range(3):
        central_difference_into(
            source,
            i3(0, 0, 0),
            i3(8, 8, 8),
            i3(2, 2, 2),
            i3(6, 6, 6),
            axis,
            axis,
            spacing,
            destination,
            axis,
            i3(0, 0, 0),
        )
    current_interior = np.transpose(current[:, 2:6, 2:6, 2:6, :], (0, 4, 1, 2, 3))
    np.testing.assert_allclose(destination, current_interior, rtol=0.0, atol=16 * EPS)

    _, rank_to_coord = level1_morton(i3(2, 2, 2))
    expected_slopes = (1.0, 2.0, 5.0)
    for block, coordinate in enumerate(rank_to_coord):
        for axis, slope in enumerate(expected_slopes):
            expected = np.full((4, 4, 4), slope)
            if coordinate[axis] == 0:
                index = [slice(None)] * 3
                index[axis] = 0
                expected[tuple(index)] = 0.5 * slope
            if coordinate[axis] == 1:
                index = [slice(None)] * 3
                index[axis] = 3
                expected[tuple(index)] = 0.5 * slope
            np.testing.assert_allclose(
                destination[block, axis],
                expected,
                rtol=0.0,
                atol=16 * EPS,
            )


def test_direct_signed_zero_differs_from_current_batch_accumulation() -> None:
    source = np.zeros((1, 1, 3, 1, 1), dtype=np.float64)
    source[0, 0, 0, 0, 0] = +0.0
    source[0, 0, 2, 0, 0] = -0.0
    destination = np.empty((1, 1, 1, 1, 1), dtype=np.float64)
    central_difference_into(
        source,
        i3(0, 0, 0),
        i3(3, 1, 1),
        i3(1, 0, 0),
        i3(2, 1, 1),
        0,
        0,
        np.ones(3),
        destination,
        0,
        i3(0, 0, 0),
    )
    assert destination[0, 0, 0, 0, 0].view(np.uint64) == np.uint64(
        0x8000000000000000
    )

    forest = AMRForest(3, 1, 1, 1, np.ones(1, dtype=np.int32))
    block = np.array([2, 2, 2], dtype=np.uint32)
    mesh = AMRMesh(3, block, block, np.zeros(3), np.array([2.0, 2.0, 2.0]), 2, 1, forest)
    mesh.load_interior_data(np.zeros((1, 1, 2, 2, 2)))
    padded = mesh.padded_view()
    padded[0, 1, 2, 2, 0] = +0.0
    padded[0, 3, 2, 2, 0] = -0.0
    current = np.empty((1, 6, 6, 6, 1), dtype=np.float64)
    mesh.first_derivative_fields(
        current,
        np.array([0], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        np.array([1.0]),
    )
    assert current[0, 2, 2, 2, 0].view(np.uint64) == np.uint64(0)


def test_nonfinite_overflow_and_center_value_is_not_read() -> None:
    source = np.zeros((1, 1, 3, 1, 4), dtype=np.float64)
    source[0, 0, 1] = np.nan
    source[0, 0, 0, 0] = [1.0, np.inf, -np.inf, -np.finfo(float).max]
    source[0, 0, 2, 0] = [3.0, np.inf, np.inf, np.finfo(float).max]
    destination = np.empty((1, 1, 1, 1, 4), dtype=np.float64)
    with np.errstate(all="ignore"):
        central_difference_into(
            source,
            i3(0, 0, 0),
            i3(3, 1, 4),
            i3(1, 0, 0),
            i3(2, 1, 4),
            0,
            0,
            np.ones(3),
            destination,
            0,
            i3(0, 0, 0),
        )
    result = destination[0, 0, 0, 0]
    assert result[0] == 1.0
    assert np.isnan(result[1])
    assert np.isposinf(result[2])
    assert np.isposinf(result[3])


def test_empty_requests_validate_metadata_and_do_not_read_neighbors() -> None:
    source = np.full((1, 1, 1, 1, 1), np.nan)
    destination = np.full((1, 1, 2, 2, 2), -3.0)
    before = destination.copy()
    central_difference_into(
        source,
        i3(0, 0, 0),
        i3(1, 1, 1),
        i3(0, 0, 0),
        i3(0, 1, 1),
        0,
        0,
        np.ones(3),
        destination,
        0,
        i3(0, 0, 0),
    )
    assert_bits_equal(destination, before)
    with pytest.raises(ValueError, match="axis"):
        central_difference_into(
            source,
            i3(0, 0, 0),
            i3(1, 1, 1),
            i3(0, 0, 0),
            i3(0, 1, 1),
            0,
            3,
            np.ones(3),
            destination,
            0,
            i3(0, 0, 0),
        )
    with pytest.raises(ValueError, match="normal"):
        central_difference_into(
            source,
            i3(0, 0, 0),
            i3(1, 1, 1),
            i3(0, 0, 0),
            i3(0, 1, 1),
            0,
            0,
            np.array([0.0, 1.0, 1.0]),
            destination,
            0,
            i3(0, 0, 0),
        )


def test_invalid_spacing_field_axis_regions_support_output_and_overlap_are_atomic() -> None:
    source = patterned((1, 2, 4, 4, 4))
    destination = np.full((1, 1, 4, 4, 4), -11.0)
    before = destination.copy()
    base = (
        i3(0, 0, 0),
        i3(4, 4, 4),
        i3(1, 0, 0),
        i3(3, 4, 4),
    )
    for bad_spacing in (
        np.array([0.0, 1.0, 1.0]),
        np.array([np.nextafter(0.0, 1.0), 1.0, 1.0]),
        np.array([np.inf, 1.0, 1.0]),
        np.array([np.nan, 1.0, 1.0]),
    ):
        with pytest.raises(ValueError):
            central_difference_into(
                source,
                base[0],
                base[1],
                base[2],
                base[3],
                0,
                0,
                bad_spacing,
                destination,
                0,
                i3(0, 0, 0),
            )
        assert_bits_equal(destination, before)

    invalid_variants = [
        (2, 0, base[0], base[1], base[2], base[3]),
        (0, 3, base[0], base[1], base[2], base[3]),
        (0, 0, base[0], base[1], i3(3, 0, 0), i3(2, 4, 4)),
        (0, 0, i3(1, 0, 0), base[1], i3(1, 0, 0), base[3]),
    ]
    for field, axis, valid_lower, valid_upper, output_lower, output_upper in invalid_variants:
        with pytest.raises(ValueError):
            central_difference_into(
                source,
                valid_lower,
                valid_upper,
                output_lower,
                output_upper,
                field,
                axis,
                np.ones(3),
                destination,
                0,
                i3(0, 0, 0),
            )
        assert_bits_equal(destination, before)

    readonly = destination.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        central_difference_into(
            source,
            *base,
            0,
            0,
            np.ones(3),
            readonly,
            0,
            i3(0, 0, 0),
        )

    overlap = patterned((1, 2, 4, 4, 4))
    overlap_before = overlap.copy()
    with pytest.raises(ValueError, match="overlap"):
        central_difference_into(
            overlap,
            *base,
            0,
            0,
            np.ones(3),
            overlap,
            0,
            i3(0, 0, 0),
        )
    assert_bits_equal(overlap, overlap_before)

    with pytest.raises(OverflowError, match="int64"):
        central_difference_into(
            source,
            *base,
            0,
            0,
            np.ones(3),
            destination,
            0,
            i3(np.iinfo(np.int64).max, 0, 0),
        )
    assert_bits_equal(destination, before)


def test_fnd_axis_reach_is_tight() -> None:
    for axis in range(3):
        reach = np.zeros(3, dtype=np.int64)
        reach[axis] = 1
        validate_access_requirement(AccessPattern.LOCAL_STENCIL, reach, reach)
        lower, upper = required_input_region(
            i3(2, 3, 4),
            i3(7, 9, 11),
            reach,
            reach,
        )
        expected_lower = i3(2, 3, 4)
        expected_upper = i3(7, 9, 11)
        expected_lower[axis] -= 1
        expected_upper[axis] += 1
        assert np.array_equal(lower, expected_lower)
        assert np.array_equal(upper, expected_upper)


def test_smooth_field_has_second_order_convergence() -> None:
    errors = []
    for cells in (12, 24, 48):
        spacing = 1.0 / cells
        x = (np.arange(-1, cells + 1, dtype=np.float64) + 0.5) * spacing
        source = np.sin(2.0 * np.pi * x).reshape(1, 1, cells + 2, 1, 1)
        destination = np.empty((1, 1, cells, 1, 1), dtype=np.float64)
        central_difference_into(
            np.ascontiguousarray(source),
            i3(0, 0, 0),
            i3(cells + 2, 1, 1),
            i3(1, 0, 0),
            i3(cells + 1, 1, 1),
            0,
            0,
            np.array([spacing, 1.0, 1.0]),
            destination,
            0,
            i3(0, 0, 0),
        )
        centers = (np.arange(cells, dtype=np.float64) + 0.5) * spacing
        exact = 2.0 * np.pi * np.cos(2.0 * np.pi * centers)
        errors.append(float(np.max(np.abs(destination[0, 0, :, 0, 0] - exact))))
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]
    assert min(orders) >= 1.8


def test_bounded_full_halo_gather_derivative_scatter_matches_reference() -> None:
    root_shape = i3(3, 3, 3)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    block_counts = i3(4, 4, 4)
    backing = patterned((27, 2, 4, 4, 4))
    field_ids = np.array([1, 0], dtype=np.int64)
    modes = np.zeros((2, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    chunk_ids = np.empty(27, dtype=np.int64)
    workspace = np.full((27, 2, 6, 6, 6), np.nan)
    derivative_workspace = np.empty((27, 1, 4, 4, 4), dtype=np.float64)
    sink = np.full((27, 1, 4, 4, 4), np.nan)
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
        central_difference_into(
            workspace[:primary_count],
            i3(0, 0, 0),
            i3(6, 6, 6),
            i3(1, 1, 1),
            i3(5, 5, 5),
            0,
            0,
            np.ones(3),
            derivative_workspace[:primary_count],
            0,
            i3(0, 0, 0),
        )
        scatter_blocks_from(
            derivative_workspace[:primary_count],
            i3(0, 0, 0),
            block_counts,
            chunk_ids[:primary_count],
            np.array([0], dtype=np.int64),
            sink,
            i3(0, 0, 0),
        )
        first += primary_count

    all_ids = np.arange(27, dtype=np.int64)
    full = np.full((27, 2, 6, 6, 6), np.nan)
    full[:, :, 1:5, 1:5, 1:5] = backing[:, field_ids]
    fill_physical_halos(full, i3(1, 1, 1), i3(5, 5, 5), all_ids, neighbors, modes, normals)
    fill_same_level_halos(full, i3(1, 1, 1), i3(5, 5, 5), all_ids, 27, neighbors, modes, normals)
    expected = np.empty_like(sink)
    reference(
        full,
        i3(1, 1, 1),
        i3(5, 5, 5),
        0,
        0,
        np.ones(3),
        expected,
        0,
        i3(0, 0, 0),
    )
    assert_bits_equal(sink, expected)
