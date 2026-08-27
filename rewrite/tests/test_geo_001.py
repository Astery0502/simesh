from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.geometry import (
    fill_level1_block_geometry,
    level1_block_geometry,
)
from simesh_rewrite.geometry_reference import (
    cell_center_reference,
    level1_block_geometry_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


EPS = np.finfo(np.float64).eps
SMALLEST_SUBNORMAL = np.nextafter(np.float64(0.0), np.float64(1.0))


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def axis_tolerance(domain_lower: np.ndarray, domain_upper: np.ndarray) -> np.ndarray:
    extent = domain_upper - domain_lower
    return np.maximum(
        8.0 * EPS * np.maximum.reduce(
            (np.abs(domain_lower), np.abs(domain_upper), extent)
        ),
        8.0 * SMALLEST_SUBNORMAL,
    )


def test_selected_slots_preserve_order_duplicates_and_empty_selection() -> None:
    root_shape = i3(4, 2, 2)
    block_cells = i3(2, 3, 4)
    domain_cells = root_shape * block_cells
    domain_lower = np.array([-1.0, 2.0, 0.0])
    domain_upper = np.array([1.0, 8.0, 4.0])
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    block_ids = np.array([7, 0, 7, 3], dtype=np.int64)
    bounds, spacing = level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        block_ids,
    )
    assert np.array_equal(spacing, [0.25, 1.0, 0.5])
    assert np.array_equal(bounds[0], bounds[2])
    assert np.array_equal(bounds[1, 0], domain_lower)
    assert np.array_equal(bounds[3, 0], [-0.5, 5.0, 0.0])

    empty_bounds, empty_spacing = level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        np.empty(0, dtype=np.int64),
    )
    assert empty_bounds.shape == (0, 2, 3)
    assert np.array_equal(empty_spacing, spacing)


def test_compiled_geometry_matches_exact_rational_reference() -> None:
    cases = [
        (
            i3(3, 2, 5),
            i3(4, 6, 8),
            np.array([-0.7, 1.1, -2.3]),
            np.array([1.9, 4.7, 7.2]),
        ),
        (
            i3(5, 3, 2),
            i3(7, 5, 9),
            np.array([1000.25, -20.5, 1.0e-4]),
            np.array([1003.75, -5.25, 0.75]),
        ),
    ]
    for root_shape, block_cells, domain_lower, domain_upper in cases:
        domain_cells = root_shape * block_cells
        coord_to_rank, rank_to_coord = level1_morton(root_shape)
        block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)[::-1].copy()
        actual_bounds, actual_spacing = level1_block_geometry(
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            block_ids,
        )
        expected_bounds, expected_spacing = level1_block_geometry_reference(
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            rank_to_coord,
            block_ids,
        )
        np.testing.assert_allclose(
            actual_spacing,
            expected_spacing,
            rtol=8.0 * EPS,
            atol=0.0,
        )
        tolerance = axis_tolerance(domain_lower, domain_upper)
        assert np.all(np.abs(actual_bounds - expected_bounds) <= tolerance)


def test_topology_neighbors_share_bit_identical_faces_and_close_domain() -> None:
    root_shape = i3(3, 2, 5)
    block_cells = i3(4, 6, 8)
    domain_cells = root_shape * block_cells
    domain_lower = np.array([-0.7, 1.1, -2.3])
    domain_upper = np.array([1.9, 4.7, 7.2])
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    bounds, _ = level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        block_ids,
    )
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    for block_id, coordinate in enumerate(rank_to_coord):
        for axis in range(3):
            if coordinate[axis] == 0:
                assert bounds[block_id, 0, axis].tobytes() == domain_lower[axis].tobytes()
            if coordinate[axis] + 1 == root_shape[axis]:
                assert bounds[block_id, 1, axis].tobytes() == domain_upper[axis].tobytes()
            neighbor = neighbors[block_id, 2 * axis + 1]
            if neighbor >= 0:
                assert (
                    bounds[block_id, 1, axis].tobytes()
                    == bounds[neighbor, 0, axis].tobytes()
                )


def test_global_center_formula_is_partition_independent_and_inside_bounds() -> None:
    domain_lower = np.array([-53.988496872238656, -1.0, 2.0])
    domain_upper = np.array([17.81653554099192, 3.0, 10.0])
    domain_cells = i3(240, 12, 16)
    global_index = i3(223, 7, 9)
    spacing = (domain_upper - domain_lower) / domain_cells
    center_from_global = domain_lower + (global_index.astype(np.float64) + 0.5) * spacing
    expected = cell_center_reference(
        domain_lower,
        domain_upper,
        domain_cells,
        global_index,
    )
    assert np.all(
        np.abs(center_from_global - expected)
        <= axis_tolerance(domain_lower, domain_upper)
    )

    for block_size in (12, 120):
        coordinate, local_index = divmod(int(global_index[0]), block_size)
        reconstructed_global = coordinate * block_size + local_index
        assert reconstructed_global == global_index[0]
        assert (
            domain_lower[0] + (float(reconstructed_global) + 0.5) * spacing[0]
            == center_from_global[0]
        )


def test_current_geometry_agrees_with_declared_tolerance() -> None:
    root_shape = i3(3, 2, 5)
    block_cells = i3(4, 6, 8)
    domain_cells = root_shape * block_cells
    domain_lower = np.array([-0.7, 1.1, -2.3])
    domain_upper = np.array([1.9, 4.7, 7.2])
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    block_ids = np.arange(rank_to_coord.shape[0], dtype=np.int64)
    bounds, spacing = level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        block_ids,
    )

    forest = AMRForest(3, *tuple(root_shape), np.ones(block_ids.size, dtype=np.int32))
    mesh = AMRMesh(
        3,
        block_cells.astype(np.uint32),
        domain_cells.astype(np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(0),
        np.uint32(1),
        forest,
    )
    current = np.asarray(mesh.rnode)
    current_bounds = current[:, :6].reshape(-1, 2, 3)
    tolerance = axis_tolerance(domain_lower, domain_upper)
    assert np.all(np.abs(bounds - current_bounds) <= tolerance)
    np.testing.assert_allclose(
        np.broadcast_to(spacing, current[:, 6:9].shape),
        current[:, 6:9],
        rtol=8.0 * EPS,
        atol=0.0,
    )


def test_large_cell_count_uses_the_canonical_last_center_formula() -> None:
    large_count = (1 << 53) + 2
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    bounds, spacing = level1_block_geometry(
        np.array([0.0, 0.0, 0.0]),
        np.array([float(large_count), 1.0, 1.0]),
        i3(large_count, 1, 1),
        i3(large_count, 1, 1),
        coord_to_rank,
        rank_to_coord,
        np.array([0], dtype=np.int64),
    )
    assert spacing[0] == 1.0
    assert bounds[0, 0, 0] == 0.0
    assert bounds[0, 1, 0] == float(large_count)
    last_index = large_count - 1
    canonical_last_center = (float(last_index) + 0.5) * spacing[0]
    assert canonical_last_center < bounds[0, 1, 0]


def test_invalid_inputs_leave_outputs_unchanged() -> None:
    root_shape = i3(4, 1, 1)
    block_cells = i3(1, 1, 1)
    domain_cells = root_shape * block_cells
    domain_lower = np.array([0.0, 0.0, 0.0])
    domain_upper = np.array([1.0, 1.0, 1.0])
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    block_ids = np.array([0, 3], dtype=np.int64)
    bounds = np.full((2, 2, 3), -7.0)
    spacing = np.full(3, -9.0)
    before_bounds = bounds.copy()
    before_spacing = spacing.copy()

    bad_ids = np.array([0, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="slot 1"):
        fill_level1_block_geometry(
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            bad_ids,
            bounds,
            spacing,
        )
    assert np.array_equal(bounds, before_bounds)
    assert np.array_equal(spacing, before_spacing)

    broken_forward = coord_to_rank.copy()
    broken_forward[0, 0, 0] = 3
    with pytest.raises(ValueError, match="slot 0"):
        fill_level1_block_geometry(
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            broken_forward,
            rank_to_coord,
            block_ids,
            bounds,
            spacing,
        )

    subnormal_lower = np.array([0.0, 0.0, 0.0])
    subnormal_upper = np.array([2.0 * SMALLEST_SUBNORMAL, 1.0, 1.0])
    subnormal_root = i3(3, 1, 1)
    subnormal_forward, subnormal_inverse = level1_morton(subnormal_root)
    with pytest.raises(ValueError, match="normal"):
        fill_level1_block_geometry(
            subnormal_lower,
            subnormal_upper,
            subnormal_root,
            i3(1, 1, 1),
            subnormal_forward,
            subnormal_inverse,
            np.array([1], dtype=np.int64),
            np.empty((1, 2, 3), dtype=np.float64),
            np.empty(3, dtype=np.float64),
        )

    accumulated_subnormal_upper = np.array(
        [249.0 * SMALLEST_SUBNORMAL, 1.0, 1.0]
    )
    accumulated_root = i3(100, 1, 1)
    accumulated_forward, accumulated_inverse = level1_morton(accumulated_root)
    with pytest.raises(ValueError, match="normal"):
        fill_level1_block_geometry(
            subnormal_lower,
            accumulated_subnormal_upper,
            accumulated_root,
            i3(1, 1, 1),
            accumulated_forward,
            accumulated_inverse,
            np.array([50], dtype=np.int64),
            np.empty((1, 2, 3), dtype=np.float64),
            np.empty(3, dtype=np.float64),
        )

    reversed_subnormal_upper = np.array(
        [10.0 * SMALLEST_SUBNORMAL, 1.0, 1.0]
    )
    reversed_root = i3(19, 1, 1)
    reversed_forward, reversed_inverse = level1_morton(reversed_root)
    with pytest.raises(ValueError, match="normal"):
        fill_level1_block_geometry(
            subnormal_lower,
            reversed_subnormal_upper,
            reversed_root,
            i3(1, 1, 1),
            reversed_forward,
            reversed_inverse,
            np.array([11], dtype=np.int64),
            np.empty((1, 2, 3), dtype=np.float64),
            np.empty(3, dtype=np.float64),
        )

    large_offset_lower = np.array([1.0e16, 0.0, 0.0])
    large_offset_upper = np.array([1.0e16 + 16.0, 1.0, 1.0])
    one_block_forward, one_block_inverse = level1_morton(i3(1, 1, 1))
    with pytest.raises(ValueError, match="representable"):
        fill_level1_block_geometry(
            large_offset_lower,
            large_offset_upper,
            i3(8, 1, 1),
            i3(8, 1, 1),
            one_block_forward,
            one_block_inverse,
            np.array([0], dtype=np.int64),
            np.empty((1, 2, 3), dtype=np.float64),
            np.empty(3, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="divisible"):
        fill_level1_block_geometry(
            domain_lower,
            domain_upper,
            i3(5, 1, 1),
            i3(2, 1, 1),
            coord_to_rank,
            rank_to_coord,
            block_ids,
            bounds,
            spacing,
        )
    with pytest.raises(ValueError, match="greater"):
        fill_level1_block_geometry(
            domain_upper,
            domain_lower,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            block_ids,
            bounds,
            spacing,
        )

    collapsed_lower = np.array([1.0e16, 0.0, 0.0])
    collapsed_upper = np.array([np.nextafter(1.0e16, np.inf), 1.0, 1.0])
    with pytest.raises(ValueError, match="representable"):
        fill_level1_block_geometry(
            collapsed_lower,
            collapsed_upper,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            block_ids,
            bounds,
            spacing,
        )

    readonly_spacing = spacing.copy()
    readonly_spacing.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_level1_block_geometry(
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            block_ids,
            bounds,
            readonly_spacing,
        )

    shared = np.empty(12, dtype=np.float64)
    overlapping_bounds = shared.reshape(2, 2, 3)
    overlapping_spacing = shared[:3]
    with pytest.raises(ValueError, match="must not overlap"):
        fill_level1_block_geometry(
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            coord_to_rank,
            rank_to_coord,
            block_ids,
            overlapping_bounds,
            overlapping_spacing,
        )
