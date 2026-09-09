from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.geometry import level1_block_geometry
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_geometry import (
    fill_refined_leaf_geometry,
    refined_leaf_geometry,
)
from simesh_rewrite.refined_geometry_reference import (
    refined_leaf_geometry_reference,
)


EPS = np.finfo(np.float64).eps
SMALLEST_SUBNORMAL = np.nextafter(np.float64(0.0), np.float64(1.0))


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def make_flags(root_shape: np.ndarray, refine) -> np.ndarray:
    _, root_coords = level1_morton(root_shape)
    flags: list[bool] = []

    def visit(level: int, coord: tuple[int, int, int]) -> None:
        split = bool(refine(level, coord))
        flags.append(not split)
        if not split:
            return
        for child in range(8):
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            visit(
                level + 1,
                tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
            )

    for root_coord in root_coords:
        visit(1, tuple(int(value) for value in root_coord))
    return np.asarray(flags, dtype=np.bool_)


def make_chain_flags(internal_levels: int) -> np.ndarray:
    def visit(remaining: int) -> list[bool]:
        if remaining == 0:
            return [True]
        return [False, *visit(remaining - 1), *([True] * 7)]

    return np.asarray(visit(internal_levels), dtype=np.bool_)


def make_forest(root_shape: np.ndarray, refine) -> RefinedForest:
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        make_flags(root_shape, refine),
    )
    assert validate_refined_forest_arrays(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    ) == forest.max_level
    return forest


def make_chain_forest(internal_levels: int) -> RefinedForest:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        make_chain_flags(internal_levels),
    )
    assert validate_refined_forest_arrays(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    ) == forest.max_level
    return forest


def geometry_arguments(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    block_cell_counts: np.ndarray,
    forest: RefinedForest,
    leaf_ids: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return (
        domain_lower,
        domain_upper,
        root_shape,
        np.ascontiguousarray(root_shape * block_cell_counts, dtype=np.int64),
        block_cell_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )


def axis_tolerance(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> np.ndarray:
    extent = domain_upper - domain_lower
    return np.maximum(
        16.0
        * EPS
        * np.maximum.reduce(
            (np.abs(domain_lower), np.abs(domain_upper), extent)
        ),
        16.0 * SMALLEST_SUBNORMAL,
    )


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_level_one_reduces_bitwise_to_geo_001() -> None:
    root_shape = i3(3, 2, 4)
    block_cells = i3(2, 3, 5)
    domain_cells = root_shape * block_cells
    domain_lower = f3(-0.7, 1.25, -3.5)
    domain_upper = f3(2.1, 7.75, 8.25)
    forest = make_forest(root_shape, lambda _level, _coord: False)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)

    actual_bounds, actual_spacing = refined_leaf_geometry(
        *geometry_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            forest,
            leaf_ids,
        )
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    expected_bounds, expected_spacing = level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        leaf_ids,
    )

    assert_bits_equal(actual_bounds, expected_bounds)
    assert_bits_equal(
        actual_spacing,
        np.broadcast_to(expected_spacing, actual_spacing.shape).copy(),
    )


def test_mixed_depth_geometry_matches_fraction_reference() -> None:
    root_shape = i3(3, 2, 2)
    block_cells = i3(2, 3, 4)
    domain_lower = f3(-0.75, 1.1, -2.3)
    domain_upper = f3(2.0, 4.7, 7.2)
    forest = make_forest(
        root_shape,
        lambda level, coord: (
            level == 1 and coord in {(0, 0, 0), (2, 1, 1)}
        )
        or (level == 2 and coord in {(0, 0, 0), (5, 3, 3)}),
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)[::-1].copy()
    arguments = geometry_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        forest,
        leaf_ids,
    )

    actual_bounds, actual_spacing = refined_leaf_geometry(*arguments)
    expected_bounds, expected_spacing = refined_leaf_geometry_reference(
        domain_lower,
        domain_upper,
        root_shape,
        arguments[3],
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )

    np.testing.assert_allclose(
        actual_spacing,
        expected_spacing,
        rtol=8.0 * EPS,
        atol=0.0,
    )
    tolerance = axis_tolerance(domain_lower, domain_upper)
    assert np.all(np.abs(actual_bounds - expected_bounds) <= tolerance)
    assert actual_bounds.nbytes + actual_spacing.nbytes == 72 * leaf_ids.size


def test_reordered_repeated_and_empty_leaf_selection() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(3, 2, 4)
    domain_lower = f3(-1.0, 2.0, 0.0)
    domain_upper = f3(3.0, 6.0, 8.0)
    forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    leaf_ids = np.array([8, 0, 8, 5], dtype=np.int64)
    arguments = geometry_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        forest,
        leaf_ids,
    )

    bounds, spacing = refined_leaf_geometry(*arguments)
    assert_bits_equal(bounds[0], bounds[2])
    assert_bits_equal(spacing[0], spacing[2])

    one_bounds, one_spacing = refined_leaf_geometry(
        *geometry_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            forest,
            np.array([8], dtype=np.int64),
        )
    )
    assert_bits_equal(bounds[0], one_bounds[0])
    assert_bits_equal(spacing[0], one_spacing[0])

    empty_bounds, empty_spacing = refined_leaf_geometry(
        *geometry_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            forest,
            np.empty(0, dtype=np.int64),
        )
    )
    assert empty_bounds.shape == (0, 2, 3)
    assert empty_spacing.shape == (0, 3)


def test_same_level_and_coarse_fine_faces_are_bit_identical() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(4, 3, 2)
    domain_lower = f3(-1.25, 2.0, -4.0)
    domain_upper = f3(3.5, 8.0, 2.0)
    forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        *geometry_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            forest,
            leaf_ids,
        )
    )
    leaf_nodes = forest.leaf_node_ids
    levels = forest.node_levels[leaf_nodes]
    coords = forest.node_coords[leaf_nodes]
    by_descriptor = {
        (int(levels[leaf]), tuple(int(value) for value in coords[leaf])): leaf
        for leaf in range(leaf_ids.size)
    }

    same_left = by_descriptor[(2, (0, 0, 0))]
    same_right = by_descriptor[(2, (1, 0, 0))]
    coarse = by_descriptor[(1, (1, 0, 0))]

    assert_bits_equal(bounds[same_left, 1, 0:1], bounds[same_right, 0, 0:1])
    assert_bits_equal(bounds[same_right, 1, 0:1], bounds[coarse, 0, 0:1])
    assert_bits_equal(spacing[coarse], np.ldexp(spacing[same_right], 1))
    for axis in range(3):
        assert_bits_equal(
            np.asarray([np.min(bounds[:, 0, axis])]),
            np.asarray([domain_lower[axis]]),
        )
        assert_bits_equal(
            np.asarray([np.max(bounds[:, 1, axis])]),
            np.asarray([domain_upper[axis]]),
        )


def test_synthetic_mixed_depth_geometry_matches_current_rnode() -> None:
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh

    root_shape = i3(2, 2, 1)
    block_cells = i3(4, 6, 8)
    domain_cells = root_shape * block_cells
    domain_lower = f3(-0.7, 1.1, -2.3)
    domain_upper = f3(1.9, 4.7, 7.2)

    def refine(level, coord):
        return level == 1 and coord in {(0, 0, 0), (1, 1, 0)}

    flags = make_flags(root_shape, refine)
    forest = make_forest(root_shape, refine)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        *geometry_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            forest,
            leaf_ids,
        )
    )
    current_forest = AMRForest(
        3,
        *tuple(int(value) for value in root_shape),
        flags.astype(np.int32),
    )
    current_mesh = AMRMesh(
        3,
        block_cells.astype(np.uint32),
        domain_cells.astype(np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(0),
        np.uint32(1),
        current_forest,
    )
    current_rnode = np.asarray(current_mesh.rnode)
    tolerance = axis_tolerance(domain_lower, domain_upper)
    assert np.all(
        np.abs(bounds - current_rnode[:, :6].reshape(-1, 2, 3)) <= tolerance
    )
    np.testing.assert_allclose(
        spacing,
        current_rnode[:, 6:9],
        rtol=8.0 * EPS,
        atol=0.0,
    )


def test_level_63_geometry_is_representable_and_exact() -> None:
    forest = make_chain_forest(62)
    deepest_level = np.flatnonzero(
        forest.node_levels[forest.leaf_node_ids] == 63
    ).astype(np.int64)
    assert deepest_level.size == 8
    deepest = deepest_level[:1]
    root_shape = i3(1, 1, 1)
    block_cells = i3(1, 1, 1)

    bounds, spacing = refined_leaf_geometry(
        *geometry_arguments(
            f3(0.0, 0.0, 0.0),
            f3(float(1 << 62), 1.0, 1.0),
            root_shape,
            block_cells,
            forest,
            deepest,
        )
    )

    assert spacing[0, 0] == 1.0
    assert bounds[0, 0, 0] == 0.0
    assert bounds[0, 1, 0] == 1.0
    assert np.all(np.isfinite(bounds))
    assert np.all(spacing >= np.finfo(np.float64).tiny)


def test_overflow_subnormal_and_collapsed_bounds_are_atomic() -> None:
    root_shape = i3(1, 1, 1)
    deep_forest = make_chain_forest(62)
    deep_leaf = np.flatnonzero(
        deep_forest.node_levels[deep_forest.leaf_node_ids] == 63
    ).astype(np.int64)[:1]
    bounds = np.full((1, 2, 3), -7.0, dtype=np.float64)
    spacing = np.full((1, 3), -9.0, dtype=np.float64)
    before_bounds = bounds.copy()
    before_spacing = spacing.copy()

    with pytest.raises(OverflowError):
        fill_refined_leaf_geometry(
            *geometry_arguments(
                f3(0.0, 0.0, 0.0),
                f3(float(1 << 63), 1.0, 1.0),
                root_shape,
                i3(2, 1, 1),
                deep_forest,
                deep_leaf,
            ),
            bounds,
            spacing,
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    refined = make_forest(
        root_shape,
        lambda level, _coord: level == 1,
    )
    selected = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *geometry_arguments(
                f3(0.0, 0.0, 0.0),
                f3(np.finfo(np.float64).tiny, 1.0, 1.0),
                root_shape,
                i3(1, 1, 1),
                refined,
                selected,
            ),
            bounds,
            spacing,
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    level_one = make_forest(root_shape, lambda _level, _coord: False)
    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *geometry_arguments(
                f3(0.0, 0.0, 0.0),
                f3(float(np.finfo(np.float64).tiny / 2.0), 1.0, 1.0),
                root_shape,
                i3(1, 1, 1),
                level_one,
                selected,
            ),
            bounds,
            spacing,
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    collapsed = make_chain_forest(4)
    collapsed_leaf = np.flatnonzero(
        collapsed.node_levels[collapsed.leaf_node_ids] == 5
    ).astype(np.int64)[:1]
    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *geometry_arguments(
                f3(1.0e16, 0.0, 0.0),
                f3(1.0e16 + 16.0, 1.0, 1.0),
                root_shape,
                i3(1, 1, 1),
                collapsed,
                collapsed_leaf,
            ),
            bounds,
            spacing,
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)


def test_invalid_inputs_and_output_overlap_preserve_outputs() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(2, 2, 2)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(1.0, 1.0, 1.0)
    forest = make_forest(root_shape, lambda _level, _coord: False)
    leaf_ids = np.array([0, 1], dtype=np.int64)
    arguments = geometry_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        forest,
        leaf_ids,
    )
    bounds = np.full((2, 2, 3), -7.0, dtype=np.float64)
    spacing = np.full((2, 3), -9.0, dtype=np.float64)
    before_bounds = bounds.copy()
    before_spacing = spacing.copy()

    bad_ids = np.array([0, 2], dtype=np.int64)
    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *arguments[:-1], bad_ids, bounds, spacing
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    with pytest.raises(TypeError, match="int64"):
        fill_refined_leaf_geometry(
            *arguments[:-1], leaf_ids.astype(np.int32), bounds, spacing
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *arguments[:6], forest.node_coords[:, :2], *arguments[7:], bounds, spacing
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    with pytest.raises(TypeError):
        fill_refined_leaf_geometry(
            *arguments,
            bounds.astype(np.float32),
            spacing,
        )
    assert_bits_equal(bounds, before_bounds)
    assert_bits_equal(spacing, before_spacing)

    wrong_shape_bounds = np.full((2, 2, 2), -15.0, dtype=np.float64)
    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *arguments,
            wrong_shape_bounds,
            spacing,
        )
    assert np.all(wrong_shape_bounds == -15.0)
    assert_bits_equal(spacing, before_spacing)

    readonly_bounds = bounds.copy()
    readonly_bounds.setflags(write=False)
    with pytest.raises(ValueError):
        fill_refined_leaf_geometry(
            *arguments,
            readonly_bounds,
            spacing,
        )
    assert_bits_equal(spacing, before_spacing)

    shared = np.full(12, -11.0, dtype=np.float64)
    overlapping_bounds = shared.reshape(2, 2, 3)
    overlapping_spacing = shared[:6].reshape(2, 3)
    shared_before = shared.copy()
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_leaf_geometry(
            *arguments,
            overlapping_bounds,
            overlapping_spacing,
        )
    assert_bits_equal(shared, shared_before)

    raw = np.zeros(6, dtype=np.int64)
    overlapping_ids = raw[:1]
    overlapping_input_bounds = raw.view(np.float64).reshape(1, 2, 3)
    input_overlap_spacing = np.full((1, 3), -13.0)
    raw_before = raw.copy()
    one_arguments = geometry_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        forest,
        overlapping_ids,
    )
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_leaf_geometry(
            *one_arguments,
            overlapping_input_bounds,
            input_overlap_spacing,
        )
    assert np.array_equal(raw, raw_before)


def test_representative_weno_geometry_metadata_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")

    from simesh.amrvac import open_dataset
    from simesh.amrvac.datio import get_metadata

    header, flags_input, tree = get_metadata(str(path))
    assert header["geometry"] == "Cartesian_3D"
    assert bool(header["staggered"])
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags_input, dtype=np.bool_),
    )
    assert validate_refined_forest_arrays(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    ) == forest.max_level
    leaf_nodes = forest.leaf_node_ids
    assert np.array_equal(
        forest.node_levels[leaf_nodes], np.asarray(tree[0], dtype=np.int64)
    )
    assert np.array_equal(
        forest.node_coords[leaf_nodes] + 1,
        np.asarray(tree[1], dtype=np.int64),
    )

    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    domain_lower = np.ascontiguousarray(header["xmin"], dtype=np.float64)
    domain_upper = np.ascontiguousarray(header["xmax"], dtype=np.float64)
    block_cells = np.ascontiguousarray(header["block_nx"], dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        *geometry_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            forest,
            leaf_ids,
        )
    )

    current = open_dataset(str(path))
    current_rnode = np.asarray(current.mesh.rnode)
    np.testing.assert_allclose(
        spacing,
        current_rnode[:, 6:9],
        rtol=8.0 * EPS,
        atol=0.0,
    )
    tolerance = axis_tolerance(domain_lower, domain_upper)
    assert np.all(
        np.abs(bounds - current_rnode[:, :6].reshape(-1, 2, 3)) <= tolerance
    )
    for axis in range(3):
        assert_bits_equal(
            np.asarray([np.min(bounds[:, 0, axis])]),
            np.asarray([domain_lower[axis]]),
        )
        assert_bits_equal(
            np.asarray([np.max(bounds[:, 1, axis])]),
            np.asarray([domain_upper[axis]]),
        )
