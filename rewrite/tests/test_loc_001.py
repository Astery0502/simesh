from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import (
    fill_refined_point_leaf_ids,
    refined_point_leaf_ids,
)
from simesh_rewrite.point_location_reference import (
    refined_point_leaf_ids_reference,
)
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.sampling_reference import _source_cell_reference


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def make_flags(
    root_shape: np.ndarray,
    refine: Callable[[int, tuple[int, int, int]], bool],
) -> np.ndarray:
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


def chain_flags(internal_levels: int) -> np.ndarray:
    def visit(remaining: int) -> list[bool]:
        if remaining == 0:
            return [True]
        return [False, *visit(remaining - 1), *([True] * 7)]

    return np.asarray(visit(internal_levels), dtype=np.bool_)


def make_forest(
    root_shape: np.ndarray,
    refine: Callable[[int, tuple[int, int, int]], bool],
) -> tuple[np.ndarray, RefinedForest]:
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
    return coord_to_rank, forest


def locator_arguments(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    forest: RefinedForest,
) -> tuple[object, ...]:
    domain_cell_counts = np.ascontiguousarray(
        root_shape * block_cell_counts, dtype=np.int64
    )
    return (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
    )


def reference_owners(
    arguments: tuple[object, ...],
    forest: RefinedForest,
    points: np.ndarray,
) -> np.ndarray:
    return refined_point_leaf_ids_reference(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        points,
    )


def all_leaf_bounds(
    arguments: tuple[object, ...],
    forest: RefinedForest,
) -> np.ndarray:
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, _ = refined_leaf_geometry(
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )
    return bounds


def test_mixed_depth_all_leaf_faces_and_nextafter_neighbors_match_scan() -> None:
    root_shape = i3(3, 2, 2)
    block_cells = i3(4, 3, 5)
    domain_lower = f3(-0.75, 1.1, -2.3)
    domain_upper = f3(4.25, 6.7, 8.9)

    def refine(level: int, coord: tuple[int, int, int]) -> bool:
        return (level == 1 and coord in {(0, 0, 0), (2, 1, 1)}) or (
            level == 2 and coord in {(0, 0, 0), (5, 3, 3)}
        )

    coord_to_rank, forest = make_forest(root_shape, refine)
    arguments = locator_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    bounds = all_leaf_bounds(arguments, forest)
    points: list[np.ndarray] = []
    for leaf_bounds in bounds:
        center = np.ascontiguousarray(
            leaf_bounds[0] + 0.5 * (leaf_bounds[1] - leaf_bounds[0]),
            dtype=np.float64,
        )
        points.append(center)
        points.append(leaf_bounds[0].copy())
        for axis in range(3):
            at_lower = center.copy()
            at_lower[axis] = leaf_bounds[0, axis]
            points.append(at_lower)
            below_lower = at_lower.copy()
            below_lower[axis] = np.nextafter(
                leaf_bounds[0, axis], -np.inf
            )
            points.append(below_lower)

            at_upper = center.copy()
            at_upper[axis] = leaf_bounds[1, axis]
            points.append(at_upper)
            below_upper = at_upper.copy()
            below_upper[axis] = np.nextafter(
                leaf_bounds[1, axis], leaf_bounds[0, axis]
            )
            points.append(below_upper)
            above_upper = at_upper.copy()
            above_upper[axis] = np.nextafter(
                leaf_bounds[1, axis], np.inf
            )
            points.append(above_upper)

    point_array = np.ascontiguousarray(points, dtype=np.float64)
    expected = reference_owners(arguments, forest, point_array)
    actual = refined_point_leaf_ids(*arguments, point_array)
    assert np.array_equal(actual, expected)
    assert actual.nbytes == 8 * point_array.shape[0]

    reordered = np.ascontiguousarray(
        point_array[np.arange(point_array.shape[0] - 1, -1, -3)],
        dtype=np.float64,
    )
    repeated = np.ascontiguousarray(
        np.concatenate((reordered, reordered[:12], reordered[:12])),
        dtype=np.float64,
    )
    assert np.array_equal(
        refined_point_leaf_ids(*arguments, repeated),
        reference_owners(arguments, forest, repeated),
    )


def test_exact_root_child_face_edge_and_corner_ties_choose_upper_side() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(4, 4, 4)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    coord_to_rank, forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    arguments = locator_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    leaf_nodes = forest.leaf_node_ids
    descriptors = {
        (
            int(forest.node_levels[node]),
            tuple(int(value) for value in forest.node_coords[node]),
        ): leaf
        for leaf, node in enumerate(leaf_nodes)
    }
    points = np.ascontiguousarray(
        [
            [1.0, 0.25, 0.25],
            [1.0, 1.0, 0.25],
            [1.0, 1.0, 1.0],
            [2.0, 0.25, 0.25],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    owners = refined_point_leaf_ids(*arguments, points)
    assert owners.tolist() == [
        descriptors[(2, (1, 0, 0))],
        descriptors[(2, (1, 1, 0))],
        descriptors[(2, (1, 1, 1))],
        descriptors[(1, (1, 0, 0))],
        descriptors[(1, (1, 0, 0))],
    ]
    assert np.array_equal(owners, reference_owners(arguments, forest, points))


def test_level_one_reduces_to_sam_002_highest_native_face_owner() -> None:
    root_shape = i3(3, 2, 2)
    block_cells = i3(3, 4, 2)
    domain_cells = root_shape * block_cells
    domain_lower = f3(-1.25, 2.0, -4.5)
    domain_upper = f3(5.75, 9.0, 3.5)
    coord_to_rank, forest = make_forest(
        root_shape, lambda _level, _coord: False
    )
    arguments = locator_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    native_spacing = (domain_upper - domain_lower) / domain_cells
    points = [domain_lower.copy()]
    for axis in range(3):
        for face_index in range(1, int(domain_cells[axis])):
            point = domain_lower + 0.37 * (domain_upper - domain_lower)
            point[axis] = np.float64(
                domain_lower[axis]
                + np.float64(face_index) * native_spacing[axis]
            )
            points.extend(
                (
                    point.copy(),
                    np.where(
                        np.arange(3) == axis,
                        np.nextafter(point, -np.inf),
                        point,
                    ).astype(np.float64),
                    np.where(
                        np.arange(3) == axis,
                        np.nextafter(point, np.inf),
                        point,
                    ).astype(np.float64),
                )
            )
    point_array = np.ascontiguousarray(points, dtype=np.float64)
    expected = np.empty(point_array.shape[0], dtype=np.int64)
    for point_index, point in enumerate(point_array):
        root_coordinate = []
        for axis in range(3):
            native_cell = _source_cell_reference(
                float(point[axis]),
                float(domain_lower[axis]),
                float(domain_upper[axis]),
                float(native_spacing[axis]),
                int(domain_cells[axis]),
            )
            root_coordinate.append(native_cell // int(block_cells[axis]))
        expected[point_index] = coord_to_rank[tuple(root_coordinate)]

    actual = refined_point_leaf_ids(*arguments, point_array)
    assert np.array_equal(actual, expected)


def test_half_open_domain_repeated_exterior_and_empty_queries() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(2, 3, 4)
    domain_lower = f3(-1.0, 2.0, 5.0)
    domain_upper = f3(3.0, 8.0, 13.0)
    coord_to_rank, forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    arguments = locator_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    points = np.ascontiguousarray(
        [
            domain_lower,
            domain_lower,
            [domain_upper[0], 3.0, 6.0],
            [0.0, domain_upper[1], 6.0],
            [0.0, 3.0, domain_upper[2]],
            [np.nextafter(domain_lower[0], -np.inf), 3.0, 6.0],
            [np.nextafter(domain_upper[0], -np.inf), 3.0, 6.0],
            [0.0, 3.0, np.nextafter(domain_upper[2], np.inf)],
        ],
        dtype=np.float64,
    )
    expected = reference_owners(arguments, forest, points)
    actual = refined_point_leaf_ids(*arguments, points)
    assert np.array_equal(actual, expected)
    assert actual[0] == actual[1]
    assert actual[2:6].tolist() == [-1, -1, -1, -1]
    assert actual[-1] == -1
    assert actual[-2] >= 0

    empty = np.empty((0, 3), dtype=np.float64)
    empty_output = np.empty(0, dtype=np.int64)
    assert fill_refined_point_leaf_ids(*arguments, empty, empty_output) is None
    assert refined_point_leaf_ids(*arguments, empty).shape == (0,)


def test_large_origin_exact_upper_is_exterior_and_nextafter_is_owned() -> None:
    root_shape = i3(26, 1, 1)
    block_cells = i3(32, 4, 4)
    domain_lower = f3(274877906944.0, 0.0, 0.0)
    domain_upper = f3(274877906944.1, 1.0, 1.0)
    coord_to_rank, forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (25, 0, 0),
    )
    arguments = locator_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    points = np.ascontiguousarray(
        [
            [domain_upper[0], 0.75, 0.75],
            [np.nextafter(domain_upper[0], -np.inf), 0.75, 0.75],
            [domain_lower[0], 0.0, 0.0],
            [np.nextafter(domain_lower[0], -np.inf), 0.75, 0.75],
        ],
        dtype=np.float64,
    )
    actual = refined_point_leaf_ids(*arguments, points)
    expected = reference_owners(arguments, forest, points)
    assert np.array_equal(actual, expected)
    assert actual.tolist() == [-1, 32, 0, -1]


def test_level_63_chain_is_accepted_without_depth_sized_scratch() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        chain_flags(62),
    )
    assert forest.max_level == 63
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(float(1 << 62), 1.0, 1.0)
    block_cells = i3(1, 1, 1)
    arguments = locator_arguments(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    points = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, np.ldexp(1.0, -63), np.ldexp(1.0, -63)],
            [float(1 << 61), 0.75, 0.75],
            [np.nextafter(domain_upper[0], -np.inf), 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(
        refined_point_leaf_ids(*arguments, points),
        reference_owners(arguments, forest, points),
    )


def test_validation_errors_are_atomic_and_read_only_inputs_are_accepted() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(2, 2, 2)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    coord_to_rank, forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    arguments = list(
        locator_arguments(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells,
            coord_to_rank,
            forest,
        )
    )
    points = np.ascontiguousarray(
        [[0.25, 0.25, 0.25], [3.75, 1.75, 1.75], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    output = np.full(points.shape[0], -77, dtype=np.int64)
    before = output.copy()

    nonfinite = points.copy()
    nonfinite[-1, 2] = np.nan
    with pytest.raises(ValueError, match="point 2, axis 2"):
        fill_refined_point_leaf_ids(*arguments, nonfinite, output)
    assert np.array_equal(output, before)

    bad_arguments = arguments.copy()
    bad_arguments[5] = np.int64(forest.max_level)
    with pytest.raises(TypeError, match="exact Python int"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)

    bad_arguments = arguments.copy()
    bad_arguments[5] = 0
    with pytest.raises(ValueError, match="positive"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)

    bad_arguments = arguments.copy()
    bad_arguments[5] = 64
    with pytest.raises(OverflowError, match="deepest"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)

    bad_arguments = arguments.copy()
    bad_arguments[3] = i3(5, 2, 2)
    with pytest.raises(ValueError, match="must equal"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)

    bad_arguments = arguments.copy()
    bad_arguments[8] = forest.child_node_ids[:, :7].copy()
    with pytest.raises(ValueError, match="shape"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)

    with pytest.raises(TypeError, match="float64"):
        fill_refined_point_leaf_ids(
            *arguments, points.astype(np.float32), output
        )
    assert np.array_equal(output, before)

    noncontiguous_points = np.empty((3, 6), dtype=np.float64)[:, ::2]
    assert noncontiguous_points.shape == points.shape
    assert not noncontiguous_points.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        fill_refined_point_leaf_ids(
            *arguments, noncontiguous_points, output
        )
    assert np.array_equal(output, before)

    readonly_output = output.copy()
    readonly_output.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_refined_point_leaf_ids(
            *arguments, points, readonly_output
        )

    with pytest.raises(TypeError, match="int64"):
        fill_refined_point_leaf_ids(
            *arguments, points, output.astype(np.int32)
        )
    assert np.array_equal(output, before)

    aliased_output = points.view(np.int64).reshape(-1)[: points.shape[0]]
    aliased_before = aliased_output.copy()
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_point_leaf_ids(
            *arguments, points, aliased_output
        )
    assert np.array_equal(aliased_output, aliased_before)

    readonly_inputs = [
        value
        for value in (*arguments[:5], *arguments[6:], points)
        if isinstance(value, np.ndarray)
    ]
    expected_inputs = [value.copy() for value in readonly_inputs]
    for value in readonly_inputs:
        value.setflags(write=False)
    fill_refined_point_leaf_ids(*arguments, points, output)
    assert np.all(output >= 0)
    for value, expected in zip(readonly_inputs, expected_inputs, strict=True):
        assert np.array_equal(value, expected)


def test_count_overflow_and_subnormal_deepest_spacing_are_atomic() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, forest = make_forest(
        root_shape,
        lambda level, _coord: level == 1,
    )
    arguments = list(
        locator_arguments(
            f3(0.0, 0.0, 0.0),
            f3(1.0, 1.0, 1.0),
            root_shape,
            i3(1, 1, 1),
            coord_to_rank,
            forest,
        )
    )
    points = np.ascontiguousarray([[0.25, 0.25, 0.25]], dtype=np.float64)
    output = np.full(1, -91, dtype=np.int64)
    before = output.copy()

    bad_arguments = arguments.copy()
    bad_arguments[2] = i3(np.iinfo(np.int64).max, 1, 1)
    bad_arguments[3] = i3(np.iinfo(np.int64).max, 1, 1)
    bad_arguments[4] = i3(2, 1, 1)
    with pytest.raises(OverflowError, match="root block-cell"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)

    bad_arguments = arguments.copy()
    bad_arguments[0] = f3(0.0, 0.0, 0.0)
    bad_arguments[1] = f3(np.finfo(np.float64).tiny, 1.0, 1.0)
    bad_arguments[5] = 2
    with pytest.raises(ValueError, match="normal"):
        fill_refined_point_leaf_ids(*bad_arguments, points, output)
    assert np.array_equal(output, before)
