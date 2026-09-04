from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from simesh_rewrite._point_location import (
    fill_refined_point_leaf_ids_with_hints_unchecked,
)
from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.hinted_location import (
    HintedLocationStats,
    fill_refined_point_leaf_ids_with_hints,
)
from simesh_rewrite.hinted_location_reference import (
    refined_point_leaf_ids_with_hints_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry


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


def mixed_fixture() -> tuple[tuple[object, ...], tuple[object, ...], RefinedForest]:
    root_shape = i3(3, 2, 2)
    block_cells = i3(4, 3, 5)
    domain_lower = f3(-0.75, 1.1, -2.3)
    domain_upper = f3(4.25, 6.7, 8.9)

    def refine(level: int, coord: tuple[int, int, int]) -> bool:
        return (level == 1 and coord in {(0, 0, 0), (2, 1, 1)}) or (
            level == 2 and coord in {(0, 0, 0), (5, 3, 3)}
        )

    coord_to_rank, forest = make_forest(root_shape, refine)
    domain_cells = np.ascontiguousarray(root_shape * block_cells, dtype=np.int64)
    loc_arguments = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
    )
    hinted_arguments = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    return loc_arguments, hinted_arguments, forest


def leaf_bounds(
    hinted_arguments: tuple[object, ...], forest: RefinedForest
) -> np.ndarray:
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, _ = refined_leaf_geometry(
        hinted_arguments[0],
        hinted_arguments[1],
        hinted_arguments[2],
        hinted_arguments[3],
        hinted_arguments[4],
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )
    return bounds


def call_reference(
    hinted_arguments: tuple[object, ...],
    points: np.ndarray,
    hints: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    return refined_point_leaf_ids_with_hints_reference(
        *hinted_arguments, points, hints
    )


def assert_matches_loc_and_reference(
    loc_arguments: tuple[object, ...],
    hinted_arguments: tuple[object, ...],
    points: np.ndarray,
    hints: np.ndarray,
) -> tuple[np.ndarray, HintedLocationStats]:
    output = np.full(points.shape[0], -77, dtype=np.int64)
    stats = fill_refined_point_leaf_ids_with_hints(
        *hinted_arguments, points, hints, output
    )
    loc = refined_point_leaf_ids(*loc_arguments, points)
    reference, reference_stats = call_reference(hinted_arguments, points, hints)
    assert np.array_equal(output, loc)
    assert np.array_equal(output, reference)
    assert tuple(stats) == reference_stats
    return output, stats


def test_empty_points_return_zero_stats() -> None:
    loc_arguments, hinted_arguments, _ = mixed_fixture()
    points = np.empty((0, 3), dtype=np.float64)
    hints = np.empty(0, dtype=np.int64)
    output = np.empty(0, dtype=np.int64)
    stats = fill_refined_point_leaf_ids_with_hints(
        *hinted_arguments, points, hints, output
    )
    assert isinstance(stats, HintedLocationStats)
    assert stats == HintedLocationStats(0, 0, 0, 0, 0, 0)
    assert np.array_equal(output, refined_point_leaf_ids(*loc_arguments, points))


def test_correct_absent_wrong_and_exterior_hints_have_exact_stats() -> None:
    loc_arguments, hinted_arguments, forest = mixed_fixture()
    bounds = leaf_bounds(hinted_arguments, forest)
    selected = [0, len(bounds) // 2, len(bounds) - 1]
    centers = [
        np.ascontiguousarray(bounds[leaf, 0] + 0.37 * (bounds[leaf, 1] - bounds[leaf, 0]))
        for leaf in selected
    ]
    points = np.ascontiguousarray(
        [
            centers[0],
            centers[1],
            centers[2],
            [np.nextafter(hinted_arguments[0][0], -np.inf), centers[0][1], centers[0][2]],
            [hinted_arguments[1][0], centers[0][1], centers[0][2]],
        ],
        dtype=np.float64,
    )
    wrong = next(
        leaf
        for leaf in range(len(bounds))
        if leaf != selected[2]
        and not (
            np.all(points[2] >= bounds[leaf, 0])
            and np.all(points[2] < bounds[leaf, 1])
        )
    )
    hints = np.asarray((selected[0], -1, wrong, selected[0], selected[0]), dtype=np.int64)
    output, stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, points, hints
    )
    assert output[0] == selected[0]
    assert output[2] == selected[2]
    assert output[3:].tolist() == [-1, -1]
    assert stats == HintedLocationStats(5, 3, 2, 4, 1, 2)


def test_long_same_leaf_run_and_one_transition_use_last_owner_hint() -> None:
    loc_arguments, hinted_arguments, forest = mixed_fixture()
    bounds = leaf_bounds(hinted_arguments, forest)
    leaf = len(bounds) // 3
    rng = np.random.default_rng(20260904)
    fractions = rng.uniform(0.1, 0.9, size=(128, 3))
    points = np.ascontiguousarray(
        bounds[leaf, 0] + fractions * (bounds[leaf, 1] - bounds[leaf, 0]),
        dtype=np.float64,
    )
    hints = np.full(points.shape[0], leaf, dtype=np.int64)
    output, stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, points, hints
    )
    assert np.all(output == leaf)
    assert stats == HintedLocationStats(128, 128, 0, 128, 128, 0)

    other = next(candidate for candidate in range(len(bounds)) if candidate != leaf)
    transition = np.ascontiguousarray(
        [points[0], bounds[other, 0] + 0.4 * (bounds[other, 1] - bounds[other, 0])],
        dtype=np.float64,
    )
    output, stats = assert_matches_loc_and_reference(
        loc_arguments,
        hinted_arguments,
        transition,
        np.asarray((leaf, leaf), dtype=np.int64),
    )
    assert output.tolist() == [leaf, other]
    assert stats == HintedLocationStats(2, 2, 0, 2, 1, 1)


def test_all_leaf_faces_and_nextafter_probes_match_half_open_reference() -> None:
    loc_arguments, hinted_arguments, forest = mixed_fixture()
    bounds = leaf_bounds(hinted_arguments, forest)
    points: list[np.ndarray] = []
    hints: list[int] = []
    expected_hits = 0
    for leaf, box in enumerate(bounds):
        center = np.ascontiguousarray(box[0] + 0.43 * (box[1] - box[0]))
        for axis in range(3):
            probes = (
                box[0, axis],
                np.nextafter(box[0, axis], np.inf),
                np.nextafter(box[0, axis], -np.inf),
                np.nextafter(box[1, axis], -np.inf),
                box[1, axis],
                np.nextafter(box[1, axis], np.inf),
            )
            for value in probes:
                point = center.copy()
                point[axis] = value
                points.append(point)
                hints.append(leaf)
                if np.all(point >= box[0]) and np.all(point < box[1]):
                    if np.all(point >= hinted_arguments[0]) and np.all(
                        point < hinted_arguments[1]
                    ):
                        expected_hits += 1
    point_array = np.ascontiguousarray(points, dtype=np.float64)
    hint_array = np.asarray(hints, dtype=np.int64)
    _, stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, point_array, hint_array
    )
    assert stats.hint_candidate_count == len(points)
    assert stats.hint_hit_count == expected_hits
    assert stats.hierarchy_fallback_count == (
        stats.interior_point_count - expected_hits
    )


def test_hinted_refined_face_edge_corner_ties_fall_back_to_upper_leaf() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(4, 4, 4)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    coord_to_rank, forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    domain_cells = np.ascontiguousarray(root_shape * block_cells, dtype=np.int64)
    loc_arguments = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
    )
    hinted_arguments = (
        *loc_arguments[:8],
        forest.node_levels,
        forest.node_coords,
        *loc_arguments[8:],
        forest.leaf_node_ids,
    )
    descriptors = {
        tuple(int(value) for value in forest.node_coords[int(node)]): leaf
        for leaf, node in enumerate(forest.leaf_node_ids)
        if int(forest.node_levels[int(node)]) == 2
    }
    lower_leaf = descriptors[(0, 0, 0)]
    upper_leaves = (
        descriptors[(1, 0, 0)],
        descriptors[(1, 1, 0)],
        descriptors[(1, 1, 1)],
    )
    tie_points = np.ascontiguousarray(
        ((1.0, 0.5, 0.5), (1.0, 1.0, 0.5), (1.0, 1.0, 1.0)),
        dtype=np.float64,
    )
    points = np.ascontiguousarray(np.concatenate((tie_points, tie_points)))
    hints = np.asarray((lower_leaf,) * 3 + upper_leaves, dtype=np.int64)
    output, stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, points, hints
    )
    assert output[:3].tolist() == list(upper_leaves)
    assert output[3:].tolist() == list(upper_leaves)
    assert stats == HintedLocationStats(6, 6, 0, 6, 3, 3)


def test_exterior_points_on_every_axis_count_candidates_but_never_fallback() -> None:
    loc_arguments, hinted_arguments, forest = mixed_fixture()
    bounds = leaf_bounds(hinted_arguments, forest)
    center = bounds[0, 0] + 0.4 * (bounds[0, 1] - bounds[0, 0])
    points = []
    for axis in range(3):
        below = center.copy()
        below[axis] = np.nextafter(hinted_arguments[0][axis], -np.inf)
        points.append(below)
        at_upper = center.copy()
        at_upper[axis] = hinted_arguments[1][axis]
        points.append(at_upper)
    point_array = np.ascontiguousarray(points, dtype=np.float64)
    hints = np.zeros(point_array.shape[0], dtype=np.int64)
    output, stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, point_array, hints
    )
    assert output.tolist() == [-1] * 6
    assert stats == HintedLocationStats(6, 0, 6, 6, 0, 0)


def test_random_hints_are_exactly_loc_equivalent() -> None:
    loc_arguments, hinted_arguments, forest = mixed_fixture()
    rng = np.random.default_rng(77123)
    lower = np.asarray(hinted_arguments[0])
    upper = np.asarray(hinted_arguments[1])
    points = np.ascontiguousarray(
        rng.uniform(lower - 0.2, upper + 0.2, size=(600, 3)),
        dtype=np.float64,
    )
    hints = rng.integers(
        -1, forest.leaf_node_ids.size, size=points.shape[0], dtype=np.int64
    )
    assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, points, hints
    )


def test_unchecked_symbol_matches_checked_after_preflight() -> None:
    loc_arguments, hinted_arguments, forest = mixed_fixture()
    bounds = leaf_bounds(hinted_arguments, forest)
    points = np.ascontiguousarray(
        [
            bounds[0, 0] + 0.3 * (bounds[0, 1] - bounds[0, 0]),
            bounds[-1, 0] + 0.7 * (bounds[-1, 1] - bounds[-1, 0]),
        ],
        dtype=np.float64,
    )
    hints = np.asarray((0, 0), dtype=np.int64)
    expected, expected_stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, points, hints
    )
    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(hinted_arguments[1][axis]) - float(hinted_arguments[0][axis])
        base_spacing[axis] = extent / float(hinted_arguments[3][axis])
    actual = np.empty_like(expected)
    hits = fill_refined_point_leaf_ids_with_hints_unchecked(
        hinted_arguments[0],
        hinted_arguments[1],
        hinted_arguments[3],
        hinted_arguments[4],
        hinted_arguments[6],
        hinted_arguments[7],
        hinted_arguments[8],
        hinted_arguments[9],
        hinted_arguments[10],
        hinted_arguments[11],
        hinted_arguments[12],
        base_spacing,
        points,
        hints,
        actual,
    )
    assert np.array_equal(actual, expected)
    assert hits == expected_stats.hint_hit_count


def test_level_63_hint_is_supported() -> None:
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
    domain_cells = block_cells.copy()
    loc_arguments = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
    )
    hinted_arguments = (
        *loc_arguments[:8],
        forest.node_levels,
        forest.node_coords,
        *loc_arguments[8:],
        forest.leaf_node_ids,
    )
    point = np.ascontiguousarray([[0.5, np.ldexp(1.0, -63), np.ldexp(1.0, -63)]])
    owner = refined_point_leaf_ids(*loc_arguments, point)
    output, stats = assert_matches_loc_and_reference(
        loc_arguments, hinted_arguments, point, owner.copy()
    )
    assert output[0] == owner[0]
    assert stats.hint_hit_count == 1


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("nonfinite", ValueError, "point 3, axis 2"),
        ("hint_low", ValueError, "first invalid entry is 3"),
        ("hint_high", ValueError, "first invalid entry is 3"),
        ("hint_dtype", TypeError, "int64"),
        ("hint_shape", ValueError, "one entry per point"),
        ("hint_layout", ValueError, "C-contiguous"),
        ("node_level_shape", ValueError, "node_levels must have shape"),
        ("hint_node", ValueError, "invalid hinted refined geometry"),
        ("hint_reciprocal", ValueError, "invalid hinted refined geometry"),
        ("hint_coordinate", ValueError, "invalid hinted refined geometry"),
    ],
)
def test_validation_errors_are_atomic(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    _, base_arguments, forest = mixed_fixture()
    arguments = list(base_arguments)
    bounds = leaf_bounds(base_arguments, forest)
    selected = (0, 1, 2, 3)
    points = np.ascontiguousarray(
        [bounds[leaf, 0] + 0.4 * (bounds[leaf, 1] - bounds[leaf, 0]) for leaf in selected],
        dtype=np.float64,
    )
    hints = np.asarray(selected, dtype=np.int64)
    output = np.full(points.shape[0], -991, dtype=np.int64)
    before = output.copy()

    if failure == "nonfinite":
        points = points.copy()
        points[3, 2] = np.nan
    elif failure == "hint_low":
        hints = hints.copy()
        hints[3] = -2
    elif failure == "hint_high":
        hints = hints.copy()
        hints[3] = forest.leaf_node_ids.size
    elif failure == "hint_dtype":
        hints = hints.astype(np.int32)
    elif failure == "hint_shape":
        hints = hints[:-1]
    elif failure == "hint_layout":
        storage = np.empty(hints.size * 2, dtype=np.int64)
        storage[::2] = hints
        hints = storage[::2]
    elif failure == "node_level_shape":
        arguments[8] = arguments[8][:-1]
    elif failure == "hint_node":
        bad_leaf_nodes = arguments[12].copy()
        bad_leaf_nodes[int(hints[3])] = arguments[8].shape[0]
        arguments[12] = bad_leaf_nodes
    elif failure == "hint_reciprocal":
        bad_node_leaf = arguments[11].copy()
        hinted_node = int(arguments[12][int(hints[3])])
        bad_node_leaf[hinted_node] = -1
        arguments[11] = bad_node_leaf
    else:
        bad_coords = arguments[9].copy()
        hinted_node = int(arguments[12][int(hints[3])])
        level = int(arguments[8][hinted_node])
        bad_coords[hinted_node, 0] = int(arguments[2][0]) * (1 << (level - 1))
        arguments[9] = bad_coords

    with pytest.raises(error, match=match):
        fill_refined_point_leaf_ids_with_hints(
            *arguments, points, hints, output
        )
    assert np.array_equal(output, before)


def test_overflow_and_subnormal_spacing_are_atomic() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, forest = make_forest(
        root_shape, lambda level, _coord: level == 1
    )
    base = [
        f3(0.0, 0.0, 0.0),
        f3(1.0, 1.0, 1.0),
        root_shape,
        i3(1, 1, 1),
        i3(1, 1, 1),
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    ]
    points = np.ascontiguousarray([[0.25, 0.25, 0.25]], dtype=np.float64)
    hints = i3(0)
    output = np.full(1, -1201, dtype=np.int64)
    before = output.copy()

    overflow = base.copy()
    overflow[2] = i3(np.iinfo(np.int64).max, 1, 1)
    overflow[3] = i3(np.iinfo(np.int64).max, 1, 1)
    overflow[4] = i3(2, 1, 1)
    with pytest.raises(OverflowError, match="root block-cell"):
        fill_refined_point_leaf_ids_with_hints(
            *overflow, points, hints, output
        )
    assert np.array_equal(output, before)

    subnormal = base.copy()
    subnormal[1] = f3(np.finfo(np.float64).tiny, 1.0, 1.0)
    subnormal[5] = 2
    with pytest.raises(ValueError, match="normal"):
        fill_refined_point_leaf_ids_with_hints(
            *subnormal, points, hints, output
        )
    assert np.array_equal(output, before)


def test_hinted_leaf_face_collapse_is_atomic() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        chain_flags(4),
    )
    deepest = np.flatnonzero(
        forest.node_levels[forest.leaf_node_ids] == 5
    ).astype(np.int64)[0]
    arguments = (
        f3(1.0e16, 0.0, 0.0),
        f3(1.0e16 + 16.0, 1.0, 1.0),
        root_shape,
        i3(1, 1, 1),
        i3(1, 1, 1),
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    points = np.ascontiguousarray([[1.0e16, 0.01, 0.01]], dtype=np.float64)
    hints = np.asarray([deepest], dtype=np.int64)
    output = np.full(1, -1251, dtype=np.int64)
    before = output.copy()
    with pytest.raises(ValueError, match="spacing and faces"):
        fill_refined_point_leaf_ids_with_hints(
            *arguments, points, hints, output
        )
    assert np.array_equal(output, before)


def test_exterior_point_does_not_skip_corrupt_hint_preflight() -> None:
    _, base_arguments, forest = mixed_fixture()
    arguments = list(base_arguments)
    point = np.ascontiguousarray(
        [[np.nextafter(arguments[0][0], -np.inf), arguments[0][1], arguments[0][2]]],
        dtype=np.float64,
    )
    hint = i3(0)
    bad_leaf_nodes = arguments[12].copy()
    bad_leaf_nodes[0] = arguments[8].shape[0]
    arguments[12] = bad_leaf_nodes
    output = np.full(1, -1271, dtype=np.int64)
    before = output.copy()
    with pytest.raises(ValueError, match="invalid hinted refined geometry"):
        fill_refined_point_leaf_ids_with_hints(
            *arguments, point, hint, output
        )
    assert np.array_equal(output, before)

def test_output_alias_and_readonly_inputs() -> None:
    _, hinted_arguments, forest = mixed_fixture()
    bounds = leaf_bounds(hinted_arguments, forest)
    points = np.ascontiguousarray(
        [bounds[0, 0] + 0.4 * (bounds[0, 1] - bounds[0, 0])],
        dtype=np.float64,
    )
    hints = i3(0)
    aliased = hints
    before = aliased.copy()
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_point_leaf_ids_with_hints(
            *hinted_arguments, points, hints, aliased
        )
    assert np.array_equal(aliased, before)

    readonly_output = np.full(1, -1300, dtype=np.int64)
    readonly_output.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_refined_point_leaf_ids_with_hints(
            *hinted_arguments, points, hints, readonly_output
        )

    output = np.full(1, -1301, dtype=np.int64)
    readonly = [
        value
        for value in (*hinted_arguments[:5], *hinted_arguments[6:], points, hints)
        if isinstance(value, np.ndarray)
    ]
    copies = [value.copy() for value in readonly]
    for value in readonly:
        value.setflags(write=False)
    stats = fill_refined_point_leaf_ids_with_hints(
        *hinted_arguments, points, hints, output
    )
    assert stats.hint_hit_count == 1
    for value, expected in zip(readonly, copies, strict=True):
        assert np.array_equal(value, expected)
