from __future__ import annotations

import math

import numpy as np
import pytest

from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.region_selection import (
    RefinedRegionSelection,
    count_refined_region_windows,
    fill_refined_region_windows,
    refined_region_windows,
)
from simesh_rewrite.region_selection_reference import (
    refined_region_windows_reference,
)


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


def mixed_case() -> tuple[np.ndarray, ...]:
    root_shape = i3(3, 2, 2)
    block_cells = i3(4, 3, 5)
    forest = make_forest(
        root_shape,
        lambda level, coord: (
            level == 1 and coord in {(0, 0, 0), (2, 1, 1)}
        )
        or (level == 2 and coord in {(0, 0, 0), (5, 3, 3)}),
    )
    domain_lower = f3(-1.25, 0.75, -3.5)
    domain_upper = f3(4.75, 6.75, 8.5)
    domain_cells = np.ascontiguousarray(root_shape * block_cells, dtype=np.int64)
    return (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
    )


def canonical_center(arguments: tuple[np.ndarray, ...], leaf: int, axis: int, local: int) -> float:
    domain_lower = arguments[0]
    domain_upper = arguments[1]
    domain_cells = arguments[3]
    block_cells = arguments[4]
    node_levels = arguments[5]
    node_coords = arguments[6]
    leaf_node_ids = arguments[7]
    node = int(leaf_node_ids[leaf])
    shift = int(node_levels[node]) - 1
    extent = float(domain_upper[axis]) - float(domain_lower[axis])
    base_h = extent / float(domain_cells[axis])
    h = math.ldexp(base_h, -shift)
    global_cell = int(node_coords[node, axis]) * int(block_cells[axis]) + local
    factor = float(global_cell) + 0.5
    offset = factor * h
    return float(domain_lower[axis]) + offset


def canonical_leaf_face(
    arguments: tuple[np.ndarray, ...], leaf: int, axis: int, side: int
) -> float:
    domain_lower = arguments[0]
    domain_upper = arguments[1]
    domain_cells = arguments[3]
    block_cells = arguments[4]
    node_levels = arguments[5]
    node_coords = arguments[6]
    leaf_node_ids = arguments[7]
    node = int(leaf_node_ids[leaf])
    shift = int(node_levels[node]) - 1
    scale = 1 << shift
    level_cells = int(domain_cells[axis]) * scale
    extent = float(domain_upper[axis]) - float(domain_lower[axis])
    base_h = extent / float(domain_cells[axis])
    spacing = math.ldexp(base_h, -shift)
    face_index = (
        int(node_coords[node, axis]) * int(block_cells[axis])
        + side * int(block_cells[axis])
    )
    if face_index == 0:
        return float(domain_lower[axis])
    if face_index == level_cells:
        return float(domain_upper[axis])
    offset = float(face_index) * spacing
    return float(domain_lower[axis]) + offset


def assert_selection_equal(
    actual: RefinedRegionSelection,
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    assert np.array_equal(actual.leaf_ids, expected[0])
    assert np.array_equal(actual.cell_lower, expected[1])
    assert np.array_equal(actual.cell_upper, expected[2])


def test_full_domain_count_fill_and_exact_output_bytes() -> None:
    arguments = mixed_case()
    region_lower = arguments[0].copy()
    region_upper = arguments[1].copy()

    selected_count = count_refined_region_windows(
        *arguments, region_lower, region_upper
    )
    assert selected_count == arguments[7].shape[0]
    result = refined_region_windows(*arguments, region_lower, region_upper)
    assert isinstance(result, RefinedRegionSelection)
    assert np.array_equal(
        result.leaf_ids, np.arange(selected_count, dtype=np.int64)
    )
    assert np.array_equal(result.cell_lower, np.zeros((selected_count, 3), dtype=np.int64))
    assert np.array_equal(
        result.cell_upper,
        np.broadcast_to(arguments[4], (selected_count, 3)),
    )
    assert sum(value.nbytes for value in result) == 56 * selected_count

    leaf_ids = np.full(selected_count, -7, dtype=np.int64)
    cell_lower = np.full((selected_count, 3), -8, dtype=np.int64)
    cell_upper = np.full((selected_count, 3), -9, dtype=np.int64)
    returned = fill_refined_region_windows(
        *arguments,
        region_lower,
        region_upper,
        leaf_ids,
        cell_lower,
        cell_upper,
    )
    assert returned is None
    assert np.array_equal(leaf_ids, result.leaf_ids)
    assert np.array_equal(cell_lower, result.cell_lower)
    assert np.array_equal(cell_upper, result.cell_upper)


def test_non_aligned_mixed_depth_region_matches_exhaustive_reference() -> None:
    arguments = mixed_case()
    region_lower = f3(-0.63, 1.83, -0.41)
    region_upper = f3(3.37, 5.92, 7.71)
    actual = refined_region_windows(*arguments, region_lower, region_upper)
    expected = refined_region_windows_reference(
        *arguments, region_lower, region_upper
    )
    assert_selection_equal(actual, expected)
    assert actual.leaf_ids.size > 1
    assert np.all(actual.leaf_ids[1:] > actual.leaf_ids[:-1])
    assert np.all(actual.cell_lower >= 0)
    assert np.all(actual.cell_lower < actual.cell_upper)
    assert np.all(actual.cell_upper <= arguments[4])


def test_exact_center_bounds_are_lower_inclusive_upper_exclusive() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(6, 3, 2)
    forest = make_forest(root_shape, lambda _level, _coord: False)
    arguments = (
        f3(-2.0, 0.0, 4.0),
        f3(4.0, 3.0, 8.0),
        root_shape,
        block_cells.copy(),
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
    )
    center_2 = canonical_center(arguments, 0, 0, 2)
    center_5 = canonical_center(arguments, 0, 0, 5)
    result = refined_region_windows(
        *arguments,
        f3(center_2, 0.0, 4.0),
        f3(center_5, 3.0, 8.0),
    )
    assert np.array_equal(result.leaf_ids, i3(0))
    assert np.array_equal(result.cell_lower, np.asarray([[2, 0, 0]], dtype=np.int64))
    assert np.array_equal(result.cell_upper, np.asarray([[5, 3, 2]], dtype=np.int64))

    above_lower = np.nextafter(center_2, np.inf)
    above = refined_region_windows(
        *arguments,
        f3(above_lower, 0.0, 4.0),
        f3(center_5, 3.0, 8.0),
    )
    assert above.cell_lower[0, 0] == 3
    below_upper = np.nextafter(center_5, -np.inf)
    below = refined_region_windows(
        *arguments,
        f3(center_2, 0.0, 4.0),
        f3(below_upper, 3.0, 8.0),
    )
    assert below.cell_upper[0, 0] == 5
    above_upper = np.nextafter(center_5, np.inf)
    above = refined_region_windows(
        *arguments,
        f3(center_2, 0.0, 4.0),
        f3(above_upper, 3.0, 8.0),
    )
    assert above.cell_upper[0, 0] == 6

    one_center = [canonical_center(arguments, 0, axis, 1) for axis in range(3)]
    one = refined_region_windows(
        *arguments,
        np.asarray(one_center, dtype=np.float64),
        np.nextafter(np.asarray(one_center, dtype=np.float64), np.inf),
    )
    assert np.array_equal(one.leaf_ids, i3(0))
    assert np.array_equal(one.cell_lower, np.asarray([[1, 1, 1]], dtype=np.int64))
    assert np.array_equal(one.cell_upper, np.asarray([[2, 2, 2]], dtype=np.int64))


def test_exact_refined_leaf_faces_select_that_leaf_window() -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(4, 4, 4)
    forest = make_forest(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    arguments = (
        f3(-1.0, 2.0, -3.0),
        f3(3.0, 6.0, 5.0),
        root_shape,
        np.ascontiguousarray(root_shape * block_cells, dtype=np.int64),
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
    )
    leaf_nodes = forest.leaf_node_ids
    target = next(
        leaf
        for leaf, node_value in enumerate(leaf_nodes)
        if int(forest.node_levels[int(node_value)]) == 2
        and tuple(int(value) for value in forest.node_coords[int(node_value)])
        == (1, 1, 1)
    )
    lower = np.asarray(
        [canonical_leaf_face(arguments, target, axis, 0) for axis in range(3)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [canonical_leaf_face(arguments, target, axis, 1) for axis in range(3)],
        dtype=np.float64,
    )
    result = refined_region_windows(*arguments, lower, upper)
    expected = refined_region_windows_reference(*arguments, lower, upper)
    assert_selection_equal(result, expected)
    assert np.array_equal(result.leaf_ids, np.asarray([target], dtype=np.int64))
    assert np.array_equal(result.cell_lower, np.zeros((1, 3), dtype=np.int64))
    assert np.array_equal(result.cell_upper, block_cells.reshape(1, 3))

    for axis in range(3):
        nudged_lower = lower.copy()
        nudged_lower[axis] = np.nextafter(nudged_lower[axis], np.inf)
        nudged = refined_region_windows(*arguments, nudged_lower, upper)
        reference = refined_region_windows_reference(
            *arguments, nudged_lower, upper
        )
        assert_selection_equal(nudged, reference)


def test_empty_and_thin_regions_emit_no_block_cover() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(4, 2, 2)
    forest = make_forest(root_shape, lambda _level, _coord: False)
    arguments = (
        f3(0.0, 0.0, 0.0),
        f3(4.0, 2.0, 2.0),
        root_shape,
        block_cells.copy(),
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
    )
    empty = refined_region_windows(
        *arguments, f3(1.0, 0.0, 0.0), f3(1.0, 2.0, 2.0)
    )
    assert empty.leaf_ids.shape == (0,)
    assert empty.cell_lower.shape == (0, 3)
    assert empty.cell_upper.shape == (0, 3)
    assert sum(value.nbytes for value in empty) == 0

    center_0 = canonical_center(arguments, 0, 0, 0)
    center_1 = canonical_center(arguments, 0, 0, 1)
    thin = refined_region_windows(
        *arguments,
        f3(np.nextafter(center_0, np.inf), 0.0, 0.0),
        f3(center_1, 2.0, 2.0),
    )
    assert thin.leaf_ids.size == 0


def test_exact_leaf_faces_and_random_regions_match_exhaustive_reference() -> None:
    arguments = mixed_case()
    rng = np.random.default_rng(20260904)
    candidate_axes: list[list[float]] = []
    for axis in range(3):
        values = [float(arguments[0][axis]), float(arguments[1][axis])]
        for leaf in range(arguments[7].shape[0]):
            for local in range(int(arguments[4][axis])):
                value = canonical_center(arguments, leaf, axis, local)
                values.extend(
                    (
                        value,
                        float(np.nextafter(value, -np.inf)),
                        float(np.nextafter(value, np.inf)),
                    )
                )
        candidate_axes.append(
            sorted(
                value
                for value in set(values)
                if float(arguments[0][axis]) <= value <= float(arguments[1][axis])
            )
        )

    for _ in range(80):
        lower = np.empty(3, dtype=np.float64)
        upper = np.empty(3, dtype=np.float64)
        for axis in range(3):
            left = int(rng.integers(0, len(candidate_axes[axis])))
            right = int(rng.integers(left, len(candidate_axes[axis])))
            lower[axis] = candidate_axes[axis][left]
            upper[axis] = candidate_axes[axis][right]
        actual = refined_region_windows(*arguments, lower, upper)
        expected = refined_region_windows_reference(*arguments, lower, upper)
        assert_selection_equal(actual, expected)


@pytest.mark.parametrize(
    ("bad_lower", "bad_upper", "message"),
    [
        (f3(np.nan, 0.0, 0.0), f3(1.0, 1.0, 1.0), "finite"),
        (f3(0.0, 0.0, 0.0), f3(np.inf, 1.0, 1.0), "finite"),
        (f3(0.8, 0.0, 0.0), f3(0.7, 1.0, 1.0), "must not exceed"),
        (f3(-0.1, 0.0, 0.0), f3(1.0, 1.0, 1.0), "contained"),
        (f3(0.0, 0.0, 0.0), f3(1.1, 1.0, 1.0), "contained"),
    ],
)
def test_invalid_region_bounds_are_rejected(
    bad_lower: np.ndarray,
    bad_upper: np.ndarray,
    message: str,
) -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(2, 2, 2)
    forest = make_forest(root_shape, lambda _level, _coord: False)
    with pytest.raises(ValueError, match=message):
        count_refined_region_windows(
            f3(0.0, 0.0, 0.0),
            f3(1.0, 1.0, 1.0),
            root_shape,
            block_cells.copy(),
            block_cells,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            bad_lower,
            bad_upper,
        )


def test_count_and_lifecycle_overflow_failures() -> None:
    valid_float = f3(0.0, 0.0, 0.0)
    valid_upper = f3(1.0, 1.0, 1.0)
    node_levels = np.ones(1, dtype=np.int64)
    node_coords = np.zeros((1, 3), dtype=np.int64)
    leaf_nodes = np.zeros(1, dtype=np.int64)

    with pytest.raises(OverflowError, match="root block-cell"):
        count_refined_region_windows(
            valid_float,
            valid_upper,
            i3(np.iinfo(np.int64).max, 1, 1),
            i3(np.iinfo(np.int64).max, 1, 1),
            i3(2, 1, 1),
            node_levels,
            node_coords,
            leaf_nodes,
            valid_float,
            valid_upper,
        )

    with pytest.raises(OverflowError, match="overflows int64"):
        count_refined_region_windows(
            valid_float,
            valid_upper,
            i3(1, 1, 1),
            i3(2, 1, 1),
            i3(2, 1, 1),
            np.asarray([63], dtype=np.int64),
            node_coords,
            leaf_nodes,
            valid_float,
            valid_upper,
        )

    with pytest.raises(OverflowError, match="overflows int64"):
        count_refined_region_windows(
            valid_float,
            valid_upper,
            i3(1, 1, 1),
            i3(1, 1, 1),
            i3(1, 1, 1),
            np.asarray([64], dtype=np.int64),
            node_coords,
            leaf_nodes,
            valid_float,
            valid_upper,
        )


def test_large_origin_center_collapse_is_rejected() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(1, 1, 1)
    forest = make_forest(
        root_shape,
        lambda level, coord: level < 5 and coord == (0, 0, 0),
    )
    domain_lower = f3(1.0e16, 0.0, 0.0)
    domain_upper = f3(1.0e16 + 16.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="cell centers"):
        count_refined_region_windows(
            domain_lower,
            domain_upper,
            root_shape,
            block_cells.copy(),
            block_cells,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            domain_lower,
            domain_upper,
        )


def test_fill_late_lifecycle_error_preserves_every_output() -> None:
    arguments = list(mixed_case())
    region_lower = arguments[0].copy()
    region_upper = arguments[1].copy()
    selected_count = arguments[7].shape[0]
    leaf_ids = np.full(selected_count, -31, dtype=np.int64)
    cell_lower = np.full((selected_count, 3), -32, dtype=np.int64)
    cell_upper = np.full((selected_count, 3), -33, dtype=np.int64)
    before = tuple(value.copy() for value in (leaf_ids, cell_lower, cell_upper))

    bad_leaf_nodes = arguments[7].copy()
    bad_leaf_nodes[-1] = arguments[5].shape[0]
    arguments[7] = bad_leaf_nodes
    with pytest.raises(ValueError, match="invalid refined forest geometry"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            leaf_ids,
            cell_lower,
            cell_upper,
        )
    for actual, expected in zip((leaf_ids, cell_lower, cell_upper), before, strict=True):
        assert np.array_equal(actual, expected)


def test_output_contract_errors_are_atomic() -> None:
    arguments = mixed_case()
    region_lower = arguments[0].copy()
    region_upper = arguments[1].copy()
    selected_count = arguments[7].shape[0]
    leaf_ids = np.full(selected_count, -41, dtype=np.int64)
    cell_lower = np.full((selected_count, 3), -42, dtype=np.int64)
    cell_upper = np.full((selected_count, 3), -43, dtype=np.int64)
    before = tuple(value.copy() for value in (leaf_ids, cell_lower, cell_upper))

    with pytest.raises(TypeError, match="int64"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            leaf_ids.astype(np.int32),
            cell_lower,
            cell_upper,
        )
    with pytest.raises(ValueError, match="shape"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            leaf_ids[:-1],
            cell_lower,
            cell_upper,
        )
    readonly = cell_upper.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            leaf_ids,
            cell_lower,
            readonly,
        )
    noncontiguous = np.full((selected_count, 6), -44, dtype=np.int64)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            leaf_ids,
            noncontiguous,
            cell_upper,
        )
    shared = np.full(selected_count * 6, -45, dtype=np.int64)
    overlapping_lower = shared[: selected_count * 3].reshape(selected_count, 3)
    overlapping_upper = shared[1 : selected_count * 3 + 1].reshape(
        selected_count, 3
    )
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            leaf_ids,
            overlapping_lower,
            overlapping_upper,
        )
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_region_windows(
            *arguments,
            region_lower,
            region_upper,
            arguments[7],
            cell_lower,
            cell_upper,
        )

    for actual, expected in zip((leaf_ids, cell_lower, cell_upper), before, strict=True):
        assert np.array_equal(actual, expected)


def test_input_type_layout_and_count_compatibility_errors() -> None:
    arguments = list(mixed_case())
    region_lower = arguments[0].copy()
    region_upper = arguments[1].copy()

    bad = arguments.copy()
    bad[2] = bad[2].astype(np.int32)
    with pytest.raises(TypeError, match="int64"):
        count_refined_region_windows(*bad, region_lower, region_upper)

    bad = arguments.copy()
    bad[3] = bad[3].copy()
    bad[3][0] += 1
    with pytest.raises(ValueError, match="root_shape"):
        count_refined_region_windows(*bad, region_lower, region_upper)

    bad = arguments.copy()
    bad[6] = np.asfortranarray(bad[6])
    assert not bad[6].flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        count_refined_region_windows(*bad, region_lower, region_upper)

    with pytest.raises(TypeError, match="float64"):
        count_refined_region_windows(
            *arguments,
            region_lower.astype(np.float32),
            region_upper,
        )
