from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.refined_sampling import (
    sample_refined_zero_order_point_groups,
)
from simesh_rewrite.refined_sampling_reference import (
    sample_refined_zero_order_point_groups_reference,
)
from simesh_rewrite.sampling import sample_level1_zero_order


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
        if split:
            for child in range(8):
                bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
                visit(
                    level + 1,
                    tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
                )

    for root_coord in root_coords:
        visit(1, tuple(int(value) for value in root_coord))
    return np.asarray(flags, dtype=np.bool_)


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


def locator_args(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    block_cells: np.ndarray,
    coord_to_rank: np.ndarray,
    forest: RefinedForest,
) -> tuple[object, ...]:
    return (
        domain_lower,
        domain_upper,
        root_shape,
        np.ascontiguousarray(root_shape * block_cells, dtype=np.int64),
        block_cells,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
    )


def groups_from_owners(
    owners: np.ndarray,
    slot_leaf_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: list[int] = []
    offsets = [0]
    for leaf_id in slot_leaf_ids:
        grouped.extend(np.flatnonzero(owners == leaf_id).tolist())
        offsets.append(len(grouped))
    return np.asarray(grouped, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def sampling_args(
    payload: np.ndarray,
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    slot_leaf_ids: np.ndarray,
    locator: tuple[object, ...],
    forest: RefinedForest,
    points: np.ndarray,
    point_indices: np.ndarray,
    offsets: np.ndarray,
    output: np.ndarray,
) -> tuple[object, ...]:
    return (
        payload,
        valid_lower,
        valid_upper,
        interior_lower,
        interior_upper,
        slot_leaf_ids,
        locator[0],
        locator[1],
        locator[3],
        locator[4],
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        points,
        point_indices,
        offsets,
        output,
    )


def patterned_payload(
    slot_leaf_ids: np.ndarray,
    field_count: int,
    block_cells: np.ndarray,
) -> np.ndarray:
    shape = tuple(int(value) for value in block_cells)
    x, y, z = np.indices(shape, dtype=np.float64)
    payload = np.empty((slot_leaf_ids.size, field_count, *shape))
    for slot, leaf_id in enumerate(slot_leaf_ids):
        for field in range(field_count):
            payload[slot, field] = (
                float(leaf_id) * 100000.0
                + field * 10000.0
                + x * 100.0
                + y * 10.0
                + z
            )
    return payload


def test_all_refined_leaf_and_local_cell_faces_match_scalar_reference() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(4, 4, 4)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(1.0, 1.0, 1.0)
    coord_to_rank, forest = make_forest(
        root_shape, lambda level, _coord: level == 1
    )
    locator = locator_args(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        domain_lower,
        domain_upper,
        root_shape,
        locator[3],
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )
    points: list[np.ndarray] = []
    for leaf_id in leaf_ids:
        center = bounds[leaf_id, 0] + 0.5 * (
            bounds[leaf_id, 1] - bounds[leaf_id, 0]
        )
        for axis in range(3):
            for cell in range(int(block_cells[axis])):
                point = center.copy()
                point[axis] = np.float64(
                    bounds[leaf_id, 0, axis]
                    + np.float64(cell) * spacing[leaf_id, axis]
                )
                points.append(point)
    points.append(f3(0.125, 0.125, 0.125))  # unselected result row
    point_array = np.ascontiguousarray(points, dtype=np.float64)
    owners = refined_point_leaf_ids(*locator, point_array)
    slot_leaf_ids = leaf_ids[::-1].copy()
    point_indices, offsets = groups_from_owners(owners[:-1], slot_leaf_ids)
    payload = patterned_payload(slot_leaf_ids, 3, block_cells)
    output = np.full((point_array.shape[0], 3), -97.0)
    output[-1] = np.asarray([np.nan, -0.0, np.inf])
    expected = output.copy()
    arguments = sampling_args(
        payload,
        i3(0, 0, 0),
        block_cells,
        i3(0, 0, 0),
        block_cells,
        slot_leaf_ids,
        locator,
        forest,
        point_array,
        point_indices,
        offsets,
        output,
    )
    sample_refined_zero_order_point_groups_reference(
        payload,
        i3(0, 0, 0),
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        locator[3],
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        point_array,
        point_indices,
        offsets,
        expected,
    )
    sample_refined_zero_order_point_groups(*arguments)
    assert np.array_equal(output.view(np.uint64), expected.view(np.uint64))


def test_arbitrary_repeated_slots_and_indices_use_later_write() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(2, 2, 2)
    coord_to_rank, forest = make_forest(root_shape, lambda _level, _coord: False)
    locator = locator_args(
        f3(0.0, 0.0, 0.0),
        f3(2.0, 2.0, 2.0),
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    payload = np.empty((2, 1, 2, 2, 2), dtype=np.float64)
    payload[0] = 1.0
    payload[1] = 2.0
    output = np.asarray([[-1.0]], dtype=np.float64)
    sample_refined_zero_order_point_groups(
        *sampling_args(
            payload,
            i3(0, 0, 0),
            block_cells,
            i3(0, 0, 0),
            block_cells,
            i3(0, 0),
            locator,
            forest,
            np.asarray([[0.25, 0.25, 0.25]], dtype=np.float64),
            i3(0, 0),
            i3(0, 1, 2),
            output,
        )
    )
    assert output[0, 0] == 2.0


def test_copied_values_preserve_signed_zero_and_nan_payload_bits() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(2, 2, 2)
    coord_to_rank, forest = make_forest(root_shape, lambda _level, _coord: False)
    locator = locator_args(
        f3(0.0, 0.0, 0.0),
        f3(2.0, 2.0, 2.0),
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    payload = np.zeros((1, 2, 2, 2, 2), dtype=np.float64)
    expected_bits = np.asarray(
        [0x8000000000000000, 0x7FF8000000001234], dtype=np.uint64
    )
    payload[0, :, 0, 0, 0] = expected_bits.view(np.float64)
    output = np.zeros((1, 2), dtype=np.float64)
    sample_refined_zero_order_point_groups(
        *sampling_args(
            payload,
            i3(0, 0, 0),
            block_cells,
            i3(0, 0, 0),
            block_cells,
            i3(0),
            locator,
            forest,
            np.asarray([[0.25, 0.25, 0.25]], dtype=np.float64),
            i3(0),
            i3(0, 1),
            output,
        )
    )
    assert np.array_equal(output.view(np.uint64)[0], expected_bits)


@pytest.mark.parametrize(
    ("sample_lower_values", "sample_upper_values", "output_shape"),
    [
        ((-1.0, 2.0, -3.0), (5.0, 6.0, 1.0), (6, 4, 2)),
        ((-1.0, 2.0, -3.0), (5.0, 6.0, 1.0), (12, 8, 4)),
        ((-1.0, 2.0, -3.0), (5.0, 6.0, 1.0), (3, 2, 1)),
        ((-0.75, 2.25, -2.75), (4.75, 5.75, 0.75), (5, 3, 4)),
    ],
    ids=("native-full-domain", "upsampled", "downsampled", "subdomain"),
)
def test_level_one_output_centers_reduce_bitwise_to_fused_sampler(
    sample_lower_values: tuple[float, float, float],
    sample_upper_values: tuple[float, float, float],
    output_shape: tuple[int, int, int],
) -> None:
    root_shape = i3(2, 2, 1)
    block_cells = i3(3, 2, 2)
    domain_lower = f3(-1.0, 2.0, -3.0)
    domain_upper = f3(5.0, 6.0, 1.0)
    coord_to_rank, forest = make_forest(root_shape, lambda _level, _coord: False)
    locator = locator_args(
        domain_lower,
        domain_upper,
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    slot_leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    payload = patterned_payload(slot_leaf_ids, 2, block_cells)
    sample_lower = f3(*sample_lower_values)
    sample_upper = f3(*sample_upper_values)
    output_spacing = (sample_upper - sample_lower) / np.asarray(output_shape)
    points = np.empty((int(np.prod(output_shape)), 3), dtype=np.float64)
    for row, index in enumerate(np.ndindex(output_shape)):
        for axis in range(3):
            factor = np.float64(index[axis]) + np.float64(0.5)
            offset = factor * np.float64(output_spacing[axis])
            points[row, axis] = np.float64(sample_lower[axis]) + offset
    owners = refined_point_leaf_ids(*locator, points)
    point_indices, offsets = groups_from_owners(owners, slot_leaf_ids)
    point_values = np.full((points.shape[0], 2), -1.0)
    sample_refined_zero_order_point_groups(
        *sampling_args(
            payload,
            i3(0, 0, 0),
            block_cells,
            i3(0, 0, 0),
            block_cells,
            slot_leaf_ids,
            locator,
            forest,
            points,
            point_indices,
            offsets,
            point_values,
        )
    )
    fused = np.full((2, *output_shape), -1.0)
    _, rank_to_coord = level1_morton(root_shape)
    sample_level1_zero_order(
        payload,
        i3(0, 0, 0),
        block_cells,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        locator[3],
        block_cells,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        fused,
    )
    grouped = point_values.reshape((*output_shape, 2)).transpose(3, 0, 1, 2)
    assert np.array_equal(grouped.view(np.uint64), fused.view(np.uint64))


def test_output_alias_with_payload_is_rejected() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(2, 2, 2)
    coord_to_rank, forest = make_forest(root_shape, lambda _level, _coord: False)
    locator = locator_args(
        f3(0.0, 0.0, 0.0),
        f3(2.0, 2.0, 2.0),
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    payload = np.ones((1, 1, 2, 2, 2), dtype=np.float64)
    aliased_output = payload.reshape(-1)[:1].reshape(1, 1)
    with pytest.raises(ValueError, match="must not overlap"):
        sample_refined_zero_order_point_groups(
            *sampling_args(
                payload,
                i3(0, 0, 0),
                block_cells,
                i3(0, 0, 0),
                block_cells,
                i3(0),
                locator,
                forest,
                np.asarray([[0.25, 0.25, 0.25]], dtype=np.float64),
                i3(0),
                i3(0, 1),
                aliased_output,
            )
        )


@pytest.mark.parametrize("failure", ["owner", "nonfinite", "index", "offset"])
def test_validation_failures_are_atomic(failure: str) -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(2, 2, 2)
    coord_to_rank, forest = make_forest(root_shape, lambda _level, _coord: False)
    locator = locator_args(
        f3(0.0, 0.0, 0.0),
        f3(4.0, 2.0, 2.0),
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    points = np.asarray([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], dtype=np.float64)
    indices = i3(0, 1)
    offsets = i3(0, 2)
    if failure == "owner":
        points[1, 0] = 2.5
    elif failure == "nonfinite":
        points[1, 0] = np.nan
    elif failure == "index":
        indices[1] = 2
    else:
        offsets[0] = 1
    payload = patterned_payload(i3(0), 2, block_cells)
    output = np.asarray([[1.0, -0.0], [np.nan, np.inf]], dtype=np.float64)
    before = output.view(np.uint64).copy()
    with pytest.raises(ValueError):
        sample_refined_zero_order_point_groups(
            *sampling_args(
                payload,
                i3(0, 0, 0),
                block_cells,
                i3(0, 0, 0),
                block_cells,
                i3(0),
                locator,
                forest,
                points,
                indices,
                offsets,
                output,
            )
        )
    assert np.array_equal(output.view(np.uint64), before)


def test_empty_slots_points_and_fields_are_accepted() -> None:
    root_shape = i3(1, 1, 1)
    block_cells = i3(2, 2, 2)
    coord_to_rank, forest = make_forest(root_shape, lambda _level, _coord: False)
    locator = locator_args(
        f3(0.0, 0.0, 0.0),
        f3(2.0, 2.0, 2.0),
        root_shape,
        block_cells,
        coord_to_rank,
        forest,
    )
    output = np.empty((0, 0), dtype=np.float64)
    sample_refined_zero_order_point_groups(
        *sampling_args(
            np.empty((0, 0, 2, 2, 2), dtype=np.float64),
            i3(0, 0, 0),
            block_cells,
            i3(0, 0, 0),
            block_cells,
            np.empty(0, dtype=np.int64),
            locator,
            forest,
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            i3(0),
            output,
        )
    )
    assert output.shape == (0, 0)
