from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.forest import refined_forest
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import refined_point_leaf_ids
from simesh_rewrite.refined_sampling import (
    sample_refined_trilinear_point_groups,
)
from simesh_rewrite.refined_sampling_reference import (
    sample_refined_trilinear_point_groups_reference,
)
from simesh_rewrite.sampling import sample_level1_trilinear


EPS = np.finfo(np.float64).eps


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def level_one_artifacts(
    root_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object]:
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ones(int(np.prod(root_shape)), dtype=np.bool_),
    )
    return coord_to_rank, rank_to_coord, forest


def completed_affine_payload(
    slot_leaf_ids: np.ndarray,
    block_cells: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    field_count: int,
) -> np.ndarray:
    spatial = tuple(int(value) + 2 for value in block_cells)
    payload = np.empty((slot_leaf_ids.size, field_count, *spatial))
    base_spacing = (domain_upper - domain_lower) / domain_cell_counts
    for slot, leaf_value in enumerate(slot_leaf_ids):
        node_id = int(leaf_node_ids[int(leaf_value)])
        shift = int(node_levels[node_id]) - 1
        spacing = np.ldexp(base_spacing, -shift)
        global_lower = node_coords[node_id] * block_cells
        block_lower = domain_lower + global_lower.astype(np.float64) * spacing
        axes = [
            block_lower[axis]
            + (np.arange(spatial[axis], dtype=np.float64) - 0.5) * spacing[axis]
            for axis in range(3)
        ]
        affine = (
            axes[0][:, None, None]
            + 2.0 * axes[1][None, :, None]
            + 3.0 * axes[2][None, None, :]
        )
        for field in range(field_count):
            payload[slot, field] = affine + field * 10.0
    return payload


def call_args(
    payload: np.ndarray,
    slot_leaf_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cells: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    offsets: np.ndarray,
    output: np.ndarray,
) -> tuple[object, ...]:
    interior_lower = i3(1, 1, 1)
    interior_upper = interior_lower + block_cells
    return (
        payload,
        i3(0, 0, 0),
        np.asarray(payload.shape[2:], dtype=np.int64),
        interior_lower,
        interior_upper,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cells,
        node_levels,
        node_coords,
        leaf_node_ids,
        points,
        point_indices,
        offsets,
        output,
    )


def test_mixed_refined_faces_match_scalar_tree_and_affine_bound() -> None:
    block_cells = i3(4, 4, 4)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(1.0, 1.0, 1.0)
    domain_cells = block_cells.copy()
    node_levels = np.asarray([1, *([2] * 8)], dtype=np.int64)
    node_coords = np.asarray(
        [
            (0, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ],
        dtype=np.int64,
    )
    leaf_node_ids = np.arange(1, 9, dtype=np.int64)
    slot_leaf_ids = np.asarray([7, 0, 4, 2], dtype=np.int64)
    points: list[np.ndarray] = []
    offsets = [0]
    for leaf_value in slot_leaf_ids:
        bits = node_coords[int(leaf_node_ids[int(leaf_value)])].astype(np.float64)
        lower = 0.5 * bits
        upper = lower + 0.5
        spacing = np.full(3, 0.125)
        points.extend(
            (
                lower.copy(),
                0.5 * (lower + upper),
                lower + spacing,
                np.nextafter(upper, lower),
            )
        )
        offsets.append(len(points))
    point_array = np.ascontiguousarray(points, dtype=np.float64)
    point_indices = np.arange(point_array.shape[0], dtype=np.int64)
    offset_array = np.asarray(offsets, dtype=np.int64)
    payload = completed_affine_payload(
        slot_leaf_ids,
        block_cells,
        domain_lower,
        domain_upper,
        domain_cells,
        node_levels,
        node_coords,
        leaf_node_ids,
        3,
    )
    output = np.full((point_array.shape[0] + 1, 3), -91.0)
    output[-1] = np.asarray([np.nan, -0.0, np.inf])
    expected = output.copy()
    sample_refined_trilinear_point_groups_reference(
        payload,
        i3(1, 1, 1),
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        node_levels,
        node_coords,
        leaf_node_ids,
        np.vstack((point_array, f3(0.1, 0.1, 0.1))),
        point_indices,
        offset_array,
        expected,
    )
    points_with_unselected = np.ascontiguousarray(
        np.vstack((point_array, f3(0.1, 0.1, 0.1))), dtype=np.float64
    )
    sample_refined_trilinear_point_groups(
        *call_args(
            payload,
            slot_leaf_ids,
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            node_levels,
            node_coords,
            leaf_node_ids,
            points_with_unselected,
            point_indices,
            offset_array,
            output,
        )
    )
    assert np.array_equal(output.view(np.uint64), expected.view(np.uint64))
    analytic = np.empty((point_array.shape[0], 3), dtype=np.float64)
    base = (
        point_array[:, 0]
        + 2.0 * point_array[:, 1]
        + 3.0 * point_array[:, 2]
    )
    for field in range(3):
        analytic[:, field] = base + field * 10.0
    scale = max(1.0, float(np.max(np.abs(analytic))))
    assert float(np.max(np.abs(output[:-1] - analytic))) <= 64.0 * EPS * scale


def test_zero_weights_still_load_all_eight_values() -> None:
    block_cells = i3(2, 2, 2)
    payload = np.ones((1, 1, 4, 4, 4), dtype=np.float64)
    payload[0, 0, 2, 2, 2] = np.nan
    output = np.zeros((1, 1), dtype=np.float64)
    sample_refined_trilinear_point_groups(
        *call_args(
            payload,
            i3(0),
            f3(0.0, 0.0, 0.0),
            f3(2.0, 2.0, 2.0),
            block_cells,
            block_cells,
            i3(1),
            np.asarray([[0, 0, 0]], dtype=np.int64),
            i3(0),
            np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64),
            i3(0),
            i3(0, 1),
            output,
        )
    )
    assert np.isnan(output[0, 0])


def test_repeated_slots_and_point_indices_use_last_write() -> None:
    block_cells = i3(2, 2, 2)
    payload = np.empty((2, 1, 4, 4, 4), dtype=np.float64)
    payload[0] = 1.0
    payload[1] = 2.0
    output = np.asarray([[-1.0]], dtype=np.float64)
    sample_refined_trilinear_point_groups(
        *call_args(
            payload,
            i3(0, 0),
            f3(0.0, 0.0, 0.0),
            f3(2.0, 2.0, 2.0),
            block_cells,
            block_cells,
            i3(1),
            np.asarray([[0, 0, 0]], dtype=np.int64),
            i3(0),
            np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64),
            i3(0, 0, 0),
            i3(0, 2, 3),
            output,
        )
    )
    assert output[0, 0] == 2.0


@pytest.mark.parametrize(
    ("sample_lower_values", "sample_upper_values", "output_shape"),
    [
        ((-2.0, 1.0, -1.0), (6.0, 4.0, 3.0), (8, 3, 2)),
        ((-2.0, 1.0, -1.0), (6.0, 4.0, 3.0), (16, 6, 4)),
        ((-2.0, 1.0, -1.0), (6.0, 4.0, 3.0), (4, 2, 1)),
        ((-1.75, 1.25, -0.75), (5.75, 3.75, 2.75), (7, 4, 3)),
    ],
    ids=("native-full-domain", "upsampled", "downsampled", "subdomain"),
)
def test_level_one_output_centers_reduce_bitwise_to_fused_sampler(
    sample_lower_values: tuple[float, float, float],
    sample_upper_values: tuple[float, float, float],
    output_shape: tuple[int, int, int],
) -> None:
    root_shape = i3(2, 1, 1)
    block_cells = i3(4, 3, 2)
    domain_cells = root_shape * block_cells
    domain_lower = f3(-2.0, 1.0, -1.0)
    domain_upper = f3(6.0, 4.0, 3.0)
    coord_to_rank, rank_to_coord, forest = level_one_artifacts(root_shape)
    slot_leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    payload = completed_affine_payload(
        slot_leaf_ids,
        block_cells,
        domain_lower,
        domain_upper,
        domain_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        2,
    )
    sample_lower = f3(*sample_lower_values)
    sample_upper = f3(*sample_upper_values)
    output_spacing = (sample_upper - sample_lower) / np.asarray(output_shape)
    points = np.empty((int(np.prod(output_shape)), 3), dtype=np.float64)
    for row, index in enumerate(np.ndindex(output_shape)):
        for axis in range(3):
            factor = np.float64(index[axis]) + np.float64(0.5)
            offset = factor * np.float64(output_spacing[axis])
            points[row, axis] = np.float64(sample_lower[axis]) + offset
    owners = refined_point_leaf_ids(
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
        points,
    )
    grouped = []
    offsets = [0]
    for leaf_id in slot_leaf_ids:
        grouped.extend(np.flatnonzero(owners == leaf_id).tolist())
        offsets.append(len(grouped))
    point_values = np.full((points.shape[0], 2), -1.0)
    sample_refined_trilinear_point_groups(
        *call_args(
            payload,
            slot_leaf_ids,
            domain_lower,
            domain_upper,
            domain_cells,
            block_cells,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            points,
            np.asarray(grouped, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            point_values,
        )
    )
    fused = np.full((2, *output_shape), -1.0)
    sample_level1_trilinear(
        payload,
        i3(0, 0, 0),
        np.asarray(payload.shape[2:], dtype=np.int64),
        i3(1, 1, 1),
        i3(1, 1, 1) + block_cells,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        fused,
    )
    point_grid = point_values.reshape((*output_shape, 2)).transpose(3, 0, 1, 2)
    assert np.array_equal(point_grid.view(np.uint64), fused.view(np.uint64))


def test_output_alias_with_payload_is_rejected() -> None:
    block_cells = i3(2, 2, 2)
    payload = np.ones((1, 1, 4, 4, 4), dtype=np.float64)
    aliased_output = payload.reshape(-1)[:1].reshape(1, 1)
    with pytest.raises(ValueError, match="must not overlap"):
        sample_refined_trilinear_point_groups(
            *call_args(
                payload,
                i3(0),
                f3(0.0, 0.0, 0.0),
                f3(2.0, 2.0, 2.0),
                block_cells,
                block_cells,
                i3(1),
                np.asarray([[0, 0, 0]], dtype=np.int64),
                i3(0),
                np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64),
                i3(0),
                i3(0, 1),
                aliased_output,
            )
        )


@pytest.mark.parametrize("failure", ["owner", "nonfinite", "upper", "halo"])
def test_validation_failures_preserve_complete_output(failure: str) -> None:
    block_cells = i3(2, 2, 2)
    payload = np.ones((1, 2, 4, 4, 4), dtype=np.float64)
    points = np.asarray([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=np.float64)
    valid_upper = i3(4, 4, 4)
    if failure == "owner":
        points[1, 0] = -0.1
    elif failure == "nonfinite":
        points[1, 1] = np.inf
    elif failure == "upper":
        points[1, 2] = 2.0
    else:
        valid_upper[0] = 3
    output = np.asarray([[1.0, -0.0], [np.nan, np.inf]], dtype=np.float64)
    before = output.view(np.uint64).copy()
    args = list(
        call_args(
            payload,
            i3(0),
            f3(0.0, 0.0, 0.0),
            f3(2.0, 2.0, 2.0),
            block_cells,
            block_cells,
            i3(1),
            np.asarray([[0, 0, 0]], dtype=np.int64),
            i3(0),
            points,
            i3(0, 1),
            i3(0, 2),
            output,
        )
    )
    args[2] = valid_upper
    with pytest.raises(ValueError):
        sample_refined_trilinear_point_groups(*args)
    assert np.array_equal(output.view(np.uint64), before)
