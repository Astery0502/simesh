"""Independent scalar references for SAM-004 and SAM-005."""

from __future__ import annotations

import math

import numpy as np


def _face(
    domain_lower: np.float64,
    domain_upper: np.float64,
    spacing: np.float64,
    index: int,
    count: int,
) -> np.float64:
    if index == 0:
        return np.float64(domain_lower)
    if index == count:
        return np.float64(domain_upper)
    offset = np.float64(index) * np.float64(spacing)
    return np.float64(np.float64(domain_lower) + offset)


def _leaf_axis(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_id: int,
    axis: int,
) -> tuple[np.float64, int, int, np.float64]:
    node_id = int(leaf_node_ids[leaf_id])
    shift = int(node_levels[node_id]) - 1
    level_cells = int(domain_cell_counts[axis]) * (1 << shift)
    extent = np.float64(domain_upper[axis]) - np.float64(domain_lower[axis])
    base_spacing = extent / np.float64(domain_cell_counts[axis])
    spacing = np.ldexp(base_spacing, -shift)
    global_lower = int(node_coords[node_id, axis]) * int(
        block_cell_counts[axis]
    )
    block_lower = _face(
        np.float64(domain_lower[axis]),
        np.float64(domain_upper[axis]),
        spacing,
        global_lower,
        level_cells,
    )
    return block_lower, global_lower, level_cells, spacing


def _local_cell(
    point: np.float64,
    domain_lower: np.float64,
    domain_upper: np.float64,
    spacing: np.float64,
    level_cells: int,
    global_lower: int,
    block_cells: int,
) -> int:
    owner = global_lower
    for index in range(global_lower + 1, global_lower + block_cells):
        if _face(
            domain_lower,
            domain_upper,
            spacing,
            index,
            level_cells,
        ) <= point:
            owner = index
    return owner - global_lower


def _axis_stencil(
    point: np.float64,
    block_lower: np.float64,
    spacing: np.float64,
) -> tuple[int, np.float64]:
    delta = np.float64(point) - np.float64(block_lower)
    ratio = delta / np.float64(spacing)
    normalized = ratio - np.float64(0.5)
    left = math.floor(float(normalized))
    weight = normalized - np.float64(left)
    return left, np.float64(weight)


def _lerp(
    left: np.float64,
    right: np.float64,
    weight: np.float64,
) -> np.float64:
    one_minus = np.float64(1.0) - np.float64(weight)
    left_term = np.float64(left) * one_minus
    right_term = np.float64(right) * np.float64(weight)
    return np.float64(left_term + right_term)


def sample_refined_zero_order_point_groups_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    slot_leaf_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    slot_point_offsets: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Copy grouped owner-cell values by scalar canonical-face scans."""
    for slot, leaf_value in enumerate(slot_leaf_ids):
        leaf_id = int(leaf_value)
        axis_geometry = [
            _leaf_axis(
                domain_lower,
                domain_upper,
                domain_cell_counts,
                block_cell_counts,
                node_levels,
                node_coords,
                leaf_node_ids,
                leaf_id,
                axis,
            )
            for axis in range(3)
        ]
        for order in range(
            int(slot_point_offsets[slot]),
            int(slot_point_offsets[slot + 1]),
        ):
            point_index = int(point_indices[order])
            local = []
            for axis in range(3):
                _, global_lower, level_cells, spacing = axis_geometry[axis]
                local.append(
                    _local_cell(
                        np.float64(points[point_index, axis]),
                        np.float64(domain_lower[axis]),
                        np.float64(domain_upper[axis]),
                        spacing,
                        level_cells,
                        global_lower,
                        int(block_cell_counts[axis]),
                    )
                )
            source = tuple(
                int(interior_lower[axis]) + local[axis] for axis in range(3)
            )
            for field in range(payload.shape[1]):
                point_values[point_index, field] = payload[
                    (slot, field, *source)
                ]


def sample_refined_trilinear_point_groups_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    slot_leaf_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    slot_point_offsets: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Evaluate grouped trilinear values with the frozen scalar blend tree."""
    for slot, leaf_value in enumerate(slot_leaf_ids):
        leaf_id = int(leaf_value)
        axis_geometry = [
            _leaf_axis(
                domain_lower,
                domain_upper,
                domain_cell_counts,
                block_cell_counts,
                node_levels,
                node_coords,
                leaf_node_ids,
                leaf_id,
                axis,
            )
            for axis in range(3)
        ]
        for order in range(
            int(slot_point_offsets[slot]),
            int(slot_point_offsets[slot + 1]),
        ):
            point_index = int(point_indices[order])
            left = []
            weight = []
            for axis in range(3):
                block_lower, _, _, spacing = axis_geometry[axis]
                left_index, axis_weight = _axis_stencil(
                    np.float64(points[point_index, axis]),
                    block_lower,
                    spacing,
                )
                left.append(int(interior_lower[axis]) + left_index)
                weight.append(axis_weight)
            right = [index + 1 for index in left]
            for field in range(payload.shape[1]):
                c00 = _lerp(
                    payload[slot, field, left[0], left[1], left[2]],
                    payload[slot, field, left[0], left[1], right[2]],
                    weight[2],
                )
                c01 = _lerp(
                    payload[slot, field, left[0], right[1], left[2]],
                    payload[slot, field, left[0], right[1], right[2]],
                    weight[2],
                )
                c10 = _lerp(
                    payload[slot, field, right[0], left[1], left[2]],
                    payload[slot, field, right[0], left[1], right[2]],
                    weight[2],
                )
                c11 = _lerp(
                    payload[slot, field, right[0], right[1], left[2]],
                    payload[slot, field, right[0], right[1], right[2]],
                    weight[2],
                )
                c0 = _lerp(c00, c01, weight[1])
                c1 = _lerp(c10, c11, weight[1])
                point_values[point_index, field] = _lerp(
                    c0,
                    c1,
                    weight[0],
                )
