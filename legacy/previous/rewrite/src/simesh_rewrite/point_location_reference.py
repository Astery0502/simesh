"""Independent all-leaf scan reference for LOC-001."""

from __future__ import annotations

import math

import numpy as np


def _canonical_face(
    lower: np.float64,
    upper: np.float64,
    spacing: np.float64,
    face_index: int,
    domain_cells: int,
) -> np.float64:
    if face_index == 0:
        return lower
    if face_index == domain_cells:
        return upper
    offset = np.float64(np.float64(face_index) * spacing)
    return np.float64(lower + offset)


def refined_point_leaf_ids_reference(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Materialize canonical leaf boxes, then scan them for each point."""
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    for axis in range(3):
        if int(domain_cell_counts[axis]) != (
            int(root_shape[axis]) * int(block_cell_counts[axis])
        ):
            raise ValueError(
                "domain_cell_counts must equal root_shape * block_cell_counts"
            )

    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = np.float64(domain_upper[axis] - domain_lower[axis])
        base_spacing[axis] = np.float64(
            extent / np.float64(domain_cell_counts[axis])
        )

    leaf_count = int(leaf_node_ids.shape[0])
    bounds = np.empty((leaf_count, 2, 3), dtype=np.float64)
    for leaf_id in range(leaf_count):
        node_id = int(leaf_node_ids[leaf_id])
        level = int(node_levels[node_id])
        shift = level - 1
        scale = 1 << shift
        for axis in range(3):
            spacing = np.float64(
                math.ldexp(float(base_spacing[axis]), -shift)
            )
            total_cells = int(domain_cell_counts[axis]) * scale
            lower_index = (
                int(node_coords[node_id, axis])
                * int(block_cell_counts[axis])
            )
            upper_index = lower_index + int(block_cell_counts[axis])
            bounds[leaf_id, 0, axis] = _canonical_face(
                np.float64(domain_lower[axis]),
                np.float64(domain_upper[axis]),
                spacing,
                lower_index,
                total_cells,
            )
            bounds[leaf_id, 1, axis] = _canonical_face(
                np.float64(domain_lower[axis]),
                np.float64(domain_upper[axis]),
                spacing,
                upper_index,
                total_cells,
            )

    owners = np.full(points.shape[0], -1, dtype=np.int64)
    for point_index, point in enumerate(points):
        if np.any(point < domain_lower) or np.any(point >= domain_upper):
            continue
        match = -1
        for leaf_id in range(leaf_count):
            if np.all(point >= bounds[leaf_id, 0]) and np.all(
                point < bounds[leaf_id, 1]
            ):
                if match >= 0:
                    raise AssertionError("validated leaf boxes overlap")
                match = leaf_id
        if match < 0:
            raise AssertionError("validated leaf boxes do not cover the point")
        owners[point_index] = match
    return owners
