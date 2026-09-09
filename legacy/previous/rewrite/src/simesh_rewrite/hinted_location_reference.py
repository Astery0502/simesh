"""Independent scalar hint-containment reference for HLO-001."""

from __future__ import annotations

import math

import numpy as np

from .point_location_reference import refined_point_leaf_ids_reference


def _face(
    lower: np.float64,
    upper: np.float64,
    spacing: np.float64,
    index: int,
    cell_count: int,
) -> np.float64:
    if index == 0:
        return lower
    if index == cell_count:
        return upper
    offset = np.float64(np.float64(index) * spacing)
    return np.float64(lower + offset)


def _hint_bounds(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    base_spacing: np.ndarray,
    hint: int,
) -> tuple[np.ndarray, np.ndarray]:
    node = int(leaf_node_ids[hint])
    shift = int(node_levels[node]) - 1
    scale = 1 << shift
    lower = np.empty(3, dtype=np.float64)
    upper = np.empty(3, dtype=np.float64)
    for axis in range(3):
        spacing = np.float64(math.ldexp(float(base_spacing[axis]), -shift))
        total_cells = int(domain_cell_counts[axis]) * scale
        first = int(node_coords[node, axis]) * int(block_cell_counts[axis])
        last = first + int(block_cell_counts[axis])
        lower[axis] = _face(
            np.float64(domain_lower[axis]),
            np.float64(domain_upper[axis]),
            spacing,
            first,
            total_cells,
        )
        upper[axis] = _face(
            np.float64(domain_lower[axis]),
            np.float64(domain_upper[axis]),
            spacing,
            last,
            total_cells,
        )
    return lower, upper


def refined_point_leaf_ids_with_hints_reference(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    hint_leaf_ids: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
    """Return scalar hint-or-independent-scan owners and exact statistics."""
    del max_level, coord_to_rank, root_node_ids, child_node_ids, node_leaf_ids
    fallback_owners = refined_point_leaf_ids_reference(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        points,
    )
    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = np.float64(domain_upper[axis] - domain_lower[axis])
        base_spacing[axis] = np.float64(
            extent / np.float64(domain_cell_counts[axis])
        )

    owners = fallback_owners.copy()
    interior = 0
    exterior = 0
    candidates = 0
    hits = 0
    for point_index, point in enumerate(points):
        hint = int(hint_leaf_ids[point_index])
        if hint >= 0:
            candidates += 1
        if int(fallback_owners[point_index]) < 0:
            exterior += 1
            continue
        interior += 1
        if hint < 0:
            continue
        hint_lower, hint_upper = _hint_bounds(
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            node_levels,
            node_coords,
            leaf_node_ids,
            base_spacing,
            hint,
        )
        if np.all(point >= hint_lower) and np.all(point < hint_upper):
            owners[point_index] = hint
            hits += 1

    stats = (
        int(points.shape[0]),
        interior,
        exterior,
        candidates,
        hits,
        interior - hits,
    )
    return owners, stats
