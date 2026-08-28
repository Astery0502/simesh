"""Independent recursive Python reference for FST-001."""

from __future__ import annotations

import numpy as np

from .forest import RefinedForest


_INDEX_MAX = int(np.iinfo(np.int64).max)


def refined_forest_reference(
    root_shape: np.ndarray,
    rank_to_coord: np.ndarray,
    is_leaf: np.ndarray,
) -> RefinedForest:
    """Reconstruct a valid forest with Python integers and recursive lists."""
    shape = tuple(int(value) for value in root_shape)
    root_count = shape[0] * shape[1] * shape[2]
    if len(shape) != 3 or any(extent <= 0 for extent in shape):
        raise ValueError("root_shape must contain three positive extents")
    if rank_to_coord.shape != (root_count, 3):
        raise ValueError("rank_to_coord has the wrong shape")
    if is_leaf.ndim != 1:
        raise ValueError("is_leaf must be one-dimensional")

    node_count = int(is_leaf.size)
    leaf_count = int(np.count_nonzero(is_leaf))
    node_levels = np.full(node_count, -1, dtype=np.int64)
    node_coords = np.full((node_count, 3), -1, dtype=np.int64)
    parent_node_ids = np.full(node_count, -1, dtype=np.int64)
    child_node_ids = np.full((node_count, 8), -1, dtype=np.int64)
    node_leaf_ids = np.full(node_count, -1, dtype=np.int64)
    leaf_node_ids = np.full(leaf_count, -1, dtype=np.int64)
    root_node_ids = np.full(root_count, -1, dtype=np.int64)
    next_node = 0
    next_leaf = 0
    max_level = 0

    def visit(
        level: int,
        coord: tuple[int, int, int],
        extent: tuple[int, int, int],
        parent: int,
    ) -> int:
        nonlocal next_node, next_leaf, max_level
        if next_node >= node_count:
            raise ValueError("forest stream is truncated")
        node = next_node
        next_node += 1
        max_level = max(max_level, level)
        node_levels[node] = level
        node_coords[node] = coord
        parent_node_ids[node] = parent
        if bool(is_leaf[node]):
            node_leaf_ids[node] = next_leaf
            leaf_node_ids[next_leaf] = node
            next_leaf += 1
            return node
        if any(value > _INDEX_MAX // 2 for value in extent):
            raise OverflowError("refined logical grid does not fit in int64")
        child_extent = tuple(2 * value for value in extent)
        for child in range(8):
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            child_coord = tuple(
                2 * coord[axis] + bits[axis] for axis in range(3)
            )
            child_node_ids[node, child] = next_node
            visit(level + 1, child_coord, child_extent, node)
        return node

    for root, coordinate in enumerate(rank_to_coord):
        root_node_ids[root] = next_node
        visit(1, tuple(int(value) for value in coordinate), shape, -1)
    if next_node != node_count:
        raise ValueError("forest stream has trailing nodes")

    return RefinedForest(
        node_levels,
        node_coords,
        parent_node_ids,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        root_node_ids,
        max_level,
    )
