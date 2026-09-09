"""Independent recursive Python reference for FST-002."""

from __future__ import annotations

import numpy as np


_INDEX_MAX = int(np.iinfo(np.int64).max)


def validate_refined_forest_arrays_reference(
    root_shape: np.ndarray,
    rank_to_coord: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    parent_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
) -> int:
    shape = tuple(int(value) for value in root_shape)
    next_node = 0
    next_leaf = 0
    max_level = 0

    def visit(
        level: int,
        coord: tuple[int, int, int],
        parent: int,
        extent: tuple[int, int, int],
    ) -> None:
        nonlocal next_node, next_leaf, max_level
        if next_node >= len(node_levels):
            raise ValueError("node stream is truncated")
        node = next_node
        next_node += 1
        if int(node_levels[node]) != level:
            raise ValueError("node level differs")
        if tuple(int(value) for value in node_coords[node]) != coord:
            raise ValueError("node coordinate differs")
        if int(parent_node_ids[node]) != parent:
            raise ValueError("node parent differs")
        max_level = max(max_level, level)
        leaf = int(node_leaf_ids[node])
        if leaf >= 0:
            if leaf != next_leaf or next_leaf >= len(leaf_node_ids):
                raise ValueError("leaf order differs")
            if int(leaf_node_ids[next_leaf]) != node:
                raise ValueError("leaf inverse differs")
            if np.any(child_node_ids[node] != -1):
                raise ValueError("leaf child sentinel differs")
            next_leaf += 1
            return
        if leaf != -1:
            raise ValueError("internal leaf sentinel differs")
        if any(value > _INDEX_MAX // 2 for value in extent):
            raise OverflowError("logical extent does not fit in int64")
        child_extent = tuple(2 * value for value in extent)
        for child in range(8):
            if int(child_node_ids[node, child]) != next_node:
                raise ValueError("child preorder differs")
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            visit(
                level + 1,
                tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
                node,
                child_extent,
            )

    for root, coordinate in enumerate(rank_to_coord):
        if int(root_node_ids[root]) != next_node:
            raise ValueError("root preorder differs")
        visit(
            1,
            tuple(int(value) for value in coordinate),
            -1,
            shape,
        )
    if next_node != len(node_levels):
        raise ValueError("trailing nodes remain")
    if next_leaf != len(leaf_node_ids):
        raise ValueError("trailing leaves remain")
    return max_level
