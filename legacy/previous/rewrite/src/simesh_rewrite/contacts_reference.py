"""Independent coordinate-dictionary reference for TOP-002."""

from __future__ import annotations

import numpy as np


def refined_contact_targets_reference(
    root_shape: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    source_leaf_ids: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    node_by_coordinate = {
        (
            int(node_levels[node]),
            *tuple(int(value) for value in node_coords[node]),
        ): node
        for node in range(node_levels.shape[0])
    }
    output = np.empty(source_leaf_ids.shape[0], dtype=np.int64)
    for query, leaf_id in enumerate(source_leaf_ids):
        source_node = int(leaf_node_ids[int(leaf_id)])
        source_level = int(node_levels[source_node])
        source_coord = tuple(int(value) for value in node_coords[source_node])
        target = tuple(
            source_coord[axis] + int(directions[query, axis])
            for axis in range(3)
        )
        level_shape = tuple(
            int(root_shape[axis]) * (1 << (source_level - 1))
            for axis in range(3)
        )
        if any(
            target[axis] < 0 or target[axis] >= level_shape[axis]
            for axis in range(3)
        ):
            output[query] = -1
            continue
        for level in range(1, source_level + 1):
            shift = source_level - level
            coordinate = tuple(value >> shift for value in target)
            node = node_by_coordinate[(level, *coordinate)]
            if node_leaf_ids[node] >= 0 or level == source_level:
                output[query] = node
                break
    return output
