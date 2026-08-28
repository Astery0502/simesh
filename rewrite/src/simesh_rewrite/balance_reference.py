"""Independent common-lattice pairwise reference for BAL-001."""

from __future__ import annotations

import numpy as np


def first_all_touch_2to1_violation_reference(
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
):
    if leaf_node_ids.size == 0:
        return None
    max_level = int(node_levels[leaf_node_ids].max())
    lowers = []
    uppers = []
    for node in leaf_node_ids:
        level = int(node_levels[node])
        scale = 1 << (max_level - level)
        lower = np.asarray(node_coords[node], dtype=object) * scale
        lowers.append(tuple(int(value) for value in lower))
        uppers.append(tuple(int(value) + scale for value in lower))
    for left in range(leaf_node_ids.size):
        left_level = int(node_levels[leaf_node_ids[left]])
        for right in range(left + 1, leaf_node_ids.size):
            if any(
                max(lowers[left][axis], lowers[right][axis])
                > min(uppers[left][axis], uppers[right][axis])
                for axis in range(3)
            ):
                continue
            right_level = int(node_levels[leaf_node_ids[right]])
            if abs(left_level - right_level) > 1:
                return left, right
    return None


def is_refined_all_touch_2to1_reference(
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
) -> bool:
    return (
        first_all_touch_2to1_violation_reference(
            node_levels,
            node_coords,
            leaf_node_ids,
        )
        is None
    )
