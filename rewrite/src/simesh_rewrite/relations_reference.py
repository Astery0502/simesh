"""Independent common-lattice reference for REL-001."""

from __future__ import annotations

from typing import Final

import numpy as np


RELATION_PHYSICAL: Final = 1
RELATION_COARSER: Final = 2
RELATION_SAME: Final = 3
RELATION_FINER: Final = 4


def balanced_refined_relations_reference(
    root_shape: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_ids: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate exact balanced relation records for selected leaves.

    Inputs are expected to belong to one FST-002/BAL-001 lifecycle.  The
    implementation uses only Python-integer boxes on a common maximum-level
    lattice: it does not call TOP/MOR or inspect parent/child pointers.
    """
    primary_count = int(leaf_ids.shape[0])
    direction_count = int(directions.shape[0])
    relation_kinds = np.empty(
        (primary_count, direction_count), dtype=np.uint8
    )
    physical_masks = np.empty(
        (primary_count, direction_count), dtype=np.uint8
    )
    source_counts = np.empty(
        (primary_count, direction_count), dtype=np.uint8
    )
    source_leaf_ids = np.full(
        (primary_count, direction_count, 4), -1, dtype=np.int64
    )

    if primary_count == 0 or direction_count == 0:
        return (
            relation_kinds,
            physical_masks,
            source_counts,
            source_leaf_ids,
        )

    maximum_level = max(
        int(node_levels[int(node_id)]) for node_id in leaf_node_ids
    )
    domain_extent = tuple(
        int(root_shape[axis]) * (1 << (maximum_level - 1))
        for axis in range(3)
    )

    leaf_boxes = []
    for leaf_id, node_id_value in enumerate(leaf_node_ids):
        node_id = int(node_id_value)
        level = int(node_levels[node_id])
        scale = 1 << (maximum_level - level)
        lower = tuple(
            int(node_coords[node_id, axis]) * scale for axis in range(3)
        )
        upper = tuple(lower[axis] + scale for axis in range(3))
        child_column = (
            (int(node_coords[node_id, 0]) & 1)
            + 2 * (int(node_coords[node_id, 1]) & 1)
            + 4 * (int(node_coords[node_id, 2]) & 1)
        )
        leaf_boxes.append(
            (leaf_id, level, lower, upper, child_column)
        )

    for primary, source_leaf_id_value in enumerate(leaf_ids):
        source_leaf_id = int(source_leaf_id_value)
        source_node_id = int(leaf_node_ids[source_leaf_id])
        source_level = int(node_levels[source_node_id])
        source_scale = 1 << (maximum_level - source_level)
        source_lower = tuple(
            int(node_coords[source_node_id, axis]) * source_scale
            for axis in range(3)
        )

        for direction_index, direction in enumerate(directions):
            reduced = [int(direction[axis]) for axis in range(3)]
            physical_mask = 0
            for axis in range(3):
                if reduced[axis] < 0 and source_lower[axis] == 0:
                    physical_mask |= 1 << axis
                    reduced[axis] = 0
                elif (
                    reduced[axis] > 0
                    and source_lower[axis] + source_scale
                    == domain_extent[axis]
                ):
                    physical_mask |= 1 << axis
                    reduced[axis] = 0

            physical_masks[primary, direction_index] = physical_mask
            if reduced == [0, 0, 0]:
                relation_kinds[primary, direction_index] = RELATION_PHYSICAL
                source_counts[primary, direction_index] = 0
                continue

            target_lower = tuple(
                source_lower[axis] + reduced[axis] * source_scale
                for axis in range(3)
            )
            phase_lower = []
            phase_upper = []
            for axis in range(3):
                lower = target_lower[axis]
                upper = lower + source_scale
                if reduced[axis] == 0 or source_scale == 1:
                    phase_lower.append(lower)
                    phase_upper.append(upper)
                    continue
                midpoint = lower + source_scale // 2
                if reduced[axis] < 0:
                    phase_lower.append(midpoint)
                    phase_upper.append(upper)
                else:
                    phase_lower.append(lower)
                    phase_upper.append(midpoint)

            candidates = [
                (leaf_id, level, child_column)
                for leaf_id, level, lower, upper, child_column in leaf_boxes
                if all(
                    max(lower[axis], phase_lower[axis])
                    < min(upper[axis], phase_upper[axis])
                    for axis in range(3)
                )
            ]
            candidate_levels = {level for _, level, _ in candidates}
            if len(candidate_levels) != 1:
                raise ValueError(
                    "reference requires one balanced relation level"
                )
            target_level = next(iter(candidate_levels))
            if target_level == source_level - 1:
                kind = RELATION_COARSER
            elif target_level == source_level:
                kind = RELATION_SAME
            elif target_level == source_level + 1:
                kind = RELATION_FINER
                candidates.sort(key=lambda candidate: candidate[2])
            else:
                raise ValueError(
                    "reference requires an all-touch two-to-one forest"
                )

            count = len(candidates)
            if count < 1 or count > 4:
                raise ValueError(
                    "reference relation source count must be in [1, 4]"
                )
            relation_kinds[primary, direction_index] = kind
            source_counts[primary, direction_index] = count
            source_leaf_ids[primary, direction_index, :count] = [
                leaf_id for leaf_id, _, _ in candidates
            ]

    return (
        relation_kinds,
        physical_masks,
        source_counts,
        source_leaf_ids,
    )
