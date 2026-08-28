"""Validated BAL-001 global all-touch two-to-one balance policy."""

from __future__ import annotations

import numpy as np

from ._balance import first_refined_balance_violation_unchecked
from .contacts import _require_contact_forest_inputs


def validate_refined_all_touch_2to1(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
) -> None:
    """Raise for the first face/edge/corner leaf-level gap above one."""
    (
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
    ) = _require_contact_forest_inputs(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
    )
    source_leaf, column, target_node, offending_node = (
        first_refined_balance_violation_unchecked(
            root_shape,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
        )
    )
    if source_leaf >= 0:
        raise ValueError(
            "all-touch two-to-one balance violation at "
            f"source leaf {source_leaf}, direction {column}, "
            f"target node {target_node}, offending node {offending_node}"
        )
