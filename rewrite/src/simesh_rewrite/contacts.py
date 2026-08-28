"""Validated TOP-002 raw refined contact target lookup."""

from __future__ import annotations

import numpy as np

from ._contacts import (
    fill_refined_contact_targets_unchecked,
    invalid_contact_direction_unchecked,
)
from ._storage import validate_indices_unchecked
from .foundation import INDEX_DTYPE, _require_index_triplet
from .forest_conformance import _require_index_array
from .morton import _root_volume
from .topology import _require_mapping_input


def _require_index_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous vector")
    return value


def fill_refined_contact_targets(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    source_leaf_ids: np.ndarray,
    directions: np.ndarray,
    target_node_ids: np.ndarray,
) -> None:
    """Fill raw target nodes for explicit refined leaf/direction queries."""
    root_shape = _require_index_triplet("root_shape", root_shape)
    root_count = _root_volume(root_shape)
    root_tuple = tuple(int(value) for value in root_shape)
    coord_to_rank = _require_mapping_input(
        "coord_to_rank",
        coord_to_rank,
        root_tuple,
    )
    node_levels = _require_index_vector("node_levels", node_levels)
    node_count = node_levels.shape[0]
    root_node_ids = _require_index_array(
        "root_node_ids", root_node_ids, (root_count,)
    )
    node_coords = _require_index_array(
        "node_coords", node_coords, (node_count, 3)
    )
    child_node_ids = _require_index_array(
        "child_node_ids", child_node_ids, (node_count, 8)
    )
    node_leaf_ids = _require_index_array(
        "node_leaf_ids", node_leaf_ids, (node_count,)
    )
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    source_leaf_ids = _require_index_vector(
        "source_leaf_ids", source_leaf_ids
    )
    query_count = source_leaf_ids.shape[0]
    directions = _require_index_array(
        "directions", directions, (query_count, 3)
    )
    target_node_ids = _require_index_array(
        "target_node_ids", target_node_ids, (query_count,)
    )
    if not target_node_ids.flags.writeable:
        raise ValueError("target_node_ids must be writable")

    inputs = (
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        source_leaf_ids,
        directions,
    )
    if any(np.shares_memory(target_node_ids, value) for value in inputs):
        raise ValueError("target_node_ids must not overlap contact inputs")

    invalid_source = int(
        validate_indices_unchecked(source_leaf_ids, leaf_node_ids.shape[0])
    )
    if invalid_source >= 0:
        raise ValueError(f"source_leaf_ids entry {invalid_source} is out of range")
    invalid_direction = int(invalid_contact_direction_unchecked(directions))
    if invalid_direction >= 0:
        raise ValueError(
            f"directions row {invalid_direction} must be noncenter values in [-1, 1]"
        )

    fill_refined_contact_targets_unchecked(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        source_leaf_ids,
        directions,
        target_node_ids,
    )


def refined_contact_targets(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    source_leaf_ids: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Allocate and return raw TOP-002 target node IDs."""
    source_leaf_ids = _require_index_vector(
        "source_leaf_ids", source_leaf_ids
    )
    output = np.empty(source_leaf_ids.shape[0], dtype=np.int64)
    fill_refined_contact_targets(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        source_leaf_ids,
        directions,
        output,
    )
    return output
