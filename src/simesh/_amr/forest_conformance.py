"""Validated FST-002 flat refined-forest artifact conformance."""

from __future__ import annotations

import numpy as np

from simesh._kernels.primitives._forest_conformance import validate_refined_forest_arrays_unchecked
from simesh._kernels.primitives._topology import validate_level1_maps_unchecked
from simesh._amr.foundation import INDEX_DTYPE, _require_index_triplet
from simesh._amr.morton import _root_volume
from simesh._amr.topology import _require_mapping_input


def _require_index_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def validate_refined_forest_arrays(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    parent_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
) -> int:
    """Return exact max level after complete FST-001 artifact conformance."""
    root_shape = _require_index_triplet("root_shape", root_shape)
    root_count = _root_volume(root_shape)
    root_tuple = tuple(int(value) for value in root_shape)
    coord_to_rank = _require_mapping_input(
        "coord_to_rank",
        coord_to_rank,
        root_tuple,
    )
    rank_to_coord = _require_mapping_input(
        "rank_to_coord",
        rank_to_coord,
        (root_count, 3),
    )

    if not isinstance(node_levels, np.ndarray):
        raise TypeError("node_levels must be a NumPy array")
    if node_levels.dtype != INDEX_DTYPE:
        raise TypeError("node_levels must have dtype int64")
    if node_levels.ndim != 1 or not node_levels.flags.c_contiguous:
        raise ValueError("node_levels must be a C-contiguous vector")
    node_count = node_levels.shape[0]
    if node_count == 0:
        raise ValueError("node_levels must contain at least one root node")
    if not isinstance(leaf_node_ids, np.ndarray):
        raise TypeError("leaf_node_ids must be a NumPy array")
    if leaf_node_ids.dtype != INDEX_DTYPE:
        raise TypeError("leaf_node_ids must have dtype int64")
    if leaf_node_ids.ndim != 1 or not leaf_node_ids.flags.c_contiguous:
        raise ValueError("leaf_node_ids must be a C-contiguous vector")
    leaf_count = leaf_node_ids.shape[0]
    if leaf_count == 0:
        raise ValueError("leaf_node_ids must contain at least one leaf")

    root_node_ids = _require_index_array(
        "root_node_ids", root_node_ids, (root_count,)
    )
    node_coords = _require_index_array(
        "node_coords", node_coords, (node_count, 3)
    )
    parent_node_ids = _require_index_array(
        "parent_node_ids", parent_node_ids, (node_count,)
    )
    child_node_ids = _require_index_array(
        "child_node_ids", child_node_ids, (node_count, 8)
    )
    node_leaf_ids = _require_index_array(
        "node_leaf_ids", node_leaf_ids, (node_count,)
    )

    invalid_rank = int(
        validate_level1_maps_unchecked(
            root_shape,
            coord_to_rank,
            rank_to_coord,
        )
    )
    if invalid_rank >= 0:
        raise ValueError(
            f"MOR maps are not a dense in-box inverse at rank {invalid_rank}"
        )

    status, bad_node, bad_leaf, max_level = (
        validate_refined_forest_arrays_unchecked(
            root_shape,
            rank_to_coord,
            root_node_ids,
            node_levels,
            node_coords,
            parent_node_ids,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
        )
    )
    if status == 0:
        return int(max_level)
    if status == 1:
        raise ValueError(f"node stream ends before expected node {bad_node}")
    if status == 2:
        raise ValueError(f"node metadata is inconsistent at node {bad_node}")
    if status == 3:
        raise ValueError(
            f"leaf stream ends at leaf {bad_leaf} for node {bad_node}"
        )
    if status == 4:
        raise ValueError(
            f"node/leaf maps are inconsistent at node {bad_node}, leaf {bad_leaf}"
        )
    if status == 5:
        raise ValueError(f"leaf node {bad_node} has a non-sentinel child")
    if status == 6:
        raise ValueError(f"internal node {bad_node} has an invalid leaf sentinel")
    if status == 7:
        raise OverflowError(
            f"logical grid below internal node {bad_node} does not fit in int64"
        )
    if status == 8:
        raise ValueError(f"child preorder is inconsistent at node {bad_node}")
    if status == 9:
        raise ValueError(
            f"root rank {bad_leaf} does not begin at expected node {bad_node}"
        )
    if status == 10:
        raise ValueError(f"unconsumed trailing nodes start at node {bad_node}")
    if status == 11:
        raise ValueError(f"unconsumed trailing leaves start at leaf {bad_leaf}")
    raise RuntimeError(f"unexpected forest conformance status {status}")
