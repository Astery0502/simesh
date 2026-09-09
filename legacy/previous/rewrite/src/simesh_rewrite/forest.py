"""Validated FST-001 Cartesian 3D refined-forest reconstruction."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._forest import (
    fill_refined_forest_unchecked,
    validate_refined_forest_unchecked,
)
from ._topology import validate_level1_maps_unchecked
from .foundation import INDEX_DTYPE, _require_index_triplet
from .morton import _root_volume
from .topology import _require_mapping_input


class RefinedForest(NamedTuple):
    """Immutable grouping of the explicit caller-independent forest arrays."""

    node_levels: np.ndarray
    node_coords: np.ndarray
    parent_node_ids: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    root_node_ids: np.ndarray
    max_level: int


def _require_leaf_flags(is_leaf: np.ndarray) -> np.ndarray:
    if not isinstance(is_leaf, np.ndarray):
        raise TypeError("is_leaf must be a NumPy array")
    if is_leaf.dtype != np.dtype(np.bool_):
        raise TypeError("is_leaf must have dtype bool")
    if is_leaf.ndim != 1:
        raise ValueError(f"is_leaf must have shape (node,), got {is_leaf.shape}")
    if not is_leaf.flags.c_contiguous:
        raise ValueError("is_leaf must be C-contiguous")
    return is_leaf


def _require_index_output(
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
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _validate_forest_inputs(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    is_leaf: np.ndarray,
) -> tuple[int, int, int, int]:
    root_shape = _require_index_triplet("root_shape", root_shape)
    root_count = _root_volume(root_shape)
    root_tuple = tuple(int(value) for value in root_shape)
    coord_to_rank = _require_mapping_input(
        "coord_to_rank", coord_to_rank, root_tuple
    )
    rank_to_coord = _require_mapping_input(
        "rank_to_coord", rank_to_coord, (root_count, 3)
    )
    is_leaf = _require_leaf_flags(is_leaf)

    invalid_rank = int(
        validate_level1_maps_unchecked(
            root_shape,
            coord_to_rank,
            rank_to_coord,
        )
    )
    if invalid_rank >= 0:
        raise ValueError(
            f"Morton maps are not a dense in-box inverse at rank {invalid_rank}"
        )

    status, bad_node, max_level = validate_refined_forest_unchecked(
        root_shape,
        rank_to_coord,
        is_leaf.view(np.uint8),
    )
    if status == 1:
        raise ValueError(
            f"is_leaf ends at node {bad_node} before all root trees are complete"
        )
    if status == 2:
        raise OverflowError(
            f"refined logical grid below node {bad_node} does not fit in int64"
        )
    if status == 3:
        raise ValueError(f"is_leaf has trailing nodes starting at node {bad_node}")
    if status != 0:
        raise RuntimeError(f"unexpected forest validation status {status}")

    node_count = int(is_leaf.size)
    leaf_count = int(np.count_nonzero(is_leaf))
    return root_count, node_count, leaf_count, int(max_level)


def _fill_validated_forest(
    rank_to_coord: np.ndarray,
    is_leaf: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    parent_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    root_node_ids: np.ndarray,
) -> None:
    fill_refined_forest_unchecked(
        rank_to_coord,
        is_leaf.view(np.uint8),
        node_levels,
        node_coords,
        parent_node_ids,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        root_node_ids,
    )


def fill_refined_forest(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    is_leaf: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    parent_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    root_node_ids: np.ndarray,
) -> int:
    """Validate and fill caller-owned flat refined-forest artifacts."""
    root_count, node_count, leaf_count, max_level = _validate_forest_inputs(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        is_leaf,
    )
    outputs = (
        _require_index_output("node_levels", node_levels, (node_count,)),
        _require_index_output("node_coords", node_coords, (node_count, 3)),
        _require_index_output(
            "parent_node_ids", parent_node_ids, (node_count,)
        ),
        _require_index_output(
            "child_node_ids", child_node_ids, (node_count, 8)
        ),
        _require_index_output("node_leaf_ids", node_leaf_ids, (node_count,)),
        _require_index_output("leaf_node_ids", leaf_node_ids, (leaf_count,)),
        _require_index_output("root_node_ids", root_node_ids, (root_count,)),
    )
    inputs = (root_shape, coord_to_rank, rank_to_coord, is_leaf)
    for index, output in enumerate(outputs):
        if any(np.shares_memory(output, source) for source in inputs):
            raise ValueError("forest outputs must not overlap inputs")
        if any(np.shares_memory(output, prior) for prior in outputs[:index]):
            raise ValueError("forest outputs must not overlap each other")

    _fill_validated_forest(
        rank_to_coord,
        is_leaf,
        *outputs,
    )
    return max_level


def refined_forest(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    is_leaf: np.ndarray,
) -> RefinedForest:
    """Allocate and return explicit refined-forest arrays."""
    root_count, node_count, leaf_count, max_level = _validate_forest_inputs(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        is_leaf,
    )
    outputs = (
        np.empty(node_count, dtype=np.int64),
        np.empty((node_count, 3), dtype=np.int64),
        np.empty(node_count, dtype=np.int64),
        np.empty((node_count, 8), dtype=np.int64),
        np.empty(node_count, dtype=np.int64),
        np.empty(leaf_count, dtype=np.int64),
        np.empty(root_count, dtype=np.int64),
    )
    _fill_validated_forest(rank_to_coord, is_leaf, *outputs)
    return RefinedForest(*outputs, max_level)
