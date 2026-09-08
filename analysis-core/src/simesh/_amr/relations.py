"""Validated REL-001 balanced refined direction relation records."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from simesh._kernels.primitives._contacts import invalid_contact_direction_unchecked
from simesh._kernels.primitives._relations import fill_balanced_refined_relations_unchecked
from simesh._kernels.primitives._storage import validate_indices_unchecked
from simesh._amr.contacts import _require_contact_forest_inputs, _require_index_vector


RELATION_PHYSICAL = 1
RELATION_COARSER = 2
RELATION_SAME = 3
RELATION_FINER = 4


def _require_direction_matrix(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("directions must be a NumPy array")
    if value.dtype != np.dtype(np.int64):
        raise TypeError("directions must have dtype int64")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError(f"directions must have shape (D, 3), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError("directions must be C-contiguous")
    return value


def _require_relation_output(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def fill_balanced_refined_relations(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_ids: np.ndarray,
    directions: np.ndarray,
    relation_kinds: np.ndarray,
    physical_masks: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> None:
    """Fill balanced relation facts for selected leaf-by-direction pairs."""
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
    leaf_ids = _require_index_vector("leaf_ids", leaf_ids)
    directions = _require_direction_matrix(directions)

    shape = (leaf_ids.shape[0], directions.shape[0])
    relation_kinds = _require_relation_output(
        "relation_kinds", relation_kinds, shape, np.dtype(np.uint8)
    )
    physical_masks = _require_relation_output(
        "physical_masks", physical_masks, shape, np.dtype(np.uint8)
    )
    source_counts = _require_relation_output(
        "source_counts", source_counts, shape, np.dtype(np.uint8)
    )
    source_leaf_ids = _require_relation_output(
        "source_leaf_ids",
        source_leaf_ids,
        (*shape, 4),
        np.dtype(np.int64),
    )

    inputs = (
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        leaf_ids,
        directions,
    )
    outputs = (
        relation_kinds,
        physical_masks,
        source_counts,
        source_leaf_ids,
    )
    if any(
        np.shares_memory(output, source)
        for output in outputs
        for source in inputs
    ) or any(
        np.shares_memory(left, right)
        for left, right in combinations(outputs, 2)
    ):
        raise ValueError("relation outputs must not overlap inputs or each other")

    invalid_leaf = int(
        validate_indices_unchecked(leaf_ids, leaf_node_ids.shape[0])
    )
    if invalid_leaf >= 0:
        raise ValueError(f"leaf_ids entry {invalid_leaf} is out of range")
    invalid_direction = int(invalid_contact_direction_unchecked(directions))
    if invalid_direction >= 0:
        raise ValueError(
            f"directions row {invalid_direction} must be noncenter values in [-1, 1]"
        )

    fill_balanced_refined_relations_unchecked(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        leaf_ids,
        directions,
        relation_kinds,
        physical_masks,
        source_counts,
        source_leaf_ids,
    )


def balanced_refined_relations(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_ids: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate and return selected balanced refined relation records."""
    leaf_ids = _require_index_vector("leaf_ids", leaf_ids)
    directions = _require_direction_matrix(directions)
    shape = (leaf_ids.shape[0], directions.shape[0])
    relation_kinds = np.empty(shape, dtype=np.uint8)
    physical_masks = np.empty(shape, dtype=np.uint8)
    source_counts = np.empty(shape, dtype=np.uint8)
    source_leaf_ids = np.empty((*shape, 4), dtype=np.int64)
    fill_balanced_refined_relations(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        leaf_ids,
        directions,
        relation_kinds,
        physical_masks,
        source_counts,
        source_leaf_ids,
    )
    return relation_kinds, physical_masks, source_counts, source_leaf_ids
