"""Validated GEO-002 selected Cartesian 3D refined leaf geometry."""

from __future__ import annotations

import numpy as np

from ._geometry import (
    fill_refined_leaf_geometry_unchecked,
    validate_selected_refined_geometry_unchecked,
)
from .forest_conformance import _require_index_array
from .foundation import INDEX_DTYPE, PAYLOAD_DTYPE, _require_index_triplet
from .geometry import _require_float_triplet, _require_index_vector
from .morton import _root_volume


def _require_refined_geometry_outputs(
    leaf_bounds: np.ndarray,
    leaf_cell_spacing: np.ndarray,
    slot_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(leaf_bounds, np.ndarray):
        raise TypeError("leaf_bounds must be a NumPy array")
    if leaf_bounds.dtype != PAYLOAD_DTYPE:
        raise TypeError("leaf_bounds must have dtype float64")
    if leaf_bounds.shape != (slot_count, 2, 3):
        raise ValueError(
            f"leaf_bounds must have shape {(slot_count, 2, 3)}, "
            f"got {leaf_bounds.shape}"
        )
    if not leaf_bounds.flags.c_contiguous:
        raise ValueError("leaf_bounds must be C-contiguous")
    if not leaf_bounds.flags.writeable:
        raise ValueError("leaf_bounds must be writable")

    if not isinstance(leaf_cell_spacing, np.ndarray):
        raise TypeError("leaf_cell_spacing must be a NumPy array")
    if leaf_cell_spacing.dtype != PAYLOAD_DTYPE:
        raise TypeError("leaf_cell_spacing must have dtype float64")
    if leaf_cell_spacing.shape != (slot_count, 3):
        raise ValueError(
            f"leaf_cell_spacing must have shape {(slot_count, 3)}, "
            f"got {leaf_cell_spacing.shape}"
        )
    if not leaf_cell_spacing.flags.c_contiguous:
        raise ValueError("leaf_cell_spacing must be C-contiguous")
    if not leaf_cell_spacing.flags.writeable:
        raise ValueError("leaf_cell_spacing must be writable")
    return leaf_bounds, leaf_cell_spacing


def fill_refined_leaf_geometry(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_ids: np.ndarray,
    leaf_bounds: np.ndarray,
    leaf_cell_spacing: np.ndarray,
) -> None:
    """Fill bounds and per-leaf spacing for explicit canonical leaf IDs."""
    domain_lower = _require_float_triplet("domain_lower", domain_lower)
    domain_upper = _require_float_triplet("domain_upper", domain_upper)
    root_shape = _require_index_triplet("root_shape", root_shape)
    domain_cell_counts = _require_index_triplet(
        "domain_cell_counts", domain_cell_counts
    )
    block_cell_counts = _require_index_triplet(
        "block_cell_counts", block_cell_counts
    )
    if not np.all(np.isfinite(domain_lower)) or not np.all(
        np.isfinite(domain_upper)
    ):
        raise ValueError("domain bounds must be finite")
    if np.any(domain_upper <= domain_lower):
        raise ValueError("domain_upper must be greater than domain_lower")
    if (
        np.any(root_shape <= 0)
        or np.any(domain_cell_counts <= 0)
        or np.any(block_cell_counts <= 0)
    ):
        raise ValueError("root shape and cell counts must be positive")
    _root_volume(root_shape)
    int64_max = np.iinfo(np.int64).max
    for axis in range(3):
        expected = int(root_shape[axis]) * int(block_cell_counts[axis])
        if expected > int64_max:
            raise OverflowError("root block-cell product does not fit in int64")
        if int(domain_cell_counts[axis]) != expected:
            raise ValueError(
                "domain_cell_counts must equal root_shape * block_cell_counts"
            )

    if not isinstance(node_levels, np.ndarray):
        raise TypeError("node_levels must be a NumPy array")
    if node_levels.dtype != INDEX_DTYPE:
        raise TypeError("node_levels must have dtype int64")
    if node_levels.ndim != 1 or not node_levels.flags.c_contiguous:
        raise ValueError("node_levels must be a C-contiguous vector")
    node_count = node_levels.shape[0]
    if node_count == 0:
        raise ValueError("node_levels must contain at least one node")
    node_coords = _require_index_array(
        "node_coords", node_coords, (node_count, 3)
    )

    if not isinstance(leaf_node_ids, np.ndarray):
        raise TypeError("leaf_node_ids must be a NumPy array")
    if leaf_node_ids.dtype != INDEX_DTYPE:
        raise TypeError("leaf_node_ids must have dtype int64")
    if leaf_node_ids.ndim != 1 or not leaf_node_ids.flags.c_contiguous:
        raise ValueError("leaf_node_ids must be a C-contiguous vector")
    if leaf_node_ids.shape[0] == 0:
        raise ValueError("leaf_node_ids must contain at least one leaf")
    leaf_ids = _require_index_vector("leaf_ids", leaf_ids)
    leaf_bounds, leaf_cell_spacing = _require_refined_geometry_outputs(
        leaf_bounds,
        leaf_cell_spacing,
        leaf_ids.shape[0],
    )

    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        base_spacing[axis] = extent / float(domain_cell_counts[axis])
    if (
        not np.all(np.isfinite(base_spacing))
        or np.any(base_spacing < np.finfo(np.float64).tiny)
    ):
        raise ValueError("base cell spacing must be finite, positive, and normal")

    outputs = (leaf_bounds, leaf_cell_spacing)
    inputs = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        leaf_ids,
    )
    if np.shares_memory(leaf_bounds, leaf_cell_spacing) or any(
        np.shares_memory(output, source)
        for output in outputs
        for source in inputs
    ):
        raise ValueError("geometry outputs must not overlap inputs or each other")

    status, bad_slot = validate_selected_refined_geometry_unchecked(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        leaf_ids,
        base_spacing,
    )
    if status == 1:
        raise ValueError(f"invalid selected refined geometry at slot {bad_slot}")
    if status == 2:
        raise OverflowError(
            f"selected refined cell indices overflow int64 at slot {bad_slot}"
        )
    if status != 0:
        raise RuntimeError(f"unexpected refined geometry status {status}")

    fill_refined_leaf_geometry_unchecked(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        leaf_ids,
        base_spacing,
        leaf_bounds,
        leaf_cell_spacing,
    )


def refined_leaf_geometry(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate and return selected refined bounds and cell spacing."""
    leaf_ids = _require_index_vector("leaf_ids", leaf_ids)
    leaf_bounds = np.empty((leaf_ids.shape[0], 2, 3), dtype=np.float64)
    leaf_cell_spacing = np.empty((leaf_ids.shape[0], 3), dtype=np.float64)
    fill_refined_leaf_geometry(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        leaf_ids,
        leaf_bounds,
        leaf_cell_spacing,
    )
    return leaf_bounds, leaf_cell_spacing
