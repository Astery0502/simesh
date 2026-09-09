"""Validated grouped refined point-sampling boundaries."""

from __future__ import annotations

import numpy as np

from ._refined_sampling import (
    sample_refined_trilinear_point_groups_unchecked,
    sample_refined_zero_order_point_groups_unchecked,
    validate_refined_point_groups_unchecked,
)
from .forest_conformance import _require_index_array
from .foundation import (
    INDEX_DTYPE,
    PAYLOAD_DTYPE,
    _require_index_triplet,
    _require_payload,
)
from .geometry import _require_float_triplet, _require_index_vector
from .morton import _root_volume


_NORMAL_MIN = np.finfo(np.float64).tiny


def _require_points(points: np.ndarray) -> np.ndarray:
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a NumPy array")
    if points.dtype != PAYLOAD_DTYPE:
        raise TypeError("points must have dtype float64")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (point, 3)")
    if not points.flags.c_contiguous:
        raise ValueError("points must be C-contiguous")
    return points


def _require_point_values(
    point_values: np.ndarray,
    point_count: int,
    field_count: int,
) -> np.ndarray:
    if not isinstance(point_values, np.ndarray):
        raise TypeError("point_values must be a NumPy array")
    if point_values.dtype != PAYLOAD_DTYPE:
        raise TypeError("point_values must have dtype float64")
    expected = (point_count, field_count)
    if point_values.shape != expected:
        raise ValueError(
            f"point_values must have shape {expected}, got {point_values.shape}"
        )
    if not point_values.flags.c_contiguous:
        raise ValueError("point_values must be C-contiguous")
    if not point_values.flags.writeable:
        raise ValueError("point_values must be writable")
    return point_values


def _base_spacing(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
) -> np.ndarray:
    spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        spacing[axis] = extent / float(domain_cell_counts[axis])
        if not np.isfinite(spacing[axis]) or spacing[axis] < _NORMAL_MIN:
            raise ValueError("base cell spacing must be finite, positive, and normal")
    return spacing


def _preflight_refined_point_groups(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    slot_leaf_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    slot_point_offsets: np.ndarray,
    *,
    trilinear: bool,
) -> tuple[np.ndarray, ...]:
    """Validate a complete group map and return normalized arrays.

    The tuple order is ``domain_lower, domain_upper, domain_cell_counts,
    block_cell_counts, node_levels, node_coords, leaf_node_ids, slot_leaf_ids,
    points, point_indices, slot_point_offsets, base_spacing``.  RPS-001 uses
    this boundary once before its first nonempty reader call.
    """
    if type(trilinear) is not bool:
        raise TypeError("trilinear must be a bool")
    domain_lower = _require_float_triplet("domain_lower", domain_lower)
    domain_upper = _require_float_triplet("domain_upper", domain_upper)
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
    if np.any(domain_cell_counts <= 0) or np.any(block_cell_counts <= 0):
        raise ValueError("cell counts must be positive")
    if np.any(domain_cell_counts % block_cell_counts != 0):
        raise ValueError(
            "domain_cell_counts must be divisible by block_cell_counts"
        )
    root_shape = np.ascontiguousarray(
        domain_cell_counts // block_cell_counts,
        dtype=np.int64,
    )
    _root_volume(root_shape)

    if not isinstance(node_levels, np.ndarray):
        raise TypeError("node_levels must be a NumPy array")
    if node_levels.dtype != INDEX_DTYPE:
        raise TypeError("node_levels must have dtype int64")
    if node_levels.ndim != 1 or not node_levels.flags.c_contiguous:
        raise ValueError("node_levels must be a C-contiguous vector")
    if node_levels.shape[0] == 0:
        raise ValueError("node_levels must contain at least one node")
    node_coords = _require_index_array(
        "node_coords", node_coords, (node_levels.shape[0], 3)
    )
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    if leaf_node_ids.shape[0] == 0:
        raise ValueError("leaf_node_ids must contain at least one leaf")
    slot_leaf_ids = _require_index_vector("slot_leaf_ids", slot_leaf_ids)

    points = _require_points(points)
    point_indices = _require_index_vector("point_indices", point_indices)
    slot_point_offsets = _require_index_vector(
        "slot_point_offsets", slot_point_offsets
    )
    expected_offsets = slot_leaf_ids.shape[0] + 1
    if slot_point_offsets.shape != (expected_offsets,):
        raise ValueError(
            "slot_point_offsets must have one more entry than slot_leaf_ids"
        )
    if int(slot_point_offsets[0]) != 0:
        raise ValueError("slot_point_offsets must start at zero")
    previous = 0
    for offset_value in slot_point_offsets[1:]:
        offset = int(offset_value)
        if offset < previous:
            raise ValueError("slot_point_offsets must be nondecreasing")
        previous = offset
    if previous != point_indices.shape[0]:
        raise ValueError("slot_point_offsets must end at len(point_indices)")

    base_spacing = _base_spacing(
        domain_lower,
        domain_upper,
        domain_cell_counts,
    )
    status, bad_slot, bad_point = validate_refined_point_groups_unchecked(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
        base_spacing,
        trilinear,
    )
    if status == 1:
        raise ValueError(f"invalid selected refined geometry at slot {bad_slot}")
    if status == 2:
        raise OverflowError(
            f"selected refined cell indices overflow int64 at slot {bad_slot}"
        )
    if status == 3:
        raise ValueError(
            f"point {bad_point} is invalid or not owned by slot {bad_slot}"
        )
    if status == 4:
        raise ValueError(
            f"invalid trilinear stencil for point {bad_point} at slot {bad_slot}"
        )
    if status != 0:
        raise RuntimeError(f"unexpected refined sampling status {status}")
    return (
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
        base_spacing,
    )


def _normalize_refined_point_sampling(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    payload_valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    slot_leaf_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    slot_point_offsets: np.ndarray,
    point_values: np.ndarray,
    *,
    trilinear: bool,
) -> tuple[np.ndarray, ...]:
    payload = _require_payload("payload", payload, writable=False)
    payload_valid_lower = _require_index_triplet(
        "payload_valid_lower", payload_valid_lower
    )
    payload_valid_upper = _require_index_triplet(
        "payload_valid_upper", payload_valid_upper
    )
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)

    normalized = _preflight_refined_point_groups(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
        trilinear=trilinear,
    )
    (
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
        base_spacing,
    ) = normalized

    if payload.shape[0] != slot_leaf_ids.shape[0]:
        raise ValueError("payload slot axis must match slot_leaf_ids")
    point_values = _require_point_values(
        point_values,
        points.shape[0],
        payload.shape[1],
    )
    spatial_shape = np.asarray(payload.shape[2:], dtype=np.int64)
    if np.any(payload_valid_lower < 0) or np.any(
        payload_valid_lower >= payload_valid_upper
    ):
        raise ValueError("payload valid region must be nonempty and nonnegative")
    if np.any(payload_valid_upper > spatial_shape):
        raise ValueError("payload valid region exceeds payload spatial shape")
    if np.any(interior_lower < 0) or np.any(interior_lower >= interior_upper):
        raise ValueError("interior must be nonempty and nonnegative")
    if np.any(interior_upper > spatial_shape):
        raise ValueError("interior exceeds payload spatial shape")
    if not np.array_equal(interior_upper - interior_lower, block_cell_counts):
        raise ValueError("interior extent must equal block_cell_counts")
    if np.any(payload_valid_lower > interior_lower) or np.any(
        payload_valid_upper < interior_upper
    ):
        raise ValueError("payload valid region must contain the interior")
    if trilinear:
        if np.any(interior_lower < 1) or np.any(interior_upper >= spatial_shape):
            raise ValueError("trilinear sampling requires one allocated halo layer")
        if np.any(payload_valid_lower > interior_lower - 1) or np.any(
            payload_valid_upper < interior_upper + 1
        ):
            raise ValueError("trilinear sampling requires one valid halo layer")

    readonly_inputs = (
        payload,
        payload_valid_lower,
        payload_valid_upper,
        interior_lower,
        interior_upper,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
    )
    if any(np.shares_memory(point_values, value) for value in readonly_inputs):
        raise ValueError("point_values must not overlap sampling inputs")

    return (
        payload,
        payload_valid_lower,
        payload_valid_upper,
        interior_lower,
        interior_upper,
        *normalized,
        point_values,
    )


def sample_refined_zero_order_point_groups(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    payload_valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    slot_leaf_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    slot_point_offsets: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Copy exact owner-level cell values for explicit point groups."""
    normalized = _normalize_refined_point_sampling(
        payload,
        payload_valid_lower,
        payload_valid_upper,
        interior_lower,
        interior_upper,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        points,
        point_indices,
        slot_point_offsets,
        point_values,
        trilinear=False,
    )
    (
        payload,
        _payload_valid_lower,
        _payload_valid_upper,
        interior_lower,
        _interior_upper,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
        base_spacing,
        point_values,
    ) = normalized
    sample_refined_zero_order_point_groups_unchecked(
        payload,
        interior_lower,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        base_spacing,
        points,
        point_indices,
        slot_point_offsets,
        point_values,
    )


def sample_refined_trilinear_point_groups(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    payload_valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    slot_leaf_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    point_indices: np.ndarray,
    slot_point_offsets: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Evaluate fixed-tree trilinear values for completed refined groups."""
    normalized = _normalize_refined_point_sampling(
        payload,
        payload_valid_lower,
        payload_valid_upper,
        interior_lower,
        interior_upper,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        points,
        point_indices,
        slot_point_offsets,
        point_values,
        trilinear=True,
    )
    (
        payload,
        _payload_valid_lower,
        _payload_valid_upper,
        interior_lower,
        _interior_upper,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        slot_leaf_ids,
        points,
        point_indices,
        slot_point_offsets,
        base_spacing,
        point_values,
    ) = normalized
    sample_refined_trilinear_point_groups_unchecked(
        payload,
        interior_lower,
        slot_leaf_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        base_spacing,
        points,
        point_indices,
        slot_point_offsets,
        point_values,
    )
