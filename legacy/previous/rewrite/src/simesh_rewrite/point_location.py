"""Validated LOC-001 exact Cartesian 3D refined point ownership."""

from __future__ import annotations

import math

import numpy as np

from ._point_location import (
    fill_refined_point_leaf_ids_unchecked,
    validate_refined_points_finite_unchecked,
)
from .forest_conformance import _require_index_array
from .foundation import INDEX_DTYPE, PAYLOAD_DTYPE, _require_index_triplet
from .geometry import _require_float_triplet, _require_index_vector
from .morton import _root_volume


_INDEX_MAX = int(np.iinfo(np.int64).max)
_NORMAL_MIN = float(np.finfo(np.float64).tiny)


def _require_points(points: np.ndarray) -> np.ndarray:
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a NumPy array")
    if points.dtype != PAYLOAD_DTYPE:
        raise TypeError("points must have dtype float64")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (point, 3), got {points.shape}")
    if not points.flags.c_contiguous:
        raise ValueError("points must be C-contiguous")
    return points


def _require_point_leaf_ids(
    point_leaf_ids: np.ndarray,
    point_count: int,
) -> np.ndarray:
    if not isinstance(point_leaf_ids, np.ndarray):
        raise TypeError("point_leaf_ids must be a NumPy array")
    if point_leaf_ids.dtype != INDEX_DTYPE:
        raise TypeError("point_leaf_ids must have dtype int64")
    if point_leaf_ids.shape != (point_count,):
        raise ValueError(
            f"point_leaf_ids must have shape {(point_count,)}, "
            f"got {point_leaf_ids.shape}"
        )
    if not point_leaf_ids.flags.c_contiguous:
        raise ValueError("point_leaf_ids must be C-contiguous")
    if not point_leaf_ids.flags.writeable:
        raise ValueError("point_leaf_ids must be writable")
    return point_leaf_ids


def _validated_locator_metadata(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
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

    root_count = _root_volume(root_shape)
    for axis in range(3):
        root_extent = int(root_shape[axis])
        block_extent = int(block_cell_counts[axis])
        if root_extent > _INDEX_MAX // block_extent:
            raise OverflowError("root block-cell product does not fit in int64")
        if int(domain_cell_counts[axis]) != root_extent * block_extent:
            raise ValueError(
                "domain_cell_counts must equal root_shape * block_cell_counts"
            )

    if type(max_level) is not int:
        raise TypeError("max_level must be an exact Python int")
    if max_level <= 0:
        raise ValueError("max_level must be positive")
    shift = max_level - 1
    if shift >= 63:
        raise OverflowError("deepest effective cell counts do not fit in int64")
    scale = 1 << shift
    for axis in range(3):
        if int(domain_cell_counts[axis]) > _INDEX_MAX // scale:
            raise OverflowError(
                "deepest effective cell counts do not fit in int64"
            )

    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        base_spacing[axis] = extent / float(domain_cell_counts[axis])
        deepest_spacing = math.ldexp(float(base_spacing[axis]), -shift)
        if (
            not math.isfinite(float(base_spacing[axis]))
            or float(base_spacing[axis]) < _NORMAL_MIN
            or not math.isfinite(deepest_spacing)
            or deepest_spacing < _NORMAL_MIN
        ):
            raise ValueError(
                "base and deepest cell spacing must be finite, positive, "
                "and normal"
            )

    coord_to_rank = _require_index_array(
        "coord_to_rank",
        coord_to_rank,
        tuple(int(value) for value in root_shape),
    )
    root_node_ids = _require_index_array(
        "root_node_ids", root_node_ids, (root_count,)
    )
    node_leaf_ids = _require_index_vector("node_leaf_ids", node_leaf_ids)
    node_count = node_leaf_ids.shape[0]
    if node_count == 0:
        raise ValueError("node_leaf_ids must contain at least one node")
    child_node_ids = _require_index_array(
        "child_node_ids", child_node_ids, (node_count, 8)
    )

    metadata = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        root_node_ids,
        child_node_ids,
        node_leaf_ids,
    )
    return metadata, base_spacing


def fill_refined_point_leaf_ids(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    points: np.ndarray,
    point_leaf_ids: np.ndarray,
) -> None:
    """Fill the canonical refined leaf owner of every finite query point."""
    metadata, base_spacing = _validated_locator_metadata(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        max_level,
        coord_to_rank,
        root_node_ids,
        child_node_ids,
        node_leaf_ids,
    )
    points = _require_points(points)
    point_leaf_ids = _require_point_leaf_ids(point_leaf_ids, points.shape[0])

    readonly_inputs = (*metadata, points)
    if any(
        np.shares_memory(point_leaf_ids, source) for source in readonly_inputs
    ):
        raise ValueError("point_leaf_ids must not overlap locator inputs")

    bad_point, bad_axis = validate_refined_points_finite_unchecked(points)
    if bad_point >= 0:
        raise ValueError(
            f"points must be finite; point {bad_point}, axis {bad_axis} is not"
        )

    fill_refined_point_leaf_ids_unchecked(
        metadata[0],
        metadata[1],
        metadata[3],
        metadata[4],
        metadata[5],
        metadata[6],
        metadata[7],
        metadata[8],
        base_spacing,
        points,
        point_leaf_ids,
    )


def refined_point_leaf_ids(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Allocate and return canonical refined leaf owners."""
    points = _require_points(points)
    point_leaf_ids = np.empty(points.shape[0], dtype=np.int64)
    fill_refined_point_leaf_ids(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        max_level,
        coord_to_rank,
        root_node_ids,
        child_node_ids,
        node_leaf_ids,
        points,
        point_leaf_ids,
    )
    return point_leaf_ids
