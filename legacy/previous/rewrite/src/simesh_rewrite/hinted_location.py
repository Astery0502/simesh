"""Validated HLO-001 last-owner hinted refined point ownership."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._point_location import (
    fill_refined_point_leaf_ids_with_hints_unchecked,
    validate_refined_point_hints_unchecked,
)
from .forest_conformance import _require_index_array
from .geometry import _require_index_vector
from .point_location import (
    _require_point_leaf_ids,
    _require_points,
    _validated_locator_metadata,
)


class HintedLocationStats(NamedTuple):
    point_count: int
    interior_point_count: int
    exterior_point_count: int
    hint_candidate_count: int
    hint_hit_count: int
    hierarchy_fallback_count: int


def fill_refined_point_leaf_ids_with_hints(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    points: np.ndarray,
    hint_leaf_ids: np.ndarray,
    point_leaf_ids: np.ndarray,
) -> HintedLocationStats:
    """Resolve exact owners by testing each valid hint before LOC descent."""
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
    node_levels = _require_index_vector("node_levels", node_levels)
    node_count = metadata[8].shape[0]
    if node_levels.shape != (node_count,):
        raise ValueError(
            f"node_levels must have shape {(node_count,)}, got {node_levels.shape}"
        )
    node_coords = _require_index_array(
        "node_coords", node_coords, (node_count, 3)
    )
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    if leaf_node_ids.shape[0] == 0:
        raise ValueError("leaf_node_ids must contain at least one leaf")
    points = _require_points(points)
    hint_leaf_ids = _require_index_vector("hint_leaf_ids", hint_leaf_ids)
    if hint_leaf_ids.shape != (points.shape[0],):
        raise ValueError(
            "hint_leaf_ids must have one entry per point; "
            f"got {hint_leaf_ids.shape} for {points.shape[0]} points"
        )
    point_leaf_ids = _require_point_leaf_ids(
        point_leaf_ids, int(points.shape[0])
    )

    readonly_inputs = (
        *metadata,
        node_levels,
        node_coords,
        leaf_node_ids,
        points,
        hint_leaf_ids,
    )
    if any(
        np.shares_memory(point_leaf_ids, source) for source in readonly_inputs
    ):
        raise ValueError("point_leaf_ids must not overlap hinted locator inputs")

    (
        status,
        bad_point,
        bad_axis,
        interior_point_count,
        exterior_point_count,
        hint_candidate_count,
    ) = validate_refined_point_hints_unchecked(
        metadata[0],
        metadata[1],
        metadata[2],
        metadata[3],
        metadata[4],
        max_level,
        node_levels,
        node_coords,
        metadata[8],
        leaf_node_ids,
        base_spacing,
        points,
        hint_leaf_ids,
    )
    if status == 1:
        raise ValueError(
            "hint_leaf_ids entries must be -1 or valid leaf IDs; "
            f"first invalid entry is {bad_point}"
        )
    if status == 2:
        raise ValueError(
            "invalid hinted refined geometry at "
            f"point {bad_point}, axis {bad_axis}"
        )
    if status == 3:
        raise OverflowError(
            "hinted refined geometry overflows int64 at "
            f"point {bad_point}, axis {bad_axis}"
        )
    if status == 4:
        raise ValueError(
            "hinted leaf spacing and faces must be finite, normal, and ordered; "
            f"first invalid point is {bad_point}, axis {bad_axis}"
        )
    if status == 5:
        raise ValueError(
            f"points must be finite; point {bad_point}, axis {bad_axis} is not"
        )
    if status != 0:
        raise RuntimeError(f"unexpected hinted location status {status}")

    hint_hit_count = int(
        fill_refined_point_leaf_ids_with_hints_unchecked(
            metadata[0],
            metadata[1],
            metadata[3],
            metadata[4],
            metadata[5],
            metadata[6],
            node_levels,
            node_coords,
            metadata[7],
            metadata[8],
            leaf_node_ids,
            base_spacing,
            points,
            hint_leaf_ids,
            point_leaf_ids,
        )
    )
    hierarchy_fallback_count = int(interior_point_count) - hint_hit_count
    return HintedLocationStats(
        int(points.shape[0]),
        int(interior_point_count),
        int(exterior_point_count),
        int(hint_candidate_count),
        hint_hit_count,
        hierarchy_fallback_count,
    )
