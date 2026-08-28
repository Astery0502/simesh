"""Validated HCL-001 complete level-1 one-block halo closure."""

from __future__ import annotations

import numpy as np

from ._chunking import (
    minimum_level1_halo_closed_slots_unchecked,
    plan_level1_halo_closed_prefix_unchecked,
)
from .face_closure import _require_face_table
from .primary import _require_primary_ids
from .workspace import _require_nonnegative_integer


def minimum_level1_halo_closed_slots(face_neighbor_ids: np.ndarray) -> int:
    """Return capacity sufficient for every level-1 one-block halo closure."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    return int(minimum_level1_halo_closed_slots_unchecked(face_neighbor_ids))


def plan_level1_halo_closed_prefix(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    selected_ids: np.ndarray,
) -> tuple[int, int]:
    """Fill maximal dense primaries plus complete clipped 3x3x3 support."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    selected_ids = _require_primary_ids("selected_ids", selected_ids)
    first_primary_id = _require_nonnegative_integer(
        "first_primary_id", first_primary_id
    )
    if first_primary_id > face_neighbor_ids.shape[0]:
        raise ValueError("first_primary_id exceeds block count")
    if np.shares_memory(face_neighbor_ids, selected_ids):
        raise ValueError("selected_ids must not overlap face_neighbor_ids")
    if (
        first_primary_id < face_neighbor_ids.shape[0]
        and selected_ids.shape[0] == 0
    ):
        raise ValueError("selected capacity must be positive before the end")
    return tuple(
        int(value)
        for value in plan_level1_halo_closed_prefix_unchecked(
            first_primary_id,
            face_neighbor_ids,
            selected_ids,
        )
    )
