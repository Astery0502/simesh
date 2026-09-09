"""Validated FCL-001 deterministic direct-face support closure."""

from __future__ import annotations

import numpy as np

from ._chunking import (
    minimum_face_closed_slots_unchecked,
    plan_direct_face_closed_prefix_unchecked,
)
from .foundation import INDEX_DTYPE
from .primary import _require_primary_ids
from .workspace import _require_nonnegative_integer


def _require_face_table(face_neighbor_ids: np.ndarray) -> np.ndarray:
    if not isinstance(face_neighbor_ids, np.ndarray):
        raise TypeError("face_neighbor_ids must be a NumPy array")
    if face_neighbor_ids.dtype != INDEX_DTYPE:
        raise TypeError("face_neighbor_ids must have dtype int64")
    if face_neighbor_ids.ndim != 2 or face_neighbor_ids.shape[1] != 6:
        raise ValueError(
            "face_neighbor_ids must have shape (block_count, 6), "
            f"got {face_neighbor_ids.shape}"
        )
    if not face_neighbor_ids.flags.c_contiguous:
        raise ValueError("face_neighbor_ids must be C-contiguous")
    return face_neighbor_ids


def minimum_direct_face_closed_slots(face_neighbor_ids: np.ndarray) -> int:
    """Return capacity sufficient for every single direct-face closure."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    return int(minimum_face_closed_slots_unchecked(face_neighbor_ids))


def plan_direct_face_closed_prefix(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    selected_ids: np.ndarray,
) -> tuple[int, int]:
    """Fill a maximal dense primary prefix plus unique direct-face support."""
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
        for value in plan_direct_face_closed_prefix_unchecked(
            first_primary_id,
            face_neighbor_ids,
            selected_ids,
        )
    )
