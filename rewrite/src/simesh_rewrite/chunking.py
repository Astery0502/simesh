"""STO-002 planning with preserved WSP-001 compatibility wrappers."""

from __future__ import annotations

import numpy as np

from ._chunking import (
    minimum_face_closed_slots_unchecked,
    minimum_halo_closed_slots_unchecked,
    plan_level1_chunk_unchecked,
    plan_level1_halo_chunk_unchecked,
)
from .foundation import INDEX_DTYPE
from .workspace import (
    _require_nonnegative_integer,
    workspace_nbytes as _workspace_nbytes,
    workspace_slot_capacity as _workspace_slot_capacity,
)


def workspace_nbytes(
    slot_capacity: int,
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    """Compatibility wrapper for WSP-001 canonical accounting."""
    return _workspace_nbytes(slot_capacity, field_count, workspace_shape)


def workspace_slot_capacity(
    budget_bytes: int,
    block_count: int,
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    """Compatibility wrapper for WSP-001 inverse accounting."""
    return _workspace_slot_capacity(
        budget_bytes,
        block_count,
        field_count,
        workspace_shape,
    )


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


def _require_chunk_ids(chunk_block_ids: np.ndarray) -> np.ndarray:
    if not isinstance(chunk_block_ids, np.ndarray):
        raise TypeError("chunk_block_ids must be a NumPy array")
    if chunk_block_ids.dtype != INDEX_DTYPE:
        raise TypeError("chunk_block_ids must have dtype int64")
    if chunk_block_ids.ndim != 1 or not chunk_block_ids.flags.c_contiguous:
        raise ValueError("chunk_block_ids must be a C-contiguous vector")
    if not chunk_block_ids.flags.writeable:
        raise ValueError("chunk_block_ids must be writable")
    return chunk_block_ids


def minimum_face_closed_slots(face_neighbor_ids: np.ndarray) -> int:
    """Return capacity sufficient for any single direct-face closure."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    return int(minimum_face_closed_slots_unchecked(face_neighbor_ids))


def minimum_halo_closed_slots(face_neighbor_ids: np.ndarray) -> int:
    """Return capacity sufficient for any one-block 3x3x3 halo closure."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    return int(minimum_halo_closed_slots_unchecked(face_neighbor_ids))


def plan_level1_chunk(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    include_face_closure: bool,
    chunk_block_ids: np.ndarray,
) -> tuple[int, int]:
    """Fill a maximal contiguous primary chunk and optional direct-face support."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    chunk_block_ids = _require_chunk_ids(chunk_block_ids)
    first_primary_id = _require_nonnegative_integer(
        "first_primary_id", first_primary_id
    )
    if first_primary_id > face_neighbor_ids.shape[0]:
        raise ValueError("first_primary_id exceeds block count")
    if type(include_face_closure) is not bool:
        raise TypeError("include_face_closure must be a bool")
    if np.shares_memory(face_neighbor_ids, chunk_block_ids):
        raise ValueError("chunk_block_ids must not overlap face_neighbor_ids")
    if (
        first_primary_id < face_neighbor_ids.shape[0]
        and chunk_block_ids.shape[0] == 0
    ):
        raise ValueError("chunk capacity must be positive before the end")

    return tuple(
        int(value)
        for value in plan_level1_chunk_unchecked(
            first_primary_id,
            face_neighbor_ids,
            include_face_closure,
            chunk_block_ids,
        )
    )


def plan_level1_halo_chunk(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    chunk_block_ids: np.ndarray,
) -> tuple[int, int]:
    """Plan a maximal primary prefix with complete one-block halo closure."""
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    chunk_block_ids = _require_chunk_ids(chunk_block_ids)
    first_primary_id = _require_nonnegative_integer(
        "first_primary_id", first_primary_id
    )
    if first_primary_id > face_neighbor_ids.shape[0]:
        raise ValueError("first_primary_id exceeds block count")
    if np.shares_memory(face_neighbor_ids, chunk_block_ids):
        raise ValueError("chunk_block_ids must not overlap face_neighbor_ids")
    if (
        first_primary_id < face_neighbor_ids.shape[0]
        and chunk_block_ids.shape[0] == 0
    ):
        raise ValueError("chunk capacity must be positive before the end")
    return tuple(
        int(value)
        for value in plan_level1_halo_chunk_unchecked(
            first_primary_id,
            face_neighbor_ids,
            chunk_block_ids,
        )
    )
