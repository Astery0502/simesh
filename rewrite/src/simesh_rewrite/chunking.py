"""STO-002 planning with preserved WSP-001 compatibility wrappers."""

from __future__ import annotations

import numpy as np

from ._chunking import (
    minimum_halo_closed_slots_unchecked,
    plan_level1_halo_chunk_unchecked,
    plan_direct_face_closed_prefix_unchecked,
)
from .face_closure import (
    _require_face_table,
    minimum_direct_face_closed_slots,
)
from ._primary import fill_ascending_primary_prefix_unchecked
from .primary import _require_primary_ids
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


def _require_chunk_ids(chunk_block_ids: np.ndarray) -> np.ndarray:
    return _require_primary_ids("chunk_block_ids", chunk_block_ids)


def minimum_face_closed_slots(face_neighbor_ids: np.ndarray) -> int:
    """Compatibility wrapper for FCL-001 minimum direct-face capacity."""
    return minimum_direct_face_closed_slots(face_neighbor_ids)


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

    if not include_face_closure:
        primary_count = int(
            fill_ascending_primary_prefix_unchecked(
                first_primary_id,
                face_neighbor_ids.shape[0],
                chunk_block_ids,
            )
        )
        return primary_count, primary_count

    return tuple(
        int(value)
        for value in plan_direct_face_closed_prefix_unchecked(
            first_primary_id,
            face_neighbor_ids,
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
