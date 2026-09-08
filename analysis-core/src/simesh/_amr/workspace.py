"""WSP-001 exact canonical payload-plus-ID workspace accounting."""

from __future__ import annotations

from typing import Final

import numpy as np

from simesh._amr.foundation import _require_index_triplet


_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


def _require_nonnegative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return value


def _workspace_bytes_per_slot(
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    field_count = _require_nonnegative_integer("field_count", field_count)
    if field_count == 0:
        raise ValueError("field_count must be positive")
    workspace_shape = _require_index_triplet("workspace_shape", workspace_shape)
    if np.any(workspace_shape <= 0):
        raise ValueError("workspace_shape entries must be positive")
    cells = 1
    for extent in workspace_shape:
        extent = int(extent)
        if cells > _INDEX_MAX // extent:
            raise OverflowError("workspace spatial volume does not fit in int64")
        cells *= extent
    values = field_count * cells
    if values > (_INDEX_MAX // 8) - 1:
        raise OverflowError("workspace bytes per slot do not fit in int64")
    return 8 * (values + 1)


def workspace_nbytes(
    slot_capacity: int,
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    """Return exact canonical payload-plus-ID bytes for full capacity."""
    slot_capacity = _require_nonnegative_integer("slot_capacity", slot_capacity)
    per_slot = _workspace_bytes_per_slot(field_count, workspace_shape)
    if slot_capacity > _INDEX_MAX // per_slot:
        raise OverflowError("managed workspace bytes do not fit in int64")
    return slot_capacity * per_slot


def workspace_slot_capacity(
    budget_bytes: int,
    block_count: int,
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    """Return maximum canonical workspace slots fitting budget and limit."""
    budget_bytes = _require_nonnegative_integer("budget_bytes", budget_bytes)
    block_count = _require_nonnegative_integer("block_count", block_count)
    per_slot = _workspace_bytes_per_slot(field_count, workspace_shape)
    return min(block_count, budget_bytes // per_slot)
