"""Independent arbitrary-integer reference for WSP-001."""

from __future__ import annotations

import numpy as np


def _workspace_bytes_per_slot_reference(
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    volume = 1
    for extent in workspace_shape:
        volume *= int(extent)
    return 8 * (int(field_count) * volume + 1)


def workspace_nbytes_reference(
    slot_capacity: int,
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    """Return exact payload-plus-ID bytes using Python integer arithmetic."""
    return int(slot_capacity) * _workspace_bytes_per_slot_reference(
        field_count,
        workspace_shape,
    )


def workspace_slot_capacity_reference(
    budget_bytes: int,
    block_count: int,
    field_count: int,
    workspace_shape: np.ndarray,
) -> int:
    """Return the exact budget-limited and block-clamped slot capacity."""
    bytes_per_slot = _workspace_bytes_per_slot_reference(
        field_count,
        workspace_shape,
    )
    return min(int(block_count), int(budget_bytes) // bytes_per_slot)
