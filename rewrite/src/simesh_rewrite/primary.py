"""Validated PRI-001 deterministic ascending primary-prefix planning."""

from __future__ import annotations

import numpy as np

from ._primary import fill_ascending_primary_prefix_unchecked
from .foundation import INDEX_DTYPE
from .workspace import _require_nonnegative_integer


def _require_primary_ids(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous vector")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def fill_ascending_primary_prefix(
    first_primary_id: int,
    block_count: int,
    primary_ids: np.ndarray,
) -> int:
    """Fill the maximal dense ascending primary prefix into caller storage."""
    first_primary_id = _require_nonnegative_integer(
        "first_primary_id", first_primary_id
    )
    block_count = _require_nonnegative_integer("block_count", block_count)
    primary_ids = _require_primary_ids("primary_ids", primary_ids)
    if first_primary_id > block_count:
        raise ValueError("first_primary_id exceeds block count")
    if first_primary_id < block_count and primary_ids.shape[0] == 0:
        raise ValueError("primary capacity must be positive before the end")
    return int(
        fill_ascending_primary_prefix_unchecked(
            first_primary_id,
            block_count,
            primary_ids,
        )
    )
