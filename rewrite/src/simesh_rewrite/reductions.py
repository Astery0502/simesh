"""Validated boundaries for streaming rewrite reductions."""

from __future__ import annotations

import numpy as np

from ._reductions import (
    accumulate_field_sum_unchecked,
    merge_field_sums_unchecked,
)
from .foundation import PAYLOAD_DTYPE, _require_index_triplet, _require_payload


def _require_field_position(value, field_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("field_position must be an integer")
    value = int(value)
    if value < 0 or value >= field_count:
        raise ValueError("field_position is outside the field axis")
    return value


def _require_accumulator(
    name: str,
    value: np.ndarray,
    *,
    writable: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != PAYLOAD_DTYPE:
        raise TypeError(f"{name} must have dtype float64")
    if value.shape != (1,) or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous one-value vector")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def accumulate_field_sum(
    payload: np.ndarray,
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    field_position: int,
    accumulator: np.ndarray,
) -> None:
    """Add one field/region to a persistent scalar accumulator in fixed order."""
    payload = _require_payload("payload", payload, writable=False)
    valid_lower = _require_index_triplet("valid_lower", valid_lower)
    valid_upper = _require_index_triplet("valid_upper", valid_upper)
    field_position = _require_field_position(field_position, payload.shape[1])
    accumulator = _require_accumulator(
        "accumulator",
        accumulator,
        writable=True,
    )

    spatial_shape = np.asarray(payload.shape[2:], dtype=np.int64)
    if np.any(valid_lower < 0) or np.any(valid_lower > valid_upper):
        raise ValueError("valid region must be ordered and nonnegative")
    if np.any(valid_upper > spatial_shape):
        raise ValueError("valid region exceeds payload spatial shape")
    metadata = (valid_lower, valid_upper)
    if np.shares_memory(payload, accumulator) or any(
        np.shares_memory(accumulator, value) for value in metadata
    ):
        raise ValueError("accumulator must not overlap payload or region metadata")
    if payload.shape[0] == 0 or np.any(valid_lower == valid_upper):
        return

    accumulate_field_sum_unchecked(
        payload,
        valid_lower,
        valid_upper,
        field_position,
        accumulator,
    )


def merge_field_sums(
    accumulator: np.ndarray,
    partial: np.ndarray,
) -> None:
    """Merge one scalar partial into a writable accumulator with one addition."""
    accumulator = _require_accumulator(
        "accumulator",
        accumulator,
        writable=True,
    )
    partial = _require_accumulator("partial", partial, writable=False)
    if np.shares_memory(accumulator, partial):
        raise ValueError("accumulator and partial must not overlap")
    merge_field_sums_unchecked(accumulator, partial)


def finalize_field_sum(accumulator: np.ndarray) -> float:
    """Return the accumulator value without arithmetic or mutation."""
    accumulator = _require_accumulator(
        "accumulator",
        accumulator,
        writable=False,
    )
    return float(accumulator[0])
