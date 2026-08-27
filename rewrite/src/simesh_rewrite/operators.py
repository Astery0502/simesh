"""Validated boundaries for concrete rewrite operators."""

from __future__ import annotations

import numpy as np

from ._operators import scaled_difference_into_unchecked
from .foundation import _require_index_triplet, _require_payload


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _require_field_position(name: str, value, field_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or value >= field_count:
        raise ValueError(f"{name} is outside the field axis")
    return value


def _require_scale(scale) -> float:
    if type(scale) is not float and type(scale) is not np.float64:
        raise TypeError("scale must be a binary64 floating scalar")
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    return scale


def scaled_difference_into(
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    left_field_position: int,
    right_field_position: int,
    scale: float,
    destination: np.ndarray,
    destination_field_position: int,
    destination_lower: np.ndarray,
) -> None:
    """Compute ``left - scale*right`` into one translated output field."""
    source = _require_payload("source", source, writable=False)
    source_valid_lower = _require_index_triplet(
        "source_valid_lower", source_valid_lower
    )
    source_valid_upper = _require_index_triplet(
        "source_valid_upper", source_valid_upper
    )
    destination = _require_payload("destination", destination, writable=True)
    destination_lower = _require_index_triplet(
        "destination_lower", destination_lower
    )
    left_field_position = _require_field_position(
        "left_field_position",
        left_field_position,
        source.shape[1],
    )
    right_field_position = _require_field_position(
        "right_field_position",
        right_field_position,
        source.shape[1],
    )
    destination_field_position = _require_field_position(
        "destination_field_position",
        destination_field_position,
        destination.shape[1],
    )
    scale = _require_scale(scale)

    if source.shape[0] != destination.shape[0]:
        raise ValueError("source and destination slot extents must match")
    source_shape = np.asarray(source.shape[2:], dtype=np.int64)
    if np.any(source_valid_lower < 0) or np.any(
        source_valid_lower > source_valid_upper
    ):
        raise ValueError("source valid region must be ordered and nonnegative")
    if np.any(source_valid_upper > source_shape):
        raise ValueError("source valid region exceeds source spatial shape")
    if np.any(destination_lower < 0):
        raise ValueError("destination_lower must be nonnegative")
    extent = source_valid_upper - source_valid_lower
    for axis in range(3):
        if int(destination_lower[axis]) > _INDEX_MAX - int(extent[axis]):
            raise OverflowError("translated destination region does not fit in int64")
    destination_shape = np.asarray(destination.shape[2:], dtype=np.int64)
    if np.any(destination_lower > destination_shape) or np.any(
        extent > destination_shape - destination_lower
    ):
        raise ValueError("translated region exceeds destination spatial shape")

    metadata = (
        source_valid_lower,
        source_valid_upper,
        destination_lower,
    )
    if np.shares_memory(source, destination) or any(
        np.shares_memory(destination, value) for value in metadata
    ):
        raise ValueError("destination must not overlap source or region metadata")

    scaled_difference_into_unchecked(
        source,
        source_valid_lower,
        source_valid_upper,
        left_field_position,
        right_field_position,
        scale,
        destination,
        destination_field_position,
        destination_lower,
    )
