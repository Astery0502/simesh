"""Validated boundaries for concrete rewrite operators."""

from __future__ import annotations

import numpy as np

from ._operators import (
    central_difference_into_unchecked,
    scaled_difference_into_unchecked,
)
from .foundation import PAYLOAD_DTYPE, _require_index_triplet, _require_payload


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


def _require_axis(axis) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    axis = int(axis)
    if axis < 0 or axis >= 3:
        raise ValueError("axis must be in [0,3)")
    return axis


def _require_spacing(cell_spacing: np.ndarray) -> np.ndarray:
    if not isinstance(cell_spacing, np.ndarray):
        raise TypeError("cell_spacing must be a NumPy array")
    if cell_spacing.dtype != PAYLOAD_DTYPE:
        raise TypeError("cell_spacing must have dtype float64")
    if cell_spacing.shape != (3,) or not cell_spacing.flags.c_contiguous:
        raise ValueError("cell_spacing must be a C-contiguous triplet")
    if not np.all(np.isfinite(cell_spacing)) or np.any(
        cell_spacing < np.finfo(np.float64).tiny
    ):
        raise ValueError("cell_spacing entries must be finite, positive, and normal")
    return cell_spacing


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


def central_difference_into(
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    source_field_position: int,
    axis: int,
    cell_spacing: np.ndarray,
    destination: np.ndarray,
    destination_field_position: int,
    destination_lower: np.ndarray,
) -> None:
    """Compute one axis-centered first derivative into a translated field."""
    source = _require_payload("source", source, writable=False)
    source_valid_lower = _require_index_triplet(
        "source_valid_lower", source_valid_lower
    )
    source_valid_upper = _require_index_triplet(
        "source_valid_upper", source_valid_upper
    )
    output_lower = _require_index_triplet("output_lower", output_lower)
    output_upper = _require_index_triplet("output_upper", output_upper)
    source_field_position = _require_field_position(
        "source_field_position",
        source_field_position,
        source.shape[1],
    )
    axis = _require_axis(axis)
    cell_spacing = _require_spacing(cell_spacing)
    destination = _require_payload("destination", destination, writable=True)
    destination_field_position = _require_field_position(
        "destination_field_position",
        destination_field_position,
        destination.shape[1],
    )
    destination_lower = _require_index_triplet(
        "destination_lower", destination_lower
    )

    if source.shape[0] != destination.shape[0]:
        raise ValueError("source and destination slot extents must match")
    source_shape = np.asarray(source.shape[2:], dtype=np.int64)
    if np.any(source_valid_lower < 0) or np.any(
        source_valid_lower > source_valid_upper
    ):
        raise ValueError("source valid region must be ordered and nonnegative")
    if np.any(source_valid_upper > source_shape):
        raise ValueError("source valid region exceeds source spatial shape")
    if np.any(output_lower < 0) or np.any(output_lower > output_upper):
        raise ValueError("output region must be ordered and nonnegative")
    if np.any(output_upper > source_shape):
        raise ValueError("output region exceeds source spatial shape")

    extent = output_upper - output_lower
    empty = bool(np.any(extent == 0))
    if not empty:
        for dimension in range(3):
            reach = 1 if dimension == axis else 0
            if (
                int(output_lower[dimension]) < reach
                or int(output_upper[dimension]) > _INDEX_MAX - reach
                or int(source_valid_lower[dimension])
                > int(output_lower[dimension]) - reach
                or int(source_valid_upper[dimension])
                < int(output_upper[dimension]) + reach
            ):
                raise ValueError("source valid region does not support output stencil")

    if np.any(destination_lower < 0):
        raise ValueError("destination_lower must be nonnegative")
    for dimension in range(3):
        if int(destination_lower[dimension]) > _INDEX_MAX - int(extent[dimension]):
            raise OverflowError("translated destination region does not fit in int64")
    destination_shape = np.asarray(destination.shape[2:], dtype=np.int64)
    if np.any(destination_lower > destination_shape) or np.any(
        extent > destination_shape - destination_lower
    ):
        raise ValueError("translated region exceeds destination spatial shape")

    metadata = (
        source_valid_lower,
        source_valid_upper,
        output_lower,
        output_upper,
        cell_spacing,
        destination_lower,
    )
    if np.shares_memory(source, destination) or any(
        np.shares_memory(destination, value) for value in metadata
    ):
        raise ValueError("destination must not overlap source or operator metadata")
    if empty or source.shape[0] == 0:
        return

    central_difference_into_unchecked(
        source,
        output_lower,
        output_upper,
        source_field_position,
        axis,
        cell_spacing,
        destination,
        destination_field_position,
        destination_lower,
    )
