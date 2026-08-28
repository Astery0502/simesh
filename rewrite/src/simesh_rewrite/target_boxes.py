"""Validated TGT-001 directed halo target boxes."""

from __future__ import annotations

import numpy as np

from ._target_boxes import (
    center_target_direction_row_unchecked,
    fill_directed_halo_target_boxes_unchecked,
    invalid_target_direction_entry_unchecked,
)
from .foundation import INDEX_DTYPE, _require_index_triplet


def _require_directions(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("directions must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("directions must have dtype int64")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError(f"directions must have shape (D,3), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError("directions must be C-contiguous")
    return value


def _require_box_output(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def fill_directed_halo_target_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    directions: np.ndarray,
    direction_lower: np.ndarray,
    direction_upper: np.ndarray,
) -> None:
    """Fill exact target boxes for explicit noncenter directions."""
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    requested_lower = _require_index_triplet(
        "requested_lower", requested_lower
    )
    requested_upper = _require_index_triplet(
        "requested_upper", requested_upper
    )
    directions = _require_directions(directions)
    output_shape = (directions.shape[0], 3)
    direction_lower = _require_box_output(
        "direction_lower", direction_lower, output_shape
    )
    direction_upper = _require_box_output(
        "direction_upper", direction_upper, output_shape
    )

    for axis in range(3):
        if int(requested_lower[axis]) < 0:
            raise ValueError("requested box must be nonnegative")
        if not (
            int(requested_lower[axis]) <= int(interior_lower[axis])
            <= int(interior_upper[axis]) <= int(requested_upper[axis])
        ):
            raise ValueError("requested box must contain the ordered interior")

    invalid_entry = int(
        invalid_target_direction_entry_unchecked(directions)
    )
    if invalid_entry >= 0:
        row, axis = divmod(invalid_entry, 3)
        raise ValueError(
            f"directions entry at row {row}, axis {axis} is outside [-1,1]"
        )
    center_row = int(center_target_direction_row_unchecked(directions))
    if center_row >= 0:
        raise ValueError(f"directions row {center_row} must be noncenter")

    inputs = (
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
    )
    if np.shares_memory(direction_lower, direction_upper) or any(
        np.shares_memory(output, input_value)
        for output in (direction_lower, direction_upper)
        for input_value in inputs
    ):
        raise ValueError("direction outputs must not overlap each other or inputs")

    fill_directed_halo_target_boxes_unchecked(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
        direction_lower,
        direction_upper,
    )
