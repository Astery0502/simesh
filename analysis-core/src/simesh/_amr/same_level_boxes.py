"""Validated SLB-001 Cartesian same-level source boxes."""

from __future__ import annotations

import numpy as np

from simesh._kernels.primitives._same_level_boxes import fill_same_level_source_boxes_unchecked
from simesh._amr.foundation import INDEX_DTYPE, _require_index_triplet


def _require_rows(name: str, value: np.ndarray, shape=None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (R,3), got {value.shape}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_output(name: str, value: np.ndarray, shape) -> np.ndarray:
    value = _require_rows(name, value, shape)
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def fill_same_level_source_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    reduced_directions: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
) -> None:
    """Translate reduced-direction target boxes into same-level source boxes."""
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    reduced_directions = _require_rows(
        "reduced_directions", reduced_directions
    )
    row_shape = reduced_directions.shape
    target_lower = _require_rows("target_lower", target_lower, row_shape)
    target_upper = _require_rows("target_upper", target_upper, row_shape)
    source_lower = _require_output("source_lower", source_lower, row_shape)
    source_upper = _require_output("source_upper", source_upper, row_shape)

    for axis in range(3):
        if int(interior_lower[axis]) < 0 or int(interior_lower[axis]) > int(
            interior_upper[axis]
        ):
            raise ValueError("interior box must be ordered and nonnegative")

    for row in range(row_shape[0]):
        noncenter = False
        for axis in range(3):
            direction = int(reduced_directions[row, axis])
            if direction < -1 or direction > 1:
                raise ValueError("reduced direction component is outside [-1,1]")
            noncenter |= direction != 0
        if not noncenter:
            raise ValueError(f"reduced_directions row {row} must be noncenter")

    for row in range(row_shape[0]):
        for axis in range(3):
            direction = int(reduced_directions[row, axis])
            start = int(target_lower[row, axis])
            stop = int(target_upper[row, axis])
            if start < 0 or start > stop:
                raise ValueError("target boxes must be ordered and nonnegative")
            lower = int(interior_lower[axis])
            upper = int(interior_upper[axis])
            extent = upper - lower
            if direction < 0:
                valid = stop == lower and lower - extent <= start <= lower
            elif direction > 0:
                valid = start == upper and upper <= stop <= upper + extent
            else:
                valid = lower <= start <= stop <= upper
            if not valid:
                raise ValueError("target box is incompatible with reduced direction")
            translated_start = start - direction * extent
            translated_stop = stop - direction * extent
            if not (
                lower <= translated_start <= translated_stop <= upper
            ):
                raise ValueError("translated source box exceeds the interior")

    inputs = (
        interior_lower,
        interior_upper,
        reduced_directions,
        target_lower,
        target_upper,
    )
    if np.shares_memory(source_lower, source_upper) or any(
        np.shares_memory(output, input_value)
        for output in (source_lower, source_upper)
        for input_value in inputs
    ):
        raise ValueError("source outputs must not overlap each other or inputs")

    fill_same_level_source_boxes_unchecked(
        interior_lower,
        interior_upper,
        reduced_directions,
        target_lower,
        target_upper,
        source_lower,
        source_upper,
    )
