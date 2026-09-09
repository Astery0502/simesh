"""Validated FRP-001 Cartesian FINER restriction placement."""

from __future__ import annotations

import numpy as np

from ._finer_boxes import fill_finer_restriction_boxes_unchecked
from .foundation import _require_index_triplet
from .same_level_boxes import _require_output, _require_rows


def _require_phases(value: np.ndarray, row_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("phase_codes must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("phase_codes must have dtype uint8")
    if value.shape != (row_count,):
        raise ValueError(
            f"phase_codes must have shape ({row_count},), got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("phase_codes must be C-contiguous")
    return value


def fill_finer_restriction_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    reduced_directions: np.ndarray,
    phase_codes: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    placed_target_lower: np.ndarray,
    placed_target_upper: np.ndarray,
    fine_source_lower: np.ndarray,
    fine_source_upper: np.ndarray,
) -> None:
    """Map active FINER child phases to RST source and target boxes."""
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    reduced_directions = _require_rows(
        "reduced_directions", reduced_directions
    )
    row_shape = reduced_directions.shape
    row_count = row_shape[0]
    phase_codes = _require_phases(phase_codes, row_count)
    target_lower = _require_rows("target_lower", target_lower, row_shape)
    target_upper = _require_rows("target_upper", target_upper, row_shape)
    outputs = (
        _require_output("placed_target_lower", placed_target_lower, row_shape),
        _require_output("placed_target_upper", placed_target_upper, row_shape),
        _require_output("fine_source_lower", fine_source_lower, row_shape),
        _require_output("fine_source_upper", fine_source_upper, row_shape),
    )

    for axis in range(3):
        if int(interior_lower[axis]) < 0 or int(interior_lower[axis]) > int(
            interior_upper[axis]
        ):
            raise ValueError("interior box must be ordered and nonnegative")

    for row in range(row_count):
        phase = int(phase_codes[row])
        if phase > 7:
            raise ValueError(f"phase_codes entry {row} is outside [0,7]")
        noncenter = False
        for axis in range(3):
            direction = int(reduced_directions[row, axis])
            if direction < -1 or direction > 1:
                raise ValueError("reduced direction component is outside [-1,1]")
            noncenter |= direction != 0
            lower = int(interior_lower[axis])
            upper = int(interior_upper[axis])
            extent = upper - lower
            start = int(target_lower[row, axis])
            stop = int(target_upper[row, axis])
            if start < 0 or start > stop:
                raise ValueError("target boxes must be ordered and nonnegative")
            phase_bit = (phase >> axis) & 1
            if direction == 0:
                if extent <= 0 or extent % 2 or start != lower or stop != upper:
                    raise ValueError(
                        "zero-direction target requires a positive even full interior"
                    )
            elif direction < 0:
                if (
                    stop != lower
                    or start < lower - extent // 2
                    or start > lower
                    or phase_bit != 1
                ):
                    raise ValueError(
                        "lower FINER target or phase is incompatible with direction"
                    )
            elif (
                start != upper
                or stop < upper
                or stop > upper + extent // 2
                or phase_bit != 0
            ):
                raise ValueError(
                    "upper FINER target or phase is incompatible with direction"
                )
        if not noncenter:
            raise ValueError(f"reduced_directions row {row} must be noncenter")

    inputs = (
        interior_lower,
        interior_upper,
        reduced_directions,
        phase_codes,
        target_lower,
        target_upper,
    )
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(outputs)
        for right in (*outputs[index + 1 :], *inputs)
    ):
        raise ValueError("box outputs must not overlap each other or inputs")

    fill_finer_restriction_boxes_unchecked(
        interior_lower,
        interior_upper,
        reduced_directions,
        phase_codes,
        target_lower,
        target_upper,
        *outputs,
    )
