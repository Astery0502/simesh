"""Validated CWP-001 Cartesian COARSER workspace geometry."""

from __future__ import annotations

import numpy as np

from ._coarser_workspace import fill_coarser_workspace_boxes_unchecked
from .foundation import _require_index_triplet
from .same_level_boxes import _require_output, _require_rows


_INDEX_MIN = int(np.iinfo(np.int64).min)
_INDEX_MAX = int(np.iinfo(np.int64).max)


def _checked_int64(name: str, value: int) -> int:
    if value < _INDEX_MIN or value > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return value


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


def fill_coarser_workspace_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    reduced_directions: np.ndarray,
    phase_codes: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    coarse_source_lower: np.ndarray,
    coarse_source_upper: np.ndarray,
    workspace_source_lower: np.ndarray,
    workspace_source_upper: np.ndarray,
    workspace_required_lower: np.ndarray,
    workspace_required_upper: np.ndarray,
    workspace_coarse_origin: np.ndarray,
) -> None:
    """Map active COARSER rows to normalized PRL workspace geometry."""
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
    outputs = tuple(
        _require_output(name, value, row_shape)
        for name, value in (
            ("coarse_source_lower", coarse_source_lower),
            ("coarse_source_upper", coarse_source_upper),
            ("workspace_source_lower", workspace_source_lower),
            ("workspace_source_upper", workspace_source_upper),
            ("workspace_required_lower", workspace_required_lower),
            ("workspace_required_upper", workspace_required_upper),
            ("workspace_coarse_origin", workspace_coarse_origin),
        )
    )

    extents = []
    for axis in range(3):
        lower = int(interior_lower[axis])
        upper = int(interior_upper[axis])
        extent = upper - lower
        if lower < 0 or extent <= 0 or extent % 2:
            raise ValueError("interior extents must be positive and even")
        extents.append(extent)

    for row in range(row_count):
        phase = int(phase_codes[row])
        if phase > 7:
            raise ValueError(f"phase_codes entry {row} is outside [0,7]")
        noncenter = False
        empty = any(
            int(target_lower[row, axis]) == int(target_upper[row, axis])
            for axis in range(3)
        )
        for axis in range(3):
            direction = int(reduced_directions[row, axis])
            if direction < -1 or direction > 1:
                raise ValueError("reduced direction component is outside [-1,1]")
            noncenter |= direction != 0
            lower = int(interior_lower[axis])
            upper = int(interior_upper[axis])
            extent = extents[axis]
            start = int(target_lower[row, axis])
            stop = int(target_upper[row, axis])
            if start < 0 or start > stop:
                raise ValueError("target boxes must be ordered and nonnegative")
            phase_bit = (phase >> axis) & 1
            if direction == 0:
                valid = start == lower and stop == upper
            elif direction < 0:
                valid = (
                    stop == lower
                    and lower - extent <= start <= lower
                )
            else:
                valid = (
                    start == upper
                    and upper <= stop <= upper + extent
                )
            if not valid:
                raise ValueError(
                    "COARSER target is incompatible with direction"
                )

            if not empty and start != stop:
                delta = (phase_bit + direction) // 2
                logical_origin = _checked_int64(
                    "logical coarse origin",
                    lower + phase_bit * (extent // 2) - delta * extent,
                )
                center_lower = _checked_int64(
                    "coarse center lower",
                    logical_origin + (start - lower) // 2,
                )
                center_upper = _checked_int64(
                    "coarse center upper",
                    logical_origin + (stop - 1 - lower) // 2 + 1,
                )
                _checked_int64("coarse required lower", center_lower - 1)
                _checked_int64("coarse required upper", center_upper + 1)
                if center_lower < lower or center_upper > upper:
                    raise ValueError(
                        "mapped coarse centers exceed the relation source interior"
                    )
        if not noncenter:
            raise ValueError(f"reduced_directions row {row} must be noncenter")

        if not empty:
            for axis in range(3):
                direction = int(reduced_directions[row, axis])
                phase_bit = (phase >> axis) & 1
                lower = int(interior_lower[axis])
                extent = extents[axis]
                start = int(target_lower[row, axis])
                stop = int(target_upper[row, axis])
                delta = (phase_bit + direction) // 2
                logical_origin = lower + phase_bit * (extent // 2) - delta * extent
                center_lower = logical_origin + (start - lower) // 2
                center_upper = logical_origin + (stop - 1 - lower) // 2 + 1
                required_lower = center_lower - 1
                base = min(required_lower, logical_origin)
                _checked_int64("workspace required lower", required_lower - base)
                _checked_int64("workspace required upper", center_upper + 1 - base)
                _checked_int64("workspace origin", logical_origin - base)

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
        raise ValueError("workspace outputs must not overlap each other or inputs")

    fill_coarser_workspace_boxes_unchecked(
        interior_lower,
        interior_upper,
        reduced_directions,
        phase_codes,
        target_lower,
        target_upper,
        *outputs,
    )
