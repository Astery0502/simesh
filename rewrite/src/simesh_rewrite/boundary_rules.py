"""Validated PBC-001 per-axis Cartesian physical boundary rules."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._boundary_rules import (
    physical_halo_source_index_unchecked,
    transform_physical_halo_value_unchecked,
)
from .foundation import _require_index_triplet


_INDEX_MIN: Final = int(np.iinfo(np.int64).min)
_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


def _require_integer_scalar(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer scalar")
    result = int(value)
    if result < _INDEX_MIN or result > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return result


def _require_face(value) -> int:
    face = _require_integer_scalar("face", value)
    if face < 0 or face >= 6:
        raise ValueError("face must be in [0, 6)")
    return face


def _require_mode(value) -> int:
    mode = _require_integer_scalar("mode", value)
    if mode < 0 or mode > 3:
        raise ValueError("mode must be a BoundaryMode code in [0, 4)")
    return mode


def physical_halo_source_index(
    target_index: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    face: int,
    mode: int,
) -> int:
    """Return one validated physical-face source coordinate."""
    target_index = _require_integer_scalar("target_index", target_index)
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    face = _require_face(face)
    mode = _require_mode(mode)
    if np.any(interior_lower < 0) or np.any(interior_lower >= interior_upper):
        raise ValueError("interior must be a nonempty nonnegative box")

    axis = face // 2
    lower = int(interior_lower[axis])
    upper = int(interior_upper[axis])
    if face % 2 == 0:
        if target_index >= lower:
            raise ValueError("target_index is not outside the lower face")
        layer = lower - target_index
    else:
        if target_index < upper:
            raise ValueError("target_index is not outside the upper face")
        layer = target_index - upper + 1
    if mode in (1, 2) and layer > upper - lower:
        raise ValueError("reflected target depth exceeds interior extent")

    return int(
        physical_halo_source_index_unchecked(
            target_index,
            lower,
            upper,
            face,
            mode,
        )
    )


def transform_physical_halo_value(
    value: float,
    field_position: int,
    normal_field_slot: int,
    face: int,
    mode: int,
) -> float:
    """Return one validated physical-face binary64 value transform."""
    if type(value) not in (float, np.float64):
        raise TypeError("value must be an exact float or numpy.float64 scalar")
    field_position = _require_integer_scalar("field_position", field_position)
    normal_field_slot = _require_integer_scalar(
        "normal_field_slot", normal_field_slot
    )
    face = _require_face(face)
    mode = _require_mode(mode)
    if field_position < 0:
        raise ValueError("field_position must be non-negative")
    if normal_field_slot < -1:
        raise ValueError("normal_field_slot must be -1 or non-negative")
    return float(
        transform_physical_halo_value_unchecked(
            value,
            field_position,
            normal_field_slot,
            face,
            mode,
        )
    )
