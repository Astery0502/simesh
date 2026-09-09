"""Validated LIM-001 Cartesian three-point limited slope."""

from __future__ import annotations

import numpy as np

from ._limiter import three_point_limited_slope_unchecked


def _require_binary64_scalar(name: str, value) -> float:
    if type(value) is not float and type(value) is not np.float64:
        raise TypeError(f"{name} must be a binary64 floating scalar")
    return float(value)


def three_point_limited_slope(
    left_value: float,
    center_value: float,
    right_value: float,
) -> float:
    """Return the exact current limited slope for three binary64 values."""
    left_value = _require_binary64_scalar("left_value", left_value)
    center_value = _require_binary64_scalar("center_value", center_value)
    right_value = _require_binary64_scalar("right_value", right_value)
    return float(
        three_point_limited_slope_unchecked(
            left_value,
            center_value,
            right_value,
        )
    )
