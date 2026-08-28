"""Independent explicit-binary64 reference for LIM-001."""

from __future__ import annotations

import numpy as np


def three_point_limited_slope_reference(
    left_value: float,
    center_value: float,
    right_value: float,
) -> float:
    """Return the contracted current three-point limited slope."""
    left = np.float64(left_value)
    center = np.float64(center_value)
    right = np.float64(right_value)
    with np.errstate(all="ignore"):
        slope_l = np.float64(center - left)
        slope_r = np.float64(right - center)
        sum_lr = np.float64(slope_l + slope_r)
        slope_c = np.float64(np.float64(0.5) * sum_lr)

        if slope_c > np.float64(0.0):
            absolute_c = (
                np.float64(-slope_c)
                if slope_c < np.float64(0.0)
                else slope_c
            )
            inner = slope_l if slope_l < slope_r else slope_r
            limited = absolute_c if absolute_c < inner else inner
            if limited <= np.float64(0.0):
                return float(np.float64(0.0))
            return float(np.float64(np.float64(1.0) * limited))

        if slope_c < np.float64(0.0):
            absolute_c = (
                np.float64(-slope_c)
                if slope_c < np.float64(0.0)
                else slope_c
            )
            negative_l = np.float64(-slope_l)
            negative_r = np.float64(-slope_r)
            inner = negative_l if negative_l < negative_r else negative_r
            limited = absolute_c if absolute_c < inner else inner
            if limited <= np.float64(0.0):
                return float(np.float64(0.0))
            return float(np.float64(np.float64(-1.0) * limited))

        return float(np.float64(0.0))
