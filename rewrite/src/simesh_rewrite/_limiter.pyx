"""Compiled public adapter for the LIM-001 shared scalar rule."""

from .limiter_core cimport three_point_limited_slope_rule


cpdef double three_point_limited_slope_unchecked(
    double left_value,
    double center_value,
    double right_value,
):
    return three_point_limited_slope_rule(
        left_value,
        center_value,
        right_value,
    )
