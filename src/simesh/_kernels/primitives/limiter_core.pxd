"""Shared allocation-free LIM-001 scalar rule."""


cdef inline double _limiter_abs(double value) noexcept nogil:
    if value < 0.0:
        return -value
    return value


cdef inline double _limiter_min(double left, double right) noexcept nogil:
    if left < right:
        return left
    return right


cdef inline double three_point_limited_slope_rule(
    double left_value,
    double center_value,
    double right_value,
) noexcept nogil:
    cdef double slope_l = center_value - left_value
    cdef double slope_r = right_value - center_value
    cdef double sum_lr = slope_l + slope_r
    cdef double slope_c = 0.5 * sum_lr
    cdef double sign_c
    cdef double limited

    if slope_c > 0.0:
        sign_c = 1.0
        limited = _limiter_min(
            _limiter_abs(slope_c),
            _limiter_min(slope_l, slope_r),
        )
    elif slope_c < 0.0:
        sign_c = -1.0
        limited = _limiter_min(
            _limiter_abs(slope_c),
            _limiter_min(-slope_l, -slope_r),
        )
    else:
        return 0.0

    if limited <= 0.0:
        return 0.0
    return sign_c * limited
