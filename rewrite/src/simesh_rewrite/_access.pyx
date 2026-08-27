"""Typed region arithmetic for FND-002 access requirements."""

from libc.stdint cimport int64_t


cpdef void required_input_region_unchecked(
    const int64_t[::1] output_lower,
    const int64_t[::1] output_upper,
    const int64_t[::1] lower_reach,
    const int64_t[::1] upper_reach,
    int64_t[::1] required_lower,
    int64_t[::1] required_upper,
):
    cdef Py_ssize_t axis
    for axis in range(3):
        required_lower[axis] = output_lower[axis] - lower_reach[axis]
        required_upper[axis] = output_upper[axis] + upper_reach[axis]


cpdef void valid_output_region_unchecked(
    const int64_t[::1] valid_lower,
    const int64_t[::1] valid_upper,
    const int64_t[::1] lower_reach,
    const int64_t[::1] upper_reach,
    int64_t[::1] output_lower,
    int64_t[::1] output_upper,
):
    cdef Py_ssize_t axis
    for axis in range(3):
        output_lower[axis] = valid_lower[axis] + lower_reach[axis]
        output_upper[axis] = valid_upper[axis] - upper_reach[axis]
