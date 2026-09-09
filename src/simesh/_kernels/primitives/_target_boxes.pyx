# cython: boundscheck=False, wraparound=False

"""TGT-001 directed halo target-box validation and fill."""

from libc.stdint cimport int64_t


cpdef int64_t invalid_target_direction_entry_unchecked(
    const int64_t[:, ::1] directions,
):
    cdef int64_t row, axis, value
    for row in range(directions.shape[0]):
        for axis in range(3):
            value = directions[row, axis]
            if value < -1 or value > 1:
                return row * 3 + axis
    return -1


cpdef int64_t center_target_direction_row_unchecked(
    const int64_t[:, ::1] directions,
):
    cdef int64_t row
    for row in range(directions.shape[0]):
        if (
            directions[row, 0] == 0
            and directions[row, 1] == 0
            and directions[row, 2] == 0
        ):
            return row
    return -1


cpdef void fill_directed_halo_target_boxes_unchecked(
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[::1] requested_lower,
    const int64_t[::1] requested_upper,
    const int64_t[:, ::1] directions,
    int64_t[:, ::1] direction_lower,
    int64_t[:, ::1] direction_upper,
):
    cdef int64_t row, axis, value
    for row in range(directions.shape[0]):
        for axis in range(3):
            value = directions[row, axis]
            if value < 0:
                direction_lower[row, axis] = requested_lower[axis]
                direction_upper[row, axis] = interior_lower[axis]
            elif value > 0:
                direction_lower[row, axis] = interior_upper[axis]
                direction_upper[row, axis] = requested_upper[axis]
            else:
                direction_lower[row, axis] = interior_lower[axis]
                direction_upper[row, axis] = interior_upper[axis]
