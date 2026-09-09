# cython: boundscheck=False, wraparound=False

"""SLB-001 same-level affine source-box fill."""

from libc.stdint cimport int64_t


cpdef void fill_same_level_source_boxes_unchecked(
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[:, ::1] reduced_directions,
    const int64_t[:, ::1] target_lower,
    const int64_t[:, ::1] target_upper,
    int64_t[:, ::1] source_lower,
    int64_t[:, ::1] source_upper,
):
    cdef int64_t row, axis, extent, direction
    for row in range(reduced_directions.shape[0]):
        for axis in range(3):
            extent = interior_upper[axis] - interior_lower[axis]
            direction = reduced_directions[row, axis]
            source_lower[row, axis] = (
                target_lower[row, axis] - direction * extent
            )
            source_upper[row, axis] = (
                target_upper[row, axis] - direction * extent
            )
