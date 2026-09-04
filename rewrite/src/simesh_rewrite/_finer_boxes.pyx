# cython: boundscheck=False, wraparound=False

"""FRP-001 exact FINER restriction box placement."""

from libc.stdint cimport int64_t, uint8_t


cpdef void fill_finer_restriction_boxes_unchecked(
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[:, ::1] reduced_directions,
    const uint8_t[::1] phase_codes,
    const int64_t[:, ::1] target_lower,
    const int64_t[:, ::1] target_upper,
    int64_t[:, ::1] placed_target_lower,
    int64_t[:, ::1] placed_target_upper,
    int64_t[:, ::1] fine_source_lower,
    int64_t[:, ::1] fine_source_upper,
):
    cdef int64_t row, axis, lower, upper, extent, half
    cdef int64_t direction, width, phase_bit
    for row in range(reduced_directions.shape[0]):
        for axis in range(3):
            lower = interior_lower[axis]
            upper = interior_upper[axis]
            extent = upper - lower
            direction = reduced_directions[row, axis]
            phase_bit = (phase_codes[row] >> axis) & 1
            if direction == 0:
                half = extent // 2
                placed_target_lower[row, axis] = lower + phase_bit * half
                placed_target_upper[row, axis] = lower + (phase_bit + 1) * half
                fine_source_lower[row, axis] = lower
                fine_source_upper[row, axis] = upper
            elif direction < 0:
                width = target_upper[row, axis] - target_lower[row, axis]
                placed_target_lower[row, axis] = target_lower[row, axis]
                placed_target_upper[row, axis] = target_upper[row, axis]
                fine_source_lower[row, axis] = upper - 2 * width
                fine_source_upper[row, axis] = upper
            else:
                width = target_upper[row, axis] - target_lower[row, axis]
                placed_target_lower[row, axis] = target_lower[row, axis]
                placed_target_upper[row, axis] = target_upper[row, axis]
                fine_source_lower[row, axis] = lower
                fine_source_upper[row, axis] = lower + 2 * width
