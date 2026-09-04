# cython: boundscheck=False, wraparound=False

"""CWP-001 normalized COARSER workspace geometry."""

from libc.stdint cimport int64_t, uint8_t


cdef inline int64_t _floor_divide_two(int64_t value) noexcept nogil:
    if value >= 0:
        return value >> 1
    return -1 - ((-1 - value) >> 1)


cpdef void fill_coarser_workspace_boxes_unchecked(
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[:, ::1] reduced_directions,
    const uint8_t[::1] phase_codes,
    const int64_t[:, ::1] target_lower,
    const int64_t[:, ::1] target_upper,
    int64_t[:, ::1] coarse_source_lower,
    int64_t[:, ::1] coarse_source_upper,
    int64_t[:, ::1] workspace_source_lower,
    int64_t[:, ::1] workspace_source_upper,
    int64_t[:, ::1] workspace_required_lower,
    int64_t[:, ::1] workspace_required_upper,
    int64_t[:, ::1] workspace_coarse_origin,
):
    cdef int64_t row, axis, lower, upper, extent, direction, phase_bit
    cdef int64_t delta, logical_origin, center_lower, center_upper
    cdef int64_t required_lower, required_upper, source_lower, source_upper, base
    cdef bint empty
    for row in range(reduced_directions.shape[0]):
        empty = False
        for axis in range(3):
            if target_lower[row, axis] == target_upper[row, axis]:
                empty = True
        if empty:
            for axis in range(3):
                coarse_source_lower[row, axis] = 0
                coarse_source_upper[row, axis] = 0
                workspace_source_lower[row, axis] = 0
                workspace_source_upper[row, axis] = 0
                workspace_required_lower[row, axis] = 0
                workspace_required_upper[row, axis] = 0
                workspace_coarse_origin[row, axis] = 0
            continue

        for axis in range(3):
            lower = interior_lower[axis]
            upper = interior_upper[axis]
            extent = upper - lower
            direction = reduced_directions[row, axis]
            phase_bit = (phase_codes[row] >> axis) & 1
            delta = _floor_divide_two(phase_bit + direction)
            logical_origin = lower + phase_bit * (extent // 2) - delta * extent
            center_lower = logical_origin + _floor_divide_two(
                target_lower[row, axis] - lower
            )
            center_upper = logical_origin + _floor_divide_two(
                target_upper[row, axis] - 1 - lower
            ) + 1
            required_lower = center_lower - 1
            required_upper = center_upper + 1
            source_lower = required_lower if required_lower > lower else lower
            source_upper = required_upper if required_upper < upper else upper
            base = required_lower if required_lower < logical_origin else logical_origin

            coarse_source_lower[row, axis] = source_lower
            coarse_source_upper[row, axis] = source_upper
            workspace_source_lower[row, axis] = source_lower - base
            workspace_source_upper[row, axis] = source_upper - base
            workspace_required_lower[row, axis] = required_lower - base
            workspace_required_upper[row, axis] = required_upper - base
            workspace_coarse_origin[row, axis] = logical_origin - base
