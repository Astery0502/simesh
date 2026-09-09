from libc.stdint cimport int64_t, uint8_t


cdef inline int64_t physical_halo_source_index_c(
    int64_t target,
    int64_t lower,
    int64_t upper,
    int64_t face,
    uint8_t mode,
) noexcept nogil:
    cdef int64_t layer
    if mode == 1 or mode == 2:
        if (face & 1) == 0:
            layer = lower - target
            return lower + layer - 1
        layer = target - upper + 1
        return upper - layer
    if (face & 1) == 0:
        return lower
    return upper - 1


cdef inline double transform_physical_halo_value_c(
    double value,
    int64_t field_position,
    int64_t normal_field_slot,
    int64_t face,
    uint8_t mode,
) noexcept nogil:
    if mode == 2:
        return -value
    if mode == 3 and field_position == normal_field_slot:
        if (face & 1) == 0:
            if value > 0.0:
                return 0.0
        elif value < 0.0:
            return 0.0
    return value
