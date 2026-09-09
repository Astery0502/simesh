"""Unchecked scalar wrappers around the PBC-001 inline rules."""

from libc.stdint cimport int64_t, uint8_t


cpdef int64_t physical_halo_source_index_unchecked(
    int64_t target,
    int64_t lower,
    int64_t upper,
    int64_t face,
    uint8_t mode,
):
    return physical_halo_source_index_c(target, lower, upper, face, mode)


cpdef double transform_physical_halo_value_unchecked(
    double value,
    int64_t field_position,
    int64_t normal_field_slot,
    int64_t face,
    uint8_t mode,
):
    return transform_physical_halo_value_c(
        value,
        field_position,
        normal_field_slot,
        face,
        mode,
    )
