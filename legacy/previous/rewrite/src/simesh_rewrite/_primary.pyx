# cython: boundscheck=False, wraparound=False

"""PRI-001 sequential dense primary-prefix fill."""

from libc.stdint cimport int64_t


cpdef int64_t fill_ascending_primary_prefix_unchecked(
    int64_t first_primary_id,
    int64_t block_count,
    int64_t[::1] primary_ids,
):
    cdef int64_t count = min(
        primary_ids.shape[0], block_count - first_primary_id
    )
    cdef int64_t index
    for index in range(count):
        primary_ids[index] = first_primary_id + index
    return count
