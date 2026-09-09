"""Fixed-order streaming reductions for the functional rewrite."""

import cython

from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accumulate_field_sum_unchecked(
    const double[:, :, :, :, ::1] payload,
    const int64_t[::1] valid_lower,
    const int64_t[::1] valid_upper,
    int64_t field_position,
    double[::1] accumulator,
):
    cdef int64_t slot, i, j, k
    cdef volatile double total = accumulator[0]
    with nogil:
        for slot in range(payload.shape[0]):
            for i in range(valid_lower[0], valid_upper[0]):
                for j in range(valid_lower[1], valid_upper[1]):
                    for k in range(valid_lower[2], valid_upper[2]):
                        total = total + payload[slot, field_position, i, j, k]
        accumulator[0] = total


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void merge_field_sums_unchecked(
    double[::1] accumulator,
    const double[::1] partial,
):
    cdef volatile double result = accumulator[0] + partial[0]
    accumulator[0] = result
