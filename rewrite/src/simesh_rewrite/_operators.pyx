"""Concrete operator kernels for the functional rewrite."""

import cython

from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void scaled_difference_into_unchecked(
    const double[:, :, :, :, ::1] source,
    const int64_t[::1] source_valid_lower,
    const int64_t[::1] source_valid_upper,
    int64_t left_field_position,
    int64_t right_field_position,
    double scale,
    double[:, :, :, :, ::1] destination,
    int64_t destination_field_position,
    const int64_t[::1] destination_lower,
):
    cdef int64_t slot, i, j, k
    cdef int64_t di, dj, dk
    cdef volatile double product
    cdef volatile double result

    with nogil:
        for slot in range(source.shape[0]):
            for i in range(source_valid_lower[0], source_valid_upper[0]):
                di = destination_lower[0] + i - source_valid_lower[0]
                for j in range(source_valid_lower[1], source_valid_upper[1]):
                    dj = destination_lower[1] + j - source_valid_lower[1]
                    for k in range(source_valid_lower[2], source_valid_upper[2]):
                        dk = destination_lower[2] + k - source_valid_lower[2]
                        product = scale * source[
                            slot,
                            right_field_position,
                            i,
                            j,
                            k,
                        ]
                        result = source[
                            slot,
                            left_field_position,
                            i,
                            j,
                            k,
                        ] - product
                        destination[
                            slot,
                            destination_field_position,
                            di,
                            dj,
                            dk,
                        ] = result
