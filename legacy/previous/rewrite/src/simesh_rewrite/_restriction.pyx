"""RST-001 Cartesian 3D ratio-two cell-average restriction."""

import cython

from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void restrict_cartesian_2to1_into_unchecked(
    const double[:, :, :, :, ::1] fine_payload,
    const int64_t[::1] fine_lower,
    const int64_t[::1] fine_upper,
    double[:, :, :, :, ::1] coarse_payload,
    const int64_t[::1] coarse_lower,
):
    cdef int64_t slot, field, qi, qj, qk
    cdef int64_t i, j, k, I, J, K
    cdef int64_t ni = (fine_upper[0] - fine_lower[0]) // 2
    cdef int64_t nj = (fine_upper[1] - fine_lower[1]) // 2
    cdef int64_t nk = (fine_upper[2] - fine_lower[2]) // 2
    cdef double total
    cdef double result

    with nogil:
        for slot in range(fine_payload.shape[0]):
            for field in range(fine_payload.shape[1]):
                for qi in range(ni):
                    i = fine_lower[0] + 2 * qi
                    I = coarse_lower[0] + qi
                    for qj in range(nj):
                        j = fine_lower[1] + 2 * qj
                        J = coarse_lower[1] + qj
                        for qk in range(nk):
                            k = fine_lower[2] + 2 * qk
                            K = coarse_lower[2] + qk
                            total = fine_payload[slot, field, i, j, k]
                            total = total + fine_payload[
                                slot, field, i + 1, j, k
                            ]
                            total = total + fine_payload[
                                slot, field, i, j + 1, k
                            ]
                            total = total + fine_payload[
                                slot, field, i + 1, j + 1, k
                            ]
                            total = total + fine_payload[
                                slot, field, i, j, k + 1
                            ]
                            total = total + fine_payload[
                                slot, field, i + 1, j, k + 1
                            ]
                            total = total + fine_payload[
                                slot, field, i, j + 1, k + 1
                            ]
                            total = total + fine_payload[
                                slot, field, i + 1, j + 1, k + 1
                            ]
                            result = total * 0.125
                            coarse_payload[slot, field, I, J, K] = result
