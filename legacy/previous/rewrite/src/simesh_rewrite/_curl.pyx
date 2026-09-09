# cython: boundscheck=False, wraparound=False

"""Fixed Cartesian 3D curl kernel for OPR-003."""

import cython

from libc.stdint cimport int64_t


cdef inline double _centered_derivative(
    double plus_value,
    double minus_value,
    double inverse_two_spacing,
) noexcept nogil:
    cdef volatile double difference = plus_value - minus_value
    cdef volatile double derivative = difference * inverse_two_spacing
    return derivative


cdef inline double _scaled_difference(
    double positive_derivative,
    double negative_derivative,
) noexcept nogil:
    cdef volatile double product = 1.0 * negative_derivative
    cdef volatile double result = positive_derivative - product
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cartesian_curl_unchecked(
    const double[:, :, :, :, ::1] source,
    const int64_t[::1] output_lower,
    const int64_t[::1] output_upper,
    const int64_t[::1] source_field_positions,
    const double[:, ::1] slot_cell_spacing,
    double[:, :, :, :, ::1] destination,
    const int64_t[::1] destination_field_positions,
    const int64_t[::1] destination_lower,
):
    cdef int64_t slot, i, j, k, di, dj, dk
    cdef int64_t bx = source_field_positions[0]
    cdef int64_t by = source_field_positions[1]
    cdef int64_t bz = source_field_positions[2]
    cdef int64_t jx = destination_field_positions[0]
    cdef int64_t jy = destination_field_positions[1]
    cdef int64_t jz = destination_field_positions[2]
    cdef volatile double inv_x, inv_y, inv_z
    cdef volatile double bz_yp, bz_ym, by_zp, by_zm
    cdef volatile double bx_zp, bx_zm, bz_xp, bz_xm
    cdef volatile double by_xp, by_xm, bx_yp, bx_ym
    cdef double positive, negative

    with nogil:
        for slot in range(source.shape[0]):
            inv_x = 0.5 / slot_cell_spacing[slot, 0]
            inv_y = 0.5 / slot_cell_spacing[slot, 1]
            inv_z = 0.5 / slot_cell_spacing[slot, 2]
            for i in range(output_lower[0], output_upper[0]):
                di = destination_lower[0] + i - output_lower[0]
                for j in range(output_lower[1], output_upper[1]):
                    dj = destination_lower[1] + j - output_lower[1]
                    for k in range(output_lower[2], output_upper[2]):
                        dk = destination_lower[2] + k - output_lower[2]

                        bz_yp = source[slot, bz, i, j + 1, k]
                        bz_ym = source[slot, bz, i, j - 1, k]
                        by_zp = source[slot, by, i, j, k + 1]
                        by_zm = source[slot, by, i, j, k - 1]
                        positive = _centered_derivative(bz_yp, bz_ym, inv_y)
                        negative = _centered_derivative(by_zp, by_zm, inv_z)
                        destination[slot, jx, di, dj, dk] = _scaled_difference(
                            positive,
                            negative,
                        )

                        bx_zp = source[slot, bx, i, j, k + 1]
                        bx_zm = source[slot, bx, i, j, k - 1]
                        bz_xp = source[slot, bz, i + 1, j, k]
                        bz_xm = source[slot, bz, i - 1, j, k]
                        positive = _centered_derivative(bx_zp, bx_zm, inv_z)
                        negative = _centered_derivative(bz_xp, bz_xm, inv_x)
                        destination[slot, jy, di, dj, dk] = _scaled_difference(
                            positive,
                            negative,
                        )

                        by_xp = source[slot, by, i + 1, j, k]
                        by_xm = source[slot, by, i - 1, j, k]
                        bx_yp = source[slot, bx, i, j + 1, k]
                        bx_ym = source[slot, bx, i, j - 1, k]
                        positive = _centered_derivative(by_xp, by_xm, inv_x)
                        negative = _centered_derivative(bx_yp, bx_ym, inv_y)
                        destination[slot, jz, di, dj, dk] = _scaled_difference(
                            positive,
                            negative,
                        )
