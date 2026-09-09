"""PRL-001 fine-output-centric ratio-two limited prolongation."""

import cython

from libc.stdint cimport int64_t

from .limiter_core cimport three_point_limited_slope_rule


cdef inline int64_t _floor_divide_two(int64_t value) noexcept nogil:
    if value >= 0:
        return value >> 1
    return -1 - ((-1 - value) >> 1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void prolong_cartesian_2to1_into_fine_centric_unchecked(
    const double[:, :, :, :, ::1] coarse_payload,
    const int64_t[::1] coarse_origin,
    double[:, :, :, :, ::1] fine_payload,
    const int64_t[::1] fine_lower,
    const int64_t[::1] fine_upper,
    const int64_t[::1] fine_origin,
):
    cdef int64_t slot, field, i, j, k, I, J, K
    cdef int64_t ri, rj, rk, qi, qj, qk
    cdef int64_t phase_i, phase_j, phase_k
    cdef double eta_x, eta_y, eta_z
    cdef double center, slope_x, slope_y, slope_z
    cdef double term_x, term_y, term_z, value

    with nogil:
        for slot in range(fine_payload.shape[0]):
            for field in range(fine_payload.shape[1]):
                for i in range(fine_lower[0], fine_upper[0]):
                    ri = i - fine_origin[0]
                    qi = _floor_divide_two(ri)
                    phase_i = ri - 2 * qi
                    I = coarse_origin[0] + qi
                    eta_x = -0.25 if phase_i == 0 else 0.25
                    for j in range(fine_lower[1], fine_upper[1]):
                        rj = j - fine_origin[1]
                        qj = _floor_divide_two(rj)
                        phase_j = rj - 2 * qj
                        J = coarse_origin[1] + qj
                        eta_y = -0.25 if phase_j == 0 else 0.25
                        for k in range(fine_lower[2], fine_upper[2]):
                            rk = k - fine_origin[2]
                            qk = _floor_divide_two(rk)
                            phase_k = rk - 2 * qk
                            K = coarse_origin[2] + qk
                            eta_z = -0.25 if phase_k == 0 else 0.25

                            center = coarse_payload[slot, field, I, J, K]
                            slope_x = three_point_limited_slope_rule(
                                coarse_payload[slot, field, I - 1, J, K],
                                center,
                                coarse_payload[slot, field, I + 1, J, K],
                            )
                            slope_y = three_point_limited_slope_rule(
                                coarse_payload[slot, field, I, J - 1, K],
                                center,
                                coarse_payload[slot, field, I, J + 1, K],
                            )
                            slope_z = three_point_limited_slope_rule(
                                coarse_payload[slot, field, I, J, K - 1],
                                center,
                                coarse_payload[slot, field, I, J, K + 1],
                            )
                            term_x = slope_x * eta_x
                            term_y = slope_y * eta_y
                            term_z = slope_z * eta_z
                            value = center
                            value = value + term_x
                            value = value + term_y
                            value = value + term_z
                            fine_payload[slot, field, i, j, k] = value


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void prolong_cartesian_2to1_into_unchecked(
    const double[:, :, :, :, ::1] coarse_payload,
    const int64_t[::1] coarse_origin,
    double[:, :, :, :, ::1] fine_payload,
    const int64_t[::1] fine_lower,
    const int64_t[::1] fine_upper,
    const int64_t[::1] fine_origin,
):
    cdef int64_t slot, field, i, j, k, I, J, K
    cdef int64_t first_I, last_I, first_J, last_J, first_K, last_K
    cdef int64_t base_i, base_j, base_k
    cdef int64_t phase_i, phase_j, phase_k
    cdef double eta_x, eta_y, eta_z
    cdef double center, slope_x, slope_y, slope_z
    cdef double term_x, term_y, term_z, value

    if (
        fine_lower[0] == fine_upper[0]
        or fine_lower[1] == fine_upper[1]
        or fine_lower[2] == fine_upper[2]
    ):
        return

    first_I = coarse_origin[0] + _floor_divide_two(
        fine_lower[0] - fine_origin[0]
    )
    last_I = coarse_origin[0] + _floor_divide_two(
        fine_upper[0] - 1 - fine_origin[0]
    )
    first_J = coarse_origin[1] + _floor_divide_two(
        fine_lower[1] - fine_origin[1]
    )
    last_J = coarse_origin[1] + _floor_divide_two(
        fine_upper[1] - 1 - fine_origin[1]
    )
    first_K = coarse_origin[2] + _floor_divide_two(
        fine_lower[2] - fine_origin[2]
    )
    last_K = coarse_origin[2] + _floor_divide_two(
        fine_upper[2] - 1 - fine_origin[2]
    )

    with nogil:
        for slot in range(fine_payload.shape[0]):
            for field in range(fine_payload.shape[1]):
                for I in range(first_I, last_I + 1):
                    base_i = fine_origin[0] + 2 * (
                        I - coarse_origin[0]
                    )
                    for J in range(first_J, last_J + 1):
                        base_j = fine_origin[1] + 2 * (
                            J - coarse_origin[1]
                        )
                        for K in range(first_K, last_K + 1):
                            base_k = fine_origin[2] + 2 * (
                                K - coarse_origin[2]
                            )
                            center = coarse_payload[slot, field, I, J, K]
                            slope_x = three_point_limited_slope_rule(
                                coarse_payload[slot, field, I - 1, J, K],
                                center,
                                coarse_payload[slot, field, I + 1, J, K],
                            )
                            slope_y = three_point_limited_slope_rule(
                                coarse_payload[slot, field, I, J - 1, K],
                                center,
                                coarse_payload[slot, field, I, J + 1, K],
                            )
                            slope_z = three_point_limited_slope_rule(
                                coarse_payload[slot, field, I, J, K - 1],
                                center,
                                coarse_payload[slot, field, I, J, K + 1],
                            )

                            for phase_i in range(2):
                                i = base_i + phase_i
                                if i < fine_lower[0] or i >= fine_upper[0]:
                                    continue
                                eta_x = -0.25 if phase_i == 0 else 0.25
                                for phase_j in range(2):
                                    j = base_j + phase_j
                                    if j < fine_lower[1] or j >= fine_upper[1]:
                                        continue
                                    eta_y = -0.25 if phase_j == 0 else 0.25
                                    for phase_k in range(2):
                                        k = base_k + phase_k
                                        if k < fine_lower[2] or k >= fine_upper[2]:
                                            continue
                                        eta_z = (
                                            -0.25 if phase_k == 0 else 0.25
                                        )
                                        term_x = slope_x * eta_x
                                        term_y = slope_y * eta_y
                                        term_z = slope_z * eta_z
                                        value = center
                                        value = value + term_x
                                        value = value + term_y
                                        value = value + term_z
                                        fine_payload[
                                            slot, field, i, j, k
                                        ] = value
