# cython: boundscheck=False, wraparound=False

"""Fixed classical RK4 arithmetic kernels for RKS-001."""

import cython
from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void field_line_rk4_stage_unchecked(
    const double[:, ::1] base_states,
    const double[:, ::1] previous_rhs,
    double step_size,
    bint half_step,
    double[:, ::1] stage_states,
):
    cdef int64_t row, component
    cdef volatile double half = 0.5 * step_size
    cdef volatile double scale
    cdef volatile double delta
    cdef volatile double result

    if half_step:
        scale = half
    else:
        scale = step_size
    with nogil:
        for row in range(base_states.shape[0]):
            for component in range(4):
                delta = scale * previous_rhs[row, component]
                result = base_states[row, component] + delta
                stage_states[row, component] = result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void field_line_rk4_finish_unchecked(
    const double[:, ::1] base_states,
    const double[:, ::1] k1,
    const double[:, ::1] k2,
    const double[:, ::1] k3,
    const double[:, ::1] k4,
    double step_size,
    double[:, ::1] candidate_states,
):
    cdef int64_t row, component
    cdef volatile double h6 = step_size / 6.0
    cdef volatile double two_k2
    cdef volatile double sum12
    cdef volatile double two_k3
    cdef volatile double sum123
    cdef volatile double sum1234
    cdef volatile double delta
    cdef volatile double result

    with nogil:
        for row in range(base_states.shape[0]):
            for component in range(4):
                two_k2 = 2.0 * k2[row, component]
                sum12 = k1[row, component] + two_k2
                two_k3 = 2.0 * k3[row, component]
                sum123 = sum12 + two_k3
                sum1234 = sum123 + k4[row, component]
                delta = h6 * sum1234
                result = base_states[row, component] + delta
                candidate_states[row, component] = result
