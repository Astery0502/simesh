"""Resumable classical RK4 for caller-owned state and compiled stage functions.

No mesh, field, diagnostic, scheduler or storage owner is imported here.
The caller supplies disjoint buffers and commits a candidate only after its
admission policy succeeds. A stage function may shrink h and return RK_RETRY,
or return a positive stop/pause code without losing completed stages.
"""

from libc.stdint cimport int64_t
from libc.math cimport isfinite

cdef enum:
    RK_RETRY = -1

ctypedef int (*StageFunction)(void* context, const double* state,
                              double* derivative, double* h,
                              int stage) noexcept nogil


cdef inline int rk4_trial(void* context, StageFunction evaluate,
                           const double* accepted, int size,
                           double* h, int64_t* stage, double* slopes,
                           double* scratch, double* candidate) noexcept nogil:
    cdef int j, code
    cdef double factor, total, previous_h
    while stage[0] < 4:
        factor = .5 if stage[0] < 3 else 1.
        for j in range(size):
            scratch[j] = accepted[j]
            if stage[0] > 0:
                scratch[j] += factor*h[0]*slopes[(stage[0]-1)*size+j]
        previous_h = h[0]
        code = evaluate(context, scratch, slopes+stage[0]*size, h, <int>stage[0])
        if code == RK_RETRY:
            if not isfinite(h[0]) or not 0. < h[0] < previous_h:
                return 11
            stage[0] = 0
            continue
        if code:
            return code
        stage[0] += 1
    for j in range(size):
        total = slopes[j]+2.*slopes[size+j]
        total = total+2.*slopes[2*size+j]
        total = total+slopes[3*size+j]
        candidate[j] = accepted[j]+(h[0]/6.)*total
        if not isfinite(candidate[j]):
            return 9
    return 0
