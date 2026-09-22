from libc.math cimport fmin
from libc.stdint cimport int64_t


cdef inline double cell_width(const double[:, ::1] spacing, int64_t leaf) noexcept nogil:
    return fmin(spacing[leaf,0], fmin(spacing[leaf,1], spacing[leaf,2]))


cdef inline double trace_step(double width, double fraction, double cap,
                              double remaining) noexcept nogil:
    cdef double step = fmin(cap, remaining)
    return fmin(step, fraction*width) if fraction > 0. else step


cdef inline bint finer_step(double step, double width, double fraction) noexcept nogil:
    return fraction > 0. and step > fraction*width*(1.+1.e-12)
