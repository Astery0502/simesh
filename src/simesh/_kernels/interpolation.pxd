"""Reconstruction arithmetic for caller-selected samples and local coordinates."""


cdef inline double trilinear(
    double v000, double v100, double v001, double v101,
    double v010, double v110, double v011, double v111,
    const double* t,
) noexcept nogil:
    # Keep the established x, then y, then z operation tree in both layouts.
    cdef double v00 = v000*(1-t[0]) + v100*t[0]
    cdef double v01 = v001*(1-t[0]) + v101*t[0]
    cdef double v10 = v010*(1-t[0]) + v110*t[0]
    cdef double v11 = v011*(1-t[0]) + v111*t[0]
    cdef double v0 = v00*(1-t[1]) + v10*t[1]
    cdef double v1 = v01*(1-t[1]) + v11*t[1]
    return v0*(1-t[2]) + v1*t[2]
