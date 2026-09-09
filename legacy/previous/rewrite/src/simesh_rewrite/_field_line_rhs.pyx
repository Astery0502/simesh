# cython: boundscheck=False, wraparound=False

"""Scale-first normalized magnetic field-line RHS for FLN-001."""

import cython

from libc.math cimport fabs, isfinite, sqrt
from libc.stdint cimport int8_t, int64_t, uint8_t


cdef uint8_t _RHS_OK = 0
cdef uint8_t _RHS_ZERO_FIELD = 1
cdef uint8_t _RHS_NONFINITE_FIELD = 2
cdef uint8_t _RHS_UNREPRESENTABLE_NORM = 3


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void field_line_rhs_unchecked(
    const double[:, ::1] field_vectors,
    const int8_t[::1] direction_signs,
    double[:, ::1] rhs_values,
    uint8_t[::1] rhs_statuses,
):
    """Apply the frozen FLN operation tree after complete outer preflight."""
    cdef int64_t row
    cdef volatile double bx, by, bz
    cdef volatile double ax, ay, az, maximum
    cdef volatile double ux, uy, uz
    cdef volatile double xx, yy, zz, xy, xyz
    cdef volatile double radius, magnitude
    cdef volatile double tx, ty, tz, sign
    cdef volatile double rhs_x, rhs_y, rhs_z, rhs_integral

    with nogil:
        for row in range(field_vectors.shape[0]):
            bx = field_vectors[row, 0]
            by = field_vectors[row, 1]
            bz = field_vectors[row, 2]
            if not isfinite(bx) or not isfinite(by) or not isfinite(bz):
                rhs_statuses[row] = _RHS_NONFINITE_FIELD
                continue

            ax = fabs(bx)
            ay = fabs(by)
            az = fabs(bz)
            maximum = ax
            if ay > maximum:
                maximum = ay
            if az > maximum:
                maximum = az
            if maximum == 0.0:
                rhs_statuses[row] = _RHS_ZERO_FIELD
                continue

            ux = bx / maximum
            uy = by / maximum
            uz = bz / maximum
            xx = ux * ux
            yy = uy * uy
            zz = uz * uz
            xy = xx + yy
            xyz = xy + zz
            radius = sqrt(xyz)
            magnitude = maximum * radius
            if not isfinite(magnitude):
                rhs_statuses[row] = _RHS_UNREPRESENTABLE_NORM
                continue

            tx = ux / radius
            ty = uy / radius
            tz = uz / radius
            sign = <double>direction_signs[row]
            rhs_x = sign * tx
            rhs_y = sign * ty
            rhs_z = sign * tz
            rhs_integral = sign * magnitude
            rhs_values[row, 0] = rhs_x
            rhs_values[row, 1] = rhs_y
            rhs_values[row, 2] = rhs_z
            rhs_values[row, 3] = rhs_integral
            rhs_statuses[row] = _RHS_OK
