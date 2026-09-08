from libc.stdint cimport int64_t
from libc.math cimport floor, isfinite

# Shared inline interpolation; interval ownership remains in native.pyx.
cdef inline bint interpolate(
    const double* p, int64_t leaf, int64_t slot,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, int halo, double* out,
) noexcept nogil:
    cdef int a, c, x, y, z
    cdef int64_t base[3]
    cdef double t[3]
    cdef double q, v00, v01, v10, v11, v0, v1
    for a in range(3):
        q = (p[a]-bounds[leaf,0,a])/spacing[leaf,a] - .5
        if not isfinite(q) or q < -halo or q >= data.shape[a+1]-halo-1:
            return False
        base[a] = <int64_t>floor(q)
        t[a] = q-base[a]
        base[a] += halo
    x,y,z = base[0],base[1],base[2]
    for c in range(data.shape[4]):
        v00 = data[slot,x,y,z,c]*(1-t[0]) + data[slot,x+1,y,z,c]*t[0]
        v01 = data[slot,x,y,z+1,c]*(1-t[0]) + data[slot,x+1,y,z+1,c]*t[0]
        v10 = data[slot,x,y+1,z,c]*(1-t[0]) + data[slot,x+1,y+1,z,c]*t[0]
        v11 = data[slot,x,y+1,z+1,c]*(1-t[0]) + data[slot,x+1,y+1,z+1,c]*t[0]
        v0 = v00*(1-t[1]) + v10*t[1]
        v1 = v01*(1-t[1]) + v11*t[1]
        out[c] = v0*(1-t[2]) + v1*t[2]
    return True


cdef int64_t ray_owner(
    const double* origin, const double[::1] direction, double t,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo, const double[:, ::1] nhi,
) noexcept nogil
