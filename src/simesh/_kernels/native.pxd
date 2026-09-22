from libc.stdint cimport int64_t
from libc.math cimport floor, isfinite

# Interval ownership remains in native.pyx; all consumers share this stencil.
cdef inline bint interpolation_stencil(
    const double* p, int64_t leaf,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, int halo, int64_t* base, double* t,
) noexcept nogil:
    cdef int a
    cdef double q
    for a in range(3):
        q = (p[a]-bounds[leaf,0,a])/spacing[leaf,a] - .5
        if not isfinite(q) or q < -halo or q >= data.shape[a+1]-halo-1:
            return False
        base[a] = <int64_t>floor(q)
        t[a] = q-base[a]
        base[a] += halo
    return True


cdef inline double interpolate_component(
    int64_t slot, int64_t c, const double[:, :, :, :, ::1] data,
    const int64_t* base, const double* t,
) noexcept nogil:
    cdef int64_t x = base[0], y = base[1], z = base[2]
    cdef double v00, v01, v10, v11, v0, v1
    v00 = data[slot,x,y,z,c]*(1-t[0]) + data[slot,x+1,y,z,c]*t[0]
    v01 = data[slot,x,y,z+1,c]*(1-t[0]) + data[slot,x+1,y,z+1,c]*t[0]
    v10 = data[slot,x,y+1,z,c]*(1-t[0]) + data[slot,x+1,y+1,z,c]*t[0]
    v11 = data[slot,x,y+1,z+1,c]*(1-t[0]) + data[slot,x+1,y+1,z+1,c]*t[0]
    v0 = v00*(1-t[1]) + v10*t[1]
    v1 = v01*(1-t[1]) + v11*t[1]
    return v0*(1-t[2]) + v1*t[2]


cdef inline bint interpolate(
    const double* p, int64_t leaf, int64_t slot,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, int halo, double* out,
) noexcept nogil:
    cdef int64_t base[3], c
    cdef double t[3]
    if not interpolation_stencil(p,leaf,bounds,spacing,data,halo,base,t):
        return False
    for c in range(data.shape[4]):
        out[c] = interpolate_component(slot,c,data,base,t)
    return True


cdef int64_t ray_owner(
    const double* origin, const double[::1] direction, double t,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo, const double[:, ::1] nhi,
) noexcept nogil


cdef int64_t owner(
    const double* p, const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi,
) noexcept nogil


cdef int64_t owner_node(
    const double* p, const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi,
) noexcept nogil

cdef bint contains_point(const double* p, const double* lo,
                         const double* hi) noexcept nogil
