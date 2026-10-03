from libc.stdint cimport int64_t
from libc.math cimport floor, isfinite
from .interpolation cimport trilinear

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
    return trilinear(data[slot,x,y,z,c], data[slot,x+1,y,z,c],
                     data[slot,x,y,z+1,c], data[slot,x+1,y,z+1,c],
                     data[slot,x,y+1,z,c], data[slot,x+1,y+1,z,c],
                     data[slot,x,y+1,z+1,c], data[slot,x+1,y+1,z+1,c], t)


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


cdef inline void interpolate_unit_gradient(
    int64_t leaf, int64_t slot, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, const int64_t* base, const double* t,
    const double* b, double norm, double* gradient,
) noexcept nogil:
    # Differentiate the same local trilinear vector used by the trajectory.
    # The caller supplies its checked sampling stencil and nonzero vector norm.
    cdef int64_t x, y, z
    cdef int c, a
    cdef double unit[3]
    cdef double v000, v100, v010, v110, v001, v101, v011, v111
    cdef double v00, v10, v01, v11, projection
    x, y, z = base[0], base[1], base[2]
    for c in range(3):
        unit[c] = b[c]/norm
        # Normalize before differencing to avoid overflow for large vectors.
        v000 = data[slot,x,y,z,c]/norm
        v100 = data[slot,x+1,y,z,c]/norm
        v010 = data[slot,x,y+1,z,c]/norm
        v110 = data[slot,x+1,y+1,z,c]/norm
        v001 = data[slot,x,y,z+1,c]/norm
        v101 = data[slot,x+1,y,z+1,c]/norm
        v011 = data[slot,x,y+1,z+1,c]/norm
        v111 = data[slot,x+1,y+1,z+1,c]/norm
        gradient[3*c] = (((v100-v000)*(1-t[1])+(v110-v010)*t[1])*(1-t[2]) +
                         ((v101-v001)*(1-t[1])+(v111-v011)*t[1])*t[2])/spacing[leaf,0]
        v00 = v000*(1-t[0])+v100*t[0]
        v10 = v010*(1-t[0])+v110*t[0]
        v01 = v001*(1-t[0])+v101*t[0]
        v11 = v011*(1-t[0])+v111*t[0]
        gradient[3*c+1] = ((v10-v00)*(1-t[2])+(v11-v01)*t[2])/spacing[leaf,1]
        gradient[3*c+2] = ((v01-v00)*(1-t[1])+(v11-v10)*t[1])/spacing[leaf,2]
    # Chain rule: D(B/|B|) = (I - b_hat b_hat^T) DB / |B|.
    for a in range(3):
        projection = 0.
        for c in range(3):
            projection += unit[c]*gradient[3*c+a]
        for c in range(3):
            gradient[3*c+a] -= unit[c]*projection


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
