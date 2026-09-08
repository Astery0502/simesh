# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""AMR interval traversal and nonlinear thermal quadrature, without Python calls.

Each row owns its entire accumulation and stack state. Shared arrays are readonly;
the Python coordinator may dispatch disjoint rows to GIL-free calls. The scalar
native locator/interpolator is reused through its Cython declaration boundary.
"""
from libc.math cimport floor, ceil, isfinite, log10, pow, nextafter, INFINITY, NAN
from libc.stdint cimport int64_t
from cython.parallel cimport prange
from .native cimport interpolate, ray_owner


cdef extern from *:
    """
    #ifdef _OPENMP
    #define SIMESH_ANALYSIS_OPENMP _OPENMP
    #else
    #define SIMESH_ANALYSIS_OPENMP 0
    #endif
    """
    int SIMESH_ANALYSIS_OPENMP


def openmp_build_info():
    return {"enabled": SIMESH_ANALYSIS_OPENMP != 0,
            "openmp_version": SIMESH_ANALYSIS_OPENMP}



cdef inline double response(double temperature, const double[::1] grid,
                            const double[::1] log_response,
                            const double[::1] slopes) noexcept nogil:
    cdef double t = log10(temperature)
    cdef int i
    if t < grid[0] or t > grid[100]:
        return 0.
    if t == grid[100]:
        return pow(10., log_response[100])
    # Uniform table index with exact stored-knot correction at roundoff ties.
    i = <int>floor((t-4.)*20.)
    if i < 0:
        i = 0
    if i > 99:
        i = 99
    while i > 0 and t < grid[i]:
        i -= 1
    while i < 99 and t >= grid[i+1]:
        i += 1
    return pow(10., log_response[i]+(t-grid[i])*slopes[i])


cdef int thermal_leaf(
    const double* origin, const double[::1] direction, int64_t leaf, int64_t slot,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, int halo, double first, double last,
    int64_t subdivisions, int64_t limit, const double[::1] grid,
    const double[::1] log_response, const double[::1] slopes,
    int64_t* samples, double* value,
) noexcept nogil:
    cdef int a, which
    cdef int sign[3]
    cdef int64_t index[3]
    cdef int64_t extent[3]
    cdef int64_t sub
    cdef double knots[3]
    cdef double p[3]
    cdef double upper[3]
    cdef double thermal[2]
    cdef double q, cursor, stop, width, start, t, weight, epsilon
    cursor = first
    for a in range(3):
        upper[a] = nextafter(bounds[leaf,1,a],bounds[leaf,0,a])
        extent[a] = data.shape[a+1]-2*halo
        knots[a] = INFINITY
        index[a] = 0
        sign[a] = 0
        if direction[a] == 0.:
            continue
        q = ((origin[a]+direction[a]*cursor)-bounds[leaf,0,a])/spacing[leaf,a]-.5
        if not isfinite(q) or q < -halo or q >= data.shape[a+1]-halo-1:
            return 8
        sign[a] = 1 if direction[a]>0. else -1
        index[a] = <int64_t>floor(q)+1 if sign[a]>0 else <int64_t>ceil(q)-1
        while 0 <= index[a] < extent[a]:
            t = (bounds[leaf,0,a]+(index[a]+.5)*spacing[leaf,a]-origin[a])/direction[a]
            if t > cursor:
                knots[a] = t
                break
            index[a] += sign[a]
    while cursor < last:
        stop = last
        for a in range(3):
            if knots[a] < stop:
                stop = knots[a]
        if stop <= cursor:
            return 5
        if subdivisions > (limit-samples[0])//2:
            return 6
        width = (stop-cursor)/subdivisions
        weight = width/2.
        for sub in range(subdivisions):
            start = cursor+width*sub
            for which in range(2):
                # Keep the reference ray_nodes expression and full tiny weights.
                t = start+weight*(1.+(-1. if which==0 else 1.)/1.7320508075688772)
                for a in range(3):
                    p[a] = origin[a]+direction[a]*t
                    if p[a] < bounds[leaf,0,a]:
                        p[a] = bounds[leaf,0,a]
                    if p[a] > upper[a]:
                        p[a] = upper[a]
                if not interpolate(p,leaf,slot,bounds,spacing,data,halo,thermal):
                    return 8
                if not isfinite(thermal[0]) or not isfinite(thermal[1]):
                    return 4
                if thermal[0] < 0. or thermal[1] <= 0.:
                    return 4
                epsilon = thermal[0]*thermal[0]*response(thermal[1],grid,log_response,slopes)
                if not isfinite(epsilon):
                    return 4
                samples[0] += 1
                value[0] = value[0]+epsilon*weight
                if not isfinite(value[0]):
                    return 7
        cursor = stop
        for a in range(3):
            if knots[a] <= cursor:
                knots[a] = INFINITY
                index[a] += sign[a]
                while 0 <= index[a] < extent[a]:
                    t = (bounds[leaf,0,a]+(index[a]+.5)*spacing[leaf,a]-origin[a])/direction[a]
                    if t > cursor:
                        knots[a] = t
                        break
                    index[a] += sign[a]
    return 0


cdef void _integrate_range(int64_t begin, int64_t end,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo, const double[:, ::1] nhi,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const int64_t[::1] slots, const double[:, :, :, :, ::1] data, int halo,
    const double[:, ::1] origins, const double[::1] direction,
    const double[::1] first, const double[::1] last,
    int64_t subdivisions, int64_t limit,
    const double[::1] grid, const double[::1] log_response, const double[::1] slopes,
    double[::1] values, int64_t[::1] status, int64_t[::1] samples,
) noexcept nogil:
    cdef int64_t ray, leaf, slot, a
    cdef int code
    cdef double cursor, stop, edge, face
    for ray in range(begin,end):
        if status[ray] != 0:
            continue
        cursor = first[ray]
        while cursor < last[ray]:
            leaf = ray_owner(&origins[ray,0],direction,cursor,roots,children,leaves,nlo,nhi)
            stop = last[ray]
            for a in range(3):
                if direction[a] != 0.:
                    face = bounds[leaf,1,a] if direction[a]>0. else bounds[leaf,0,a]
                    edge = (face-origins[ray,a])/direction[a]
                    if edge < stop:
                        stop = edge
            if stop <= cursor:
                status[ray] = 5
                break
            slot = slots[leaf]
            if slot < 0:
                status[ray] = 3
                break
            code = thermal_leaf(&origins[ray,0],direction,leaf,slot,bounds,spacing,
                data,halo,cursor,stop,subdivisions,limit,grid,log_response,slopes,
                &samples[ray],&values[ray])
            if code:
                status[ray] = code
                break
            cursor = stop
        if status[ray] == 0:
            status[ray] = 1
        if status[ray] >= 3:
            values[ray] = NAN


cpdef void integrate_ready(
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo, const double[:, ::1] nhi,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const int64_t[::1] slots, const double[:, :, :, :, ::1] data, int halo,
    const double[:, ::1] origins, const double[::1] direction,
    const double[::1] first, const double[::1] last,
    int64_t subdivisions, int64_t limit,
    const double[::1] grid, const double[::1] log_response, const double[::1] slopes,
    double[::1] values, int64_t[::1] status, int64_t[::1] samples,
    int workers=1, int dispatch=1,
):
    cdef int64_t task, begin, end, count = origins.shape[0]
    if workers < 1 or dispatch not in (0,1):
        raise ValueError("invalid thermal worker or schedule settings")
    if workers > 1 and not SIMESH_ANALYSIS_OPENMP:
        raise RuntimeError("thermal OpenMP requires an OpenMP-enabled analysis build")
    with nogil:
        if workers == 1:
            _integrate_range(0,count,roots,children,leaves,nlo,nhi,bounds,spacing,slots,data,halo,origins,direction,first,last,subdivisions,limit,grid,log_response,slopes,values,status,samples)
        elif dispatch == 0:
            for task in prange(workers, schedule='static', num_threads=workers):
                begin = count*task//workers
                end = count*(task+1)//workers
                _integrate_range(begin,end,roots,children,leaves,nlo,nhi,bounds,spacing,slots,data,halo,origins,direction,first,last,subdivisions,limit,grid,log_response,slopes,values,status,samples)
        else:
            # Small coherent ray ranges balance variable AMR depth and traversal.
            for task in prange((count+31)//32, schedule='dynamic', chunksize=1, num_threads=workers):
                begin = task*32
                end = min(begin+32,count)
                _integrate_range(begin,end,roots,children,leaves,nlo,nhi,bounds,spacing,slots,data,halo,origins,direction,first,last,subdivisions,limit,grid,log_response,slopes,values,status,samples)
