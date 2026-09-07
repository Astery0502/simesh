# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Allocation-free compiled geometry and direct prepared-field consumption.

Unchecked internal functions receive arrays validated by the provider/core.
No file, cache, Dataset or Python callback participates in a value query.
"""

from libc.math cimport floor, isfinite, NAN, sqrt, hypot
from libc.stdint cimport int64_t


cdef inline int64_t owner(
    const double* p, const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi,
) noexcept nogil:
    cdef int a, bit
    cdef int64_t r[3]
    cdef int64_t node, child
    for a in range(3):
        if not isfinite(p[a]) or p[a] < lo[a] or p[a] >= hi[a]:
            return -1
        r[a] = <int64_t>floor((p[a]-lo[a])/(hi[a]-lo[a])*roots.shape[a])
        if r[a] == roots.shape[a]:
            r[a] -= 1
    node = roots[r[0], r[1], r[2]]
    # Correct division rounding at an internal root face using stored geometry.
    for a in range(3):
        if p[a] < nlo[node,a] and r[a] > 0:
            r[a] -= 1
        elif p[a] >= nhi[node,a] and r[a]+1 < roots.shape[a]:
            r[a] += 1
        node = roots[r[0],r[1],r[2]]
    while leaves[node] < 0:
        child = children[node,0]
        bit = 0
        for a in range(3):
            if p[a] >= nhi[child,a]:
                bit |= 1 << a
        node = children[node,bit]
    return leaves[node]


cpdef void locate(
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi, const double[:, ::1] points,
    int64_t[::1] owners,
):
    cdef Py_ssize_t i
    with nogil:
        for i in range(points.shape[0]):
            owners[i] = owner(&points[i,0],lo,hi,roots,children,leaves,nlo,nhi)


cdef inline void interpolate(
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


cpdef void sample_ready(
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi, const double[:, :, ::1] bounds,
    const double[:, ::1] spacing, const int64_t[::1] slots,
    const double[:, :, :, :, ::1] data, int halo,
    const double[:, ::1] points, double[:, ::1] output,
    int64_t[::1] owners, unsigned char[::1] valid,
):
    cdef Py_ssize_t i, c
    cdef int64_t leaf, slot
    with nogil:
        for i in range(points.shape[0]):
            leaf = owner(&points[i,0],lo,hi,roots,children,leaves,nlo,nhi)
            owners[i] = leaf
            slot = slots[leaf] if leaf >= 0 else -1
            valid[i] = 1 if slot >= 0 else 0
            if slot < 0:
                for c in range(data.shape[4]):
                    output[i,c] = NAN
            else:
                interpolate(&points[i,0],leaf,slot,bounds,spacing,data,halo,&output[i,0])


cpdef void differentiate(
    const double[:, :, :, :, ::1] data, const int64_t[::1] slots,
    const int64_t[::1] ids, const double[:, ::1] spacing,
    const int64_t[:, ::1] terms, const double[::1] coefficients,
    double[:, :, :, :, ::1] output,
):
    cdef int64_t s, slot, leaf, i,j,k,c,t,axis,component,out
    cdef int x,y,z
    cdef double delta
    with nogil:
        for s in range(output.shape[0]):
            leaf = ids[s]
            slot = slots[leaf]
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    for k in range(output.shape[3]):
                        for c in range(output.shape[4]):
                            output[s,i,j,k,c] = 0.
                        for t in range(terms.shape[0]):
                            out,component,axis = terms[t,0],terms[t,1],terms[t,2]
                            x = 1 if axis == 0 else 0
                            y = 1 if axis == 1 else 0
                            z = 1 if axis == 2 else 0
                            delta = (data[slot,i+1+x,j+1+y,k+1+z,component] -
                                     data[slot,i+1-x,j+1-y,k+1-z,component])/(2.*spacing[leaf,axis])
                            output[s,i,j,k,out] = output[s,i,j,k,out] + coefficients[t]*delta


cpdef void advance_lines(
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi, const double[:, :, ::1] bounds,
    const double[:, ::1] spacing, const int64_t[::1] slots,
    const double[:, :, :, :, ::1] data, int halo,
    double step, int64_t max_steps, double max_length, double null_threshold, int direction,
    double[:, ::1] positions, double[::1] length, int64_t[::1] steps,
    int64_t[::1] status, int64_t[::1] stages, double[:, :, ::1] tangents,
    int64_t[::1] requested, int64_t[::1] samples, int64_t[::1] misses,
):
    cdef int64_t seed, stage, a, leaf, slot
    cdef double p[3]
    cdef double b[3]
    cdef double h, norm, factor, total
    with nogil:
        for seed in range(positions.shape[0]):
            requested[seed] = -1
            while status[seed] == 0:
                if steps[seed] >= max_steps:
                    status[seed] = 1
                    break
                if length[seed] >= max_length:
                    status[seed] = 2
                    break
                h = step
                if max_length-length[seed] < h:
                    h = max_length-length[seed]
                stage = stages[seed]
                for a in range(3):
                    p[a] = positions[seed,a]
                    if stage > 0:
                        factor = .5 if stage < 3 else 1.
                        p[a] = p[a] + factor*h*tangents[seed,stage-1,a]
                leaf = owner(p,lo,hi,roots,children,leaves,nlo,nhi)
                if leaf < 0:
                    status[seed] = 3
                    break
                slot = slots[leaf]
                if slot < 0:
                    requested[seed] = leaf
                    misses[seed] += 1
                    break
                interpolate(p,leaf,slot,bounds,spacing,data,halo,b)
                samples[seed] += 1
                if not (isfinite(b[0]) and isfinite(b[1]) and isfinite(b[2])):
                    status[seed] = 5
                    break
                norm = hypot(hypot(b[0],b[1]),b[2])
                if not isfinite(norm):
                    status[seed] = 8
                    break
                if norm <= null_threshold:
                    status[seed] = 4
                    break
                for a in range(3):
                    tangents[seed,stage,a] = direction*(b[a]/norm)
                if stage < 3:
                    stages[seed] += 1
                    continue
                for a in range(3):
                    total = tangents[seed,0,a] + 2.*tangents[seed,1,a]
                    total = total + 2.*tangents[seed,2,a]
                    total = total + tangents[seed,3,a]
                    p[a] = positions[seed,a] + (h/6.)*total
                if owner(p,lo,hi,roots,children,leaves,nlo,nhi) < 0:
                    status[seed] = 3
                    break
                for a in range(3):
                    positions[seed,a] = p[a]
                length[seed] = length[seed]+h
                steps[seed] += 1
                stages[seed] = 0
