# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Allocation-free compiled geometry and direct prepared-field consumption.

Unchecked internal functions receive arrays validated by the provider/core.
No file, cache, Dataset or Python callback participates in a value query.
"""

from libc.math cimport floor, isfinite, NAN, sqrt, hypot, ceil, nextafter, INFINITY
from libc.stdint cimport int64_t
from cython.parallel cimport prange

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



cdef inline int64_t owner_node(
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
    return node


cdef inline int64_t owner(
    const double* p, const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi,
) noexcept nogil:
    cdef int64_t node = owner_node(p,lo,hi,roots,children,leaves,nlo,nhi)
    return leaves[node] if node >= 0 else -1


cdef inline bint contains_point(
    const double* p, const double* lo, const double* hi,
) noexcept nogil:
    # Ordered comparisons reject NaN and keep the exact half-open ownership.
    return (lo[0] <= p[0] < hi[0] and lo[1] <= p[1] < hi[1] and
            lo[2] <= p[2] < hi[2])


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


cpdef void sample_ready(
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi, const double[:, :, ::1] bounds,
    const double[:, ::1] spacing, const int64_t[::1] slots,
    const double[:, :, :, :, ::1] data, int halo,
    const double[:, ::1] points, double[:, :] output,
    int64_t[:] owners, unsigned char[:] valid,
    const int64_t[::1] field_positions,
):
    cdef Py_ssize_t i, column
    cdef int64_t base[3], leaf, slot
    cdef double t[3]
    with nogil:
        for i in range(points.shape[0]):
            leaf = owner(&points[i,0],lo,hi,roots,children,leaves,nlo,nhi)
            owners[i] = leaf
            slot = slots[leaf] if leaf >= 0 else -1
            valid[i] = 1 if slot >= 0 else 0
            if slot >= 0 and not interpolation_stencil(&points[i,0],leaf,bounds,spacing,data,halo,base,t):
                valid[i] = 0
            for column in range(field_positions.shape[0]):
                output[i,column] = (interpolate_component(slot,field_positions[column],data,base,t)
                                    if valid[i] else NAN)


cpdef void differentiate(
    const double[:, :, :, :, ::1] data, const int64_t[::1] slots,
    const int64_t[::1] ids, const double[:, ::1] spacing,
    const int64_t[:, ::1] terms, const double[::1] coefficients,
    double[:, :, :, :, ::1] output, int offset=1,
):
    """Same per-output operation tree; invariant term metadata stays outside cells."""
    cdef int64_t s, slot, leaf, i,j,k,c,t,axis,component,out
    cdef int x,y,z
    cdef double delta, denominator, coefficient
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
                denominator = 2.*spacing[leaf,axis]
                coefficient = coefficients[t]
                for i in range(output.shape[1]):
                    for j in range(output.shape[2]):
                        for k in range(output.shape[3]):
                            delta = (data[slot,i+offset+x,j+offset+y,k+offset+z,component] -
                                     data[slot,i+offset-x,j+offset-y,k+offset-z,component])/denominator
                            output[s,i,j,k,out] = output[s,i,j,k,out] + coefficient*delta


cdef void _advance_line_range(int64_t first, int64_t last,
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
    const int64_t[::1] curl_slots, const double[:, :, :, :, ::1] curl_data,
    int curl_halo, double[:, ::1] alpha, double[::1] twist,
    double[:, :, ::1] paths, int64_t path_start,
) noexcept nogil:
    cdef int64_t seed
    cdef int64_t  stage, a, leaf, slot, curl_slot, node_hint
    cdef double p[3]
    cdef double b[3]
    cdef double cb[3]
    cdef double h, norm, factor, total, integrand
    cdef bint want_twist = twist.shape[0] > 0
    cdef bint save_paths = paths.shape[0] > 0
    for seed in range(first,last):
        requested[seed] = -1
        node_hint = -1
        while status[seed] == 0:
            if steps[seed] >= max_steps:
                status[seed] = 1
                break
            if length[seed] >= max_length:
                status[seed] = 2
                break
            # A full path segment pauses delivery without terminating RK state.
            if save_paths and steps[seed]-path_start >= paths.shape[1]-1:
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
            # Most consecutive RK samples remain in one leaf. Reuse its exact
            # node box; only a crossing needs a root-to-leaf traversal.
            if node_hint < 0 or not contains_point(p,&nlo[node_hint,0],&nhi[node_hint,0]):
                node_hint = owner_node(p,lo,hi,roots,children,leaves,nlo,nhi)
            if node_hint < 0:
                status[seed] = 3
                break
            leaf = leaves[node_hint]
            slot = slots[leaf]
            if slot < 0:
                requested[seed] = leaf
                misses[seed] += 1
                break
            if not interpolate(p,leaf,slot,bounds,spacing,data,halo,b):
                status[seed] = 10
                break
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
            if want_twist:
                curl_slot = curl_slots[leaf]
                if curl_slot < 0:
                    requested[seed] = leaf
                    misses[seed] += 1
                    break
                if not interpolate(p,leaf,curl_slot,bounds,spacing,curl_data,curl_halo,cb):
                    status[seed] = 10
                    break
                # Algebraically curl(B).B / (4*pi*|B|^2), without squaring
                # a large norm. This ordering belongs to the named strategy.
                integrand = (cb[0]/norm)*(b[0]/norm)
                integrand = integrand + (cb[1]/norm)*(b[1]/norm)
                integrand = integrand + (cb[2]/norm)*(b[2]/norm)
                integrand = integrand / (4.*3.141592653589793)
                if not isfinite(integrand):
                    status[seed] = 9
                    break
                alpha[seed,stage] = integrand
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
            # The proposal only needs physical-domain admission. Its field
            # owner is resolved when the next accepted-step sample needs it.
            if not contains_point(p,&lo[0],&hi[0]):
                status[seed] = 3
                break
            for a in range(3):
                positions[seed,a] = p[a]
            length[seed] = length[seed]+h
            if want_twist:
                total = alpha[seed,0]+2.*alpha[seed,1]
                total = total+2.*alpha[seed,2]
                total = total+alpha[seed,3]
                twist[seed] = twist[seed]+(h/6.)*total
            steps[seed] += 1
            if save_paths:
                for a in range(3):
                    paths[seed,steps[seed]-path_start,a] = p[a]
            stages[seed] = 0




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
    const int64_t[::1] curl_slots, const double[:, :, :, :, ::1] curl_data,
    int curl_halo, double[:, ::1] alpha, double[::1] twist,
    double[:, :, ::1] paths,
    int workers=1, int dispatch=0, int64_t path_start=0,
):
    cdef int64_t task, first, last, count = positions.shape[0]
    if workers < 1 or dispatch not in (0,1):
        raise ValueError("invalid native worker/dispatch setting")
    if workers > 1 and not SIMESH_ANALYSIS_OPENMP:
        raise RuntimeError("native analysis was built without OpenMP")
    with nogil:
        if workers == 1:
            _advance_line_range(0, count, lo, hi, roots, children, leaves, nlo, nhi, bounds, spacing, slots, data, halo, step, max_steps, max_length, null_threshold, direction, positions, length, steps, status, stages, tangents, requested, samples, misses, curl_slots, curl_data, curl_halo, alpha, twist, paths, path_start)
        elif dispatch == 0:
            for task in prange(workers, schedule='static', num_threads=workers):
                first = count*task//workers
                last = count*(task+1)//workers
                _advance_line_range(first, last, lo, hi, roots, children, leaves, nlo, nhi, bounds, spacing, slots, data, halo, step, max_steps, max_length, null_threshold, direction, positions, length, steps, status, stages, tangents, requested, samples, misses, curl_slots, curl_data, curl_halo, alpha, twist, paths, path_start)
        else:
            for task in prange((count+7)//8, schedule='dynamic', chunksize=1, num_threads=workers):
                first = task*8
                last = first+8
                if last > count:
                    last = count
                _advance_line_range(first, last, lo, hi, roots, children, leaves, nlo, nhi, bounds, spacing, slots, data, halo, step, max_steps, max_length, null_threshold, direction, positions, length, steps, status, stages, tangents, requested, samples, misses, curl_slots, curl_data, curl_halo, alpha, twist, paths, path_start)



cdef inline bint interpolate_scalar(
    const double* p, int64_t leaf, int64_t slot, int64_t component,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, int halo, double* output,
) noexcept nogil:
    cdef int64_t base[3]
    cdef double t[3]
    if not interpolation_stencil(p,leaf,bounds,spacing,data,halo,base,t):
        return False
    output[0] = interpolate_component(slot,component,data,base,t)
    return True


cdef inline void _initialize_ray(
    const double[::1] lo, const double[::1] hi, const double* origin,
    const double* direction, double near, double far,
    double* start, double* end, int64_t* status,
) noexcept nogil:
    cdef int a
    cdef double first = near, last = far, left, right, temp
    cdef bint empty = False
    for a in range(3):
        if direction[a] == 0.:
            if origin[a] < lo[a] or origin[a] >= hi[a]:
                empty = True
            continue
        left = (lo[a]-origin[a])/direction[a]
        right = (hi[a]-origin[a])/direction[a]
        if left > right:
            temp = left
            left = right
            right = temp
        if left > first:
            first = left
        if right < last:
            last = right
    if empty or first >= last:
        start[0] = end[0] = 0.
        status[0] = 2
        return
    if not isfinite(first) or not isfinite(last):
        start[0] = end[0] = NAN
        status[0] = 5
        return
    start[0],end[0] = first,last
    status[0] = 0


cpdef void initialize_rays(
    const double[::1] lo, const double[::1] hi, const double[:, ::1] origins,
    const double[::1] direction, const double[::1] near, const double[::1] far,
    double[::1] starts, double[::1] ends,
    int64_t[::1] status,
):
    cdef int64_t ray
    with nogil:
        for ray in range(origins.shape[0]):
            _initialize_ray(lo,hi,&origins[ray,0],&direction[0],near[ray],far[ray],
                            &starts[ray],&ends[ray],&status[ray])


cdef inline bint ray_after_face(double origin,double direction,double t,double face) noexcept nogil:
    if direction > 0.:
        return t >= (face-origin)/direction
    if direction < 0.:
        return t < (face-origin)/direction
    return origin >= face


cpdef void initialize_ray_set(
    const double[::1] lo, const double[::1] hi, const double[:, ::1] origins,
    const double[:, ::1] directions, const double[::1] near, const double[::1] far,
    double[::1] starts, double[::1] ends, int64_t[::1] status,
):
    cdef int64_t ray
    with nogil:
        for ray in range(origins.shape[0]):
            _initialize_ray(lo,hi,&origins[ray,0],&directions[ray,0],near[ray],far[ray],
                            &starts[ray],&ends[ray],&status[ray])


cpdef void integrate_ray_set(
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo, const double[:, ::1] nhi,
    const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const int64_t[::1] slots, const double[:, :, :, :, ::1] data, int halo,
    const double[:, ::1] origins, const double[:, ::1] directions,
    const double[::1] near, const double[::1] far, int64_t component,
    double step_fraction, int quadrature, int64_t max_samples,
    double[::1] values, double[::1] starts, double[::1] ends,
    int64_t[::1] status, int64_t[::1] samples, int64_t[::1] misses,
):
    import numpy as np
    cdef int64_t ray, count = origins.shape[0]
    cdef double[::1] progress = np.empty(count)
    cdef int64_t[::1] requested = np.full(count,-1,dtype=np.int64)
    cdef bint stalled = False
    initialize_ray_set(lo,hi,origins,directions,near,far,starts,ends,status)
    with nogil:
        for ray in range(count):
            progress[ray] = starts[ray]
            values[ray] = NAN if status[ray] >= 3 else 0.
            samples[ray] = misses[ray] = 0
            _advance_ray_range(ray,ray+1,lo,hi,roots,children,leaves,nlo,nhi,bounds,spacing,
                slots,data,halo,component,origins,directions[ray],progress,ends,
                step_fraction,quadrature,max_samples,values,status,requested,samples,misses)
            if status[ray] == 0:
                if requested[ray] < 0:
                    stalled = True
                status[ray] = 3
                values[ray] = NAN
    if stalled:
        raise RuntimeError("LOS made no progress without a preparation request")


cdef int64_t ray_owner(
    const double* origin, const double[::1] direction,double t,
    const int64_t[:, :, ::1] roots,const int64_t[:, ::1] children,
    const int64_t[::1] leaves,const double[:, ::1] nlo,const double[:, ::1] nhi,
) noexcept nogil:
    """One-sided interval ownership from face times, without coordinate nudges."""
    cdef int64_t r[3]
    cdef int64_t axis,left,right,middle,node,child,bits
    for axis in range(3):
        left,right = 0,roots.shape[axis]
        while left+1 < right:
            middle = (left+right)//2
            if axis==0:
                node = roots[middle,0,0]
            elif axis==1:
                node = roots[0,middle,0]
            else:
                node = roots[0,0,middle]
            if ray_after_face(origin[axis],direction[axis],t,nlo[node,axis]):
                left = middle
            else:
                right = middle
        r[axis] = left
    node = roots[r[0],r[1],r[2]]
    while leaves[node] < 0:
        child = children[node,0]
        bits = 0
        for axis in range(3):
            if ray_after_face(origin[axis],direction[axis],t,nhi[child,axis]):
                bits |= 1 << axis
        node = children[node,bits]
    return leaves[node]


cdef int gauss_leaf(
    const double* origin, const double[::1] direction, int64_t leaf, int64_t slot,
    int64_t component, const double[:, :, ::1] bounds, const double[:, ::1] spacing,
    const double[:, :, :, :, ::1] data, int halo, double first, double last,
    int64_t limit, int64_t* samples, double* value,
) noexcept nogil:
    cdef int a,sign[3],which
    cdef int64_t index[3],extent[3]
    cdef double knots[3],p[3],q,cursor,stop,delta,mid,offset,t,scalar[2],weighted
    cursor = first
    for a in range(3):
        extent[a] = data.shape[a+1]-2*halo
        knots[a] = INFINITY
        if direction[a] == 0.:
            sign[a] = 0
            index[a] = 0
            continue
        q = ((origin[a]+direction[a]*cursor)-bounds[leaf,0,a])/spacing[leaf,a]-.5
        if not isfinite(q) or q < -halo or q >= data.shape[a+1]-halo-1:
            return 8
        sign[a] = 1 if direction[a]>0. else -1
        index[a] = <int64_t>floor(q)+1 if sign[a]>0 else <int64_t>ceil(q)-1
        while 0 <= index[a] < extent[a]:
            t = (bounds[leaf,0,a]+(index[a]+.5)*spacing[leaf,a]-origin[a])/direction[a]
            if t>cursor:
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
        if samples[0] > limit-2:
            return 6
        delta = stop-cursor
        mid = cursor+.5*delta
        offset = .5*delta/1.7320508075688772
        for which in range(2):
            t = mid-offset if which==0 else mid+offset
            for a in range(3):
                p[a] = origin[a]+direction[a]*t
            if not interpolate_scalar(p,leaf,slot,component,bounds,spacing,data,halo,&scalar[which]):
                return 8
            samples[0] += 1
            if not isfinite(scalar[which]):
                return 4
        weighted = .5*scalar[0]+.5*scalar[1]
        value[0] = value[0]+weighted*delta
        if not isfinite(value[0]):
            return 7
        cursor = stop
        for a in range(3):
            if knots[a] <= cursor:
                knots[a] = INFINITY
                index[a] += sign[a]
                while 0 <= index[a] < extent[a]:
                    t = (bounds[leaf,0,a]+(index[a]+.5)*spacing[leaf,a]-origin[a])/direction[a]
                    if t>cursor:
                        knots[a] = t
                        break
                    index[a] += sign[a]
    return 0


cdef void _advance_ray_range(int64_t first, int64_t last,
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi, const double[:, :, ::1] bounds,
    const double[:, ::1] spacing, const int64_t[::1] slots,
    const double[:, :, :, :, ::1] data, int halo, int64_t component,
    const double[:, ::1] origins, const double[::1] direction,
    double[::1] progress, const double[::1] ends,
    double step_fraction, int quadrature, int64_t max_samples, double[::1] values,
    int64_t[::1] status, int64_t[::1] requested,
    int64_t[::1] samples, int64_t[::1] misses,
) noexcept nogil:
    cdef int64_t ray
    cdef int64_t leaf,slot,a,j,count
    cdef int code
    cdef double stop,edge[3],face[3],delta,width,nsteps,t,scalar
    cdef double p[3]
    for ray in range(first,last):
        requested[ray] = -1
        while status[ray] == 0:
            if progress[ray] >= ends[ray]:
                status[ray] = 1
                break
            leaf = ray_owner(&origins[ray,0],direction,progress[ray],roots,children,leaves,nlo,nhi)
            if leaf < 0:
                status[ray] = 5
                break
            stop = ends[ray]
            width = spacing[leaf,0]
            for a in range(3):
                if spacing[leaf,a] < width:
                    width = spacing[leaf,a]
                if direction[a] == 0.:
                    edge[a] = INFINITY
                    face[a] = origins[ray,a]
                else:
                    face[a] = bounds[leaf,1,a] if direction[a]>0. else bounds[leaf,0,a]
                    edge[a] = (face[a]-origins[ray,a])/direction[a]
                    if edge[a] < stop:
                        stop = edge[a]
            if stop <= progress[ray]:
                status[ray] = 5
                break
            slot = slots[leaf]
            if slot < 0:
                requested[ray] = leaf
                misses[ray] += 1
                break
            if quadrature == 1:
                code = gauss_leaf(&origins[ray,0],direction,leaf,slot,component,bounds,
                    spacing,data,halo,progress[ray],stop,max_samples,&samples[ray],&values[ray])
                if code:
                    status[ray] = code
                    break
            else:
                nsteps = ceil((stop-progress[ray])/(width*step_fraction))
                if nsteps < 1.:
                    nsteps = 1.
                if not isfinite(nsteps) or nsteps > max_samples-samples[ray]:
                    status[ray] = 6
                    break
                count = <int64_t>nsteps
                delta = (stop-progress[ray])/count
                for j in range(count):
                    t = progress[ray]+(j+.5)*delta
                    for a in range(3):
                        p[a] = origins[ray,a]+direction[a]*t
                    if not interpolate_scalar(p,leaf,slot,component,bounds,spacing,data,halo,&scalar):
                        status[ray] = 8
                        break
                    samples[ray] += 1
                    if not isfinite(scalar):
                        status[ray] = 4
                        break
                    values[ray] = values[ray]+scalar*delta
                    if not isfinite(values[ray]):
                        status[ray] = 7
                        break
            if status[ray]!=0:
                break
            progress[ray] = stop
        if status[ray] >= 3:
            values[ray] = NAN




cpdef void advance_rays(
    const double[::1] lo, const double[::1] hi,
    const int64_t[:, :, ::1] roots, const int64_t[:, ::1] children,
    const int64_t[::1] leaves, const double[:, ::1] nlo,
    const double[:, ::1] nhi, const double[:, :, ::1] bounds,
    const double[:, ::1] spacing, const int64_t[::1] slots,
    const double[:, :, :, :, ::1] data, int halo, int64_t component,
    const double[:, ::1] origins, const double[::1] direction,
    double[::1] progress, const double[::1] ends,
    double step_fraction, int quadrature, int64_t max_samples, double[::1] values,
    int64_t[::1] status, int64_t[::1] requested,
    int64_t[::1] samples, int64_t[::1] misses,
    int workers=1, int dispatch=0,
):
    cdef int64_t task, first, last, count = origins.shape[0]
    if workers < 1 or dispatch not in (0,1):
        raise ValueError("invalid native worker/dispatch setting")
    if workers > 1 and not SIMESH_ANALYSIS_OPENMP:
        raise RuntimeError("native analysis was built without OpenMP")
    with nogil:
        if workers == 1:
            _advance_ray_range(0, count, lo, hi, roots, children, leaves, nlo, nhi, bounds, spacing, slots, data, halo, component, origins, direction, progress, ends, step_fraction, quadrature, max_samples, values, status, requested, samples, misses)
        elif dispatch == 0:
            for task in prange(workers, schedule='static', num_threads=workers):
                first = count*task//workers
                last = count*(task+1)//workers
                _advance_ray_range(first, last, lo, hi, roots, children, leaves, nlo, nhi, bounds, spacing, slots, data, halo, component, origins, direction, progress, ends, step_fraction, quadrature, max_samples, values, status, requested, samples, misses)
        else:
            for task in prange((count+7)//8, schedule='dynamic', chunksize=1, num_threads=workers):
                first = task*8
                last = first+8
                if last > count:
                    last = count
                _advance_ray_range(first, last, lo, hi, roots, children, leaves, nlo, nhi, bounds, spacing, slots, data, halo, component, origins, direction, progress, ends, step_fraction, quadrature, max_samples, values, status, requested, samples, misses)
