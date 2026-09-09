# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""AMR field-line and transverse-variation integration with surface events.

Independent implementation of the published variational equations. Geometry
ownership and trilinear interpolation are shared with the native N4 consumers.
"""

from libc.math cimport fabs, fmin, fmax, hypot, isfinite, log, nextafter, INFINITY
from libc.stdint cimport int64_t
from .native cimport owner, interpolate


cdef class _Sampler:
    cdef const double[::1] lo, hi
    cdef const double[::1] sample_lo, sample_hi
    cdef const int64_t[:, :, ::1] roots
    cdef const int64_t[:, ::1] children
    cdef const int64_t[::1] leaves, slots, gslots, cslots
    cdef const double[:, ::1] nlo, nhi, spacing
    cdef const double[:, :, ::1] bounds
    cdef const double[:, :, :, :, ::1] values, gradient, curl
    cdef int halo, ghalo, chalo
    cdef double null_threshold
    cdef double center[3]
    cdef double radius
    cdef bint twist, variational

    def __init__(self, fields, gradient, companion, null_threshold, lower, upper):
        mesh = fields.mesh
        self.lo, self.hi = mesh.lower, mesh.upper
        self.sample_lo, self.sample_hi = lower, upper
        self.roots, self.children, self.leaves = mesh.roots, mesh.children, mesh.node_leaves
        self.nlo, self.nhi = mesh.node_lower, mesh.node_upper
        self.bounds, self.spacing = mesh.bounds, mesh.spacing
        self.slots, self.values, self.halo = fields.slot_of_leaf, fields.values, fields.storage_halo
        self.variational = gradient is not None
        if self.variational:
            self.gslots, self.gradient, self.ghalo = gradient.slot_of_leaf, gradient.values, gradient.storage_halo
        self.null_threshold = null_threshold
        self.radius = 0.
        self.twist = companion is not None
        if self.twist:
            self.cslots, self.curl, self.chalo = companion.slot_of_leaf, companion.values, companion.storage_halo

    cdef int evaluate(self, const double* p, double* b, double* g,
                      double* alpha, double* width) noexcept nogil:
        cdef double q[3]
        cdef double cb[3]
        cdef double norm, distance
        cdef int a
        cdef int64_t leaf, slot
        for a in range(3):
            if not isfinite(p[a]):
                return 10
            # Only event trials may leave the physical box. Constant normal
            # extension lets the scalar event root be bracketed without I/O.
            q[a] = fmax(self.sample_lo[a], fmin(nextafter(self.sample_hi[a], self.sample_lo[a]), p[a]))
        if self.radius > 0.:
            distance = hypot(hypot(q[0]-self.center[0], q[1]-self.center[1]), q[2]-self.center[2])
            if distance >= self.radius:
                # Extend the local target surface as well as the box: a trial
                # beyond the sphere must not read unrelated missing coverage.
                for a in range(3):
                    q[a] = nextafter(self.center[a]+(self.radius/distance)*(q[a]-self.center[a]), self.center[a])
        leaf = owner(q, self.lo, self.hi, self.roots, self.children,
                     self.leaves, self.nlo, self.nhi)
        if leaf < 0:
            return 10
        slot = self.slots[leaf]
        if slot < 0:
            return 7
        width[0] = fmin(self.spacing[leaf,0], fmin(self.spacing[leaf,1], self.spacing[leaf,2]))
        if not interpolate(q, leaf, slot, self.bounds, self.spacing, self.values, self.halo, b):
            return 10
        for a in range(3):
            if not isfinite(b[a]):
                return 5
        norm = hypot(hypot(b[0], b[1]), b[2])
        if not isfinite(norm):
            return 8
        if norm <= self.null_threshold:
            return 4
        if self.variational:
            if self.gslots[leaf] < 0:
                return 7
            if not interpolate(q, leaf, self.gslots[leaf], self.bounds, self.spacing,
                               self.gradient, self.ghalo, g):
                return 10
            for a in range(9):
                if not isfinite(g[a]):
                    return 9
        alpha[0] = 0.
        if self.twist:
            if self.cslots[leaf] < 0:
                return 7
            if not interpolate(q, leaf, self.cslots[leaf], self.bounds, self.spacing,
                               self.curl, self.chalo, cb):
                return 10
            for a in range(3):
                alpha[0] += (cb[a]/norm)*(b[a]/norm)
            alpha[0] /= 12.566370614359172
            if not isfinite(alpha[0]):
                return 9
        return 0


cdef int _rhs(_Sampler sampler, const double* y, int direction,
              double* out, double* width) noexcept nogil:
    cdef double b[3]
    cdef double g[9]
    cdef double alpha, norm
    cdef int a, j, column, code
    code = sampler.evaluate(y, b, g, &alpha, width)
    if code:
        return code
    norm = hypot(hypot(b[0], b[1]), b[2])
    for a in range(3):
        out[a] = direction*b[a]/norm
        for column in range(2):
            out[3+3*column+a] = 0.
            if sampler.variational:
                for j in range(3):
                    out[3+3*column+a] += direction*g[3*a+j]*y[3+3*column+j]
    out[9] = alpha
    return 0


cdef int _rk4(_Sampler sampler, const double* y, double h, int direction,
              double* out, double* min_width,
              const double* lo, const double* hi, bint* outside_stage) noexcept nogil:
    cdef double k[4][10]
    cdef double stage[10]
    cdef double width, factor, total
    cdef int i, j, code
    min_width[0] = INFINITY
    outside_stage[0] = False
    for i in range(4):
        factor = .5 if i < 3 else 1.
        for j in range(10):
            stage[j] = y[j] if i == 0 else y[j] + factor*h*k[i-1][j]
        for j in range(3):
            if stage[j] < lo[j] or stage[j] > hi[j]:
                outside_stage[0] = True
        if sampler.radius > 0. and hypot(hypot(stage[0]-sampler.center[0], stage[1]-sampler.center[1]), stage[2]-sampler.center[2]) > sampler.radius:
            outside_stage[0] = True
        code = _rhs(sampler, stage, direction, k[i], &width)
        if code:
            return code
        min_width[0] = fmin(min_width[0], width)
    for j in range(10):
        total = k[0][j] + 2.*k[1][j] + 2.*k[2][j] + k[3][j]
        out[j] = y[j] + (h/6.)*total
        if not isfinite(out[j]):
            return 9
    return 0


cdef double _margin(const double* p, const double* lo, const double* hi,
                    const double* seed, double radius) noexcept nogil:
    cdef int a
    cdef double margin = INFINITY
    for a in range(3):
        margin = fmin(margin, fmin(p[a]-lo[a], hi[a]-p[a]))
    if radius > 0:
        margin = fmin(margin, radius-hypot(hypot(p[0]-seed[0], p[1]-seed[1]), p[2]-seed[2]))
    return margin


cdef int _boundary(double* p, const double* lo, const double* hi,
                   const double* seed, double radius, double tol,
                   const double* tangent, bint outward_only, double* normal) noexcept nogil:
    cdef int a, count = 0, face = 0
    cdef double distance, radial_speed = 0.
    for a in range(3):
        normal[a] = 0.
    for a in range(3):
        if fabs(p[a]-lo[a]) <= tol and (not outward_only or tangent[a] < 0.):
            p[a] = lo[a]
            normal[a] = -1.
            face = 5-2*a
            count += 1
        elif fabs(p[a]-hi[a]) <= tol and (not outward_only or tangent[a] > 0.):
            p[a] = hi[a]
            normal[a] = 1.
            face = 6-2*a
            count += 1
    if count > 1:
        return 7
    if count:
        return face
    if radius > 0:
        distance = hypot(hypot(p[0]-seed[0], p[1]-seed[1]), p[2]-seed[2])
        if fabs(distance-radius) <= tol:
            for a in range(3):
                normal[a] = (p[a]-seed[a])/distance
                radial_speed += normal[a]*tangent[a]
            if not outward_only or radial_speed > 0.:
                for a in range(3):
                    p[a] = seed[a]+radius*normal[a]
                return 9
    return 0


cdef void _half(_Sampler sampler, const double* seed, const double* center, int direction,
                const double* lo, const double* hi, double fraction, double max_step,
                int64_t max_steps, double max_length, double tolerance, double radius,
                double* position, double* uv, double* scaling, double* b_end,
                double* normal, double* length, double* twist, int64_t* steps,
                int64_t* status, int64_t* face) noexcept nogil:
    cdef double y[10]
    cdef double trial[10]
    cdef double middle[10]
    cdef double b[3]
    cdef double g[9]
    cdef double tangent[3]
    cdef double width, alpha, norm, h, min_width, low, high, mid, scale, s
    cdef int a, j, axis, code, iteration, retry
    cdef bint outside, crossed
    for a in range(3):
        y[a] = seed[a]
        sampler.center[a] = center[a]
        if seed[a] < lo[a] or seed[a] > hi[a]:
            status[0] = 6
            return
    sampler.radius = radius
    code = sampler.evaluate(y, b, g, &alpha, &width)
    if code:
        status[0] = code
        return
    norm = hypot(hypot(b[0], b[1]), b[2])
    axis = 0
    for a in range(3):
        tangent[a] = direction*b[a]/norm
        if fabs(b[a]) < fabs(b[axis]):
            axis = a
    if sampler.variational:
        # Project the least parallel Cartesian axis for a stable seed basis.
        for a in range(3):
            y[3+a] = (1. if a == axis else 0.) - (b[axis]/norm)*(b[a]/norm)
        s = hypot(hypot(y[3], y[4]), y[5])
        for a in range(3):
            y[3+a] /= s
        y[6] = (b[1]/norm)*y[5]-(b[2]/norm)*y[4]
        y[7] = (b[2]/norm)*y[3]-(b[0]/norm)*y[5]
        y[8] = (b[0]/norm)*y[4]-(b[1]/norm)*y[3]
    else:
        for j in range(3,9):
            y[j] = 0.
    y[9] = 0.
    face[0] = _boundary(y, lo, hi, center, radius, tolerance, tangent, True, normal)
    if face[0]:
        status[0] = 12 if face[0] == 9 else 3
    while status[0] == 0:
        if steps[0] >= max_steps:
            status[0] = 1
            break
        if length[0] >= max_length:
            status[0] = 2
            break
        h = fmin(max_step, fmin(fraction*width, max_length-length[0]))
        if radius > 0.:
            h = fmin(h, radius*.25)
        crossed = False
        for retry in range(64):
            if h <= 0. or length[0]+h == length[0]:
                status[0] = 11
                break
            code = _rk4(sampler, y, h, direction, trial, &min_width, lo, hi, &outside)
            if code:
                status[0] = code
                break
            if h > fraction*min_width*(1.+1.e-12):
                h = fraction*min_width
                continue
            crossed = _margin(trial, lo, hi, center, radius) <= 0.
            if outside and not crossed:
                h *= .5
                continue
            break
        else:
            status[0] = 11
        if status[0]:
            break
        if crossed:
            low, high = 0., h
            for iteration in range(80):
                mid = .5*(low+high)
                code = _rk4(sampler, y, mid, direction, middle, &min_width, lo, hi, &outside)
                if code:
                    status[0] = code
                    break
                if _margin(middle, lo, hi, center, radius) > 0.:
                    low = mid
                else:
                    high = mid
                if high-low <= tolerance*.25:
                    break
            if status[0]:
                break
            h = .5*(low+high)
            code = _rk4(sampler, y, h, direction, trial, &min_width, lo, hi, &outside)
            if code:
                status[0] = code
                break
        if trial[0] == y[0] and trial[1] == y[1] and trial[2] == y[2]:
            status[0] = 11
            break
        for j in range(10):
            y[j] = trial[j]
        length[0] += h
        steps[0] += 1
        code = sampler.evaluate(y, b, g, &alpha, &width)
        if code:
            status[0] = code
            break
        norm = hypot(hypot(b[0], b[1]), b[2])
        for a in range(3):
            tangent[a] = direction*b[a]/norm
        face[0] = _boundary(y, lo, hi, center, radius, tolerance, tangent, True, normal)
        if face[0]:
            status[0] = 12 if face[0] == 9 else 3
        elif crossed:
            status[0] = 11
        if sampler.variational:
            scale = 0.
            for j in range(3,9):
                scale = fmax(scale, fabs(y[j]))
            if scale == 0. or not isfinite(scale):
                status[0] = 9
                break
            scaling[0] += log(scale)
            for j in range(3,9):
                y[j] /= scale
    for a in range(3):
        position[a] = y[a]
        b_end[a] = b[a]
        uv[a] = y[3+a]
        uv[3+a] = y[6+a]
    twist[0] = y[9]


def trace_halves(fields, gradient, companion, const double[:, ::1] seeds,
                 const double[:, ::1] centers,
                 const double[::1] lower, const double[::1] upper,
                 double step_fraction, double max_step, int64_t max_steps,
                 double max_length, double null_threshold, double boundary_tolerance,
                 double local_radius, double[:, :, ::1] positions,
                 double[:, :, :, ::1] vectors, double[:, ::1] scales,
                 double[:, :, ::1] endpoint_fields, double[:, :, ::1] normals,
                 double[:, ::1] lengths, double[:, ::1] twists,
                 int64_t[:, ::1] steps, int64_t[:, ::1] status, int64_t[:, ::1] faces):
    cdef _Sampler sampler = _Sampler(fields, gradient, companion, null_threshold, lower, upper)
    cdef Py_ssize_t i
    cdef int side
    with nogil:
        for i in range(seeds.shape[0]):
            for side in range(2):
                _half(sampler, &seeds[i,0], &centers[i,0], 2*side-1, &lower[0], &upper[0],
                      step_fraction, max_step, max_steps, max_length,
                      boundary_tolerance, local_radius, &positions[i,side,0],
                      &vectors[i,side,0,0], &scales[i,side], &endpoint_fields[i,side,0],
                      &normals[i,side,0], &lengths[i,side], &twists[i,side],
                      &steps[i,side], &status[i,side], &faces[i,side])
