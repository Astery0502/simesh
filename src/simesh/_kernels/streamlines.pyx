# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Prepared-field adapter and accepted-prefix driver for the shared RK kernel."""

import numpy as np
from libc.math cimport hypot, isfinite
from libc.stdint cimport int64_t
from cython.parallel cimport prange, threadid
from .native cimport owner_node, contains_point, interpolate
from .tracing_step cimport cell_width, trace_step, finer_step
from .rk4 cimport rk4_trial, RK_RETRY
from .line_quantities cimport twist_density


cdef class _Inputs:
    cdef const double[::1] lo, hi
    cdef const int64_t[:, :, ::1] roots
    cdef const int64_t[:, ::1] children
    cdef const int64_t[::1] leaves, slots, cslots, islots
    cdef const double[:, ::1] nlo, nhi, spacing
    cdef const double[:, :, ::1] bounds
    cdef const double[:, :, :, :, ::1] values, curl, integrands
    cdef int halo, chalo, ihalo, extra
    cdef bint twist

    def __init__(self, fields, companion, integrands):
        mesh = fields.mesh
        self.lo, self.hi = mesh.lower, mesh.upper
        self.roots, self.children, self.leaves = mesh.roots, mesh.children, mesh.node_leaves
        self.nlo, self.nhi = mesh.node_lower, mesh.node_upper
        self.bounds, self.spacing = mesh.bounds, mesh.spacing
        self.slots, self.values, self.halo = fields.slot_of_leaf, fields.values, fields.storage_halo
        self.twist = companion is not None
        if self.twist:
            self.cslots, self.curl, self.chalo = companion.slot_of_leaf, companion.values, companion.storage_halo
        self.extra = 0 if integrands is None else len(integrands.fields)
        if self.extra:
            self.islots, self.integrands, self.ihalo = integrands.slot_of_leaf, integrands.values, integrands.storage_halo


cdef class _Batch:
    cdef double[:, ::1] positions, alpha, integral_values
    cdef double[:, :, ::1] tangents, paths, integral_slopes
    cdef double[::1] length, step_size, twist
    cdef int64_t[::1] steps, status, stages, requested, samples, misses

    def __init__(self, state):
        self.positions, self.length, self.steps = state.positions, state.length, state.steps
        self.status, self.stages, self.step_size = state.status, state.stages, state.step_size
        self.tangents, self.alpha, self.twist = state.tangents, state.alpha, state.twist
        self.paths = state.paths
        self.requested, self.samples, self.misses = state.requested, state.samples, state.misses
        self.integral_values, self.integral_slopes = state.integrals, state.integral_slopes


cdef struct _Context:
    void* inputs
    int64_t node_hint
    int64_t requested
    int64_t samples
    int64_t misses
    double cap
    double fraction
    double remaining
    double length
    double null_threshold
    int direction


cdef int _evaluate(_Inputs data, _Context* context, const double* p,
                   double* out, double* h, int stage) noexcept nogil:
    cdef double b[3]
    cdef double cb[3]
    cdef double norm, width, value
    cdef int a, offset = 3+data.twist
    cdef int64_t leaf, slot
    if context.node_hint < 0 or not contains_point(p, &data.nlo[context.node_hint,0], &data.nhi[context.node_hint,0]):
        context.node_hint = owner_node(p, data.lo, data.hi, data.roots, data.children,
                                      data.leaves, data.nlo, data.nhi)
    if context.node_hint < 0:
        return 3
    leaf = data.leaves[context.node_hint]
    width = cell_width(data.spacing, leaf)
    if stage == 0:
        h[0] = trace_step(width, context.fraction, context.cap if h[0] == 0. else h[0], context.remaining)
        if h[0] <= 0. or not isfinite(h[0]) or context.length+h[0] == context.length:
            return 11
    elif finer_step(h[0], width, context.fraction):
        h[0] = trace_step(width, context.fraction, h[0], context.remaining)
        return RK_RETRY
    slot = data.slots[leaf]
    if slot < 0:
        context.requested = leaf
        context.misses += 1
        return 7
    if not interpolate(p, leaf, slot, data.bounds, data.spacing, data.values, data.halo, b):
        return 10
    context.samples += 1
    if not (isfinite(b[0]) and isfinite(b[1]) and isfinite(b[2])):
        return 5
    norm = hypot(hypot(b[0], b[1]), b[2])
    if not isfinite(norm):
        return 8
    if norm <= context.null_threshold:
        return 4
    if data.twist:
        slot = data.cslots[leaf]
        if slot < 0:
            context.requested = leaf
            context.misses += 1
            return 7
        if not interpolate(p, leaf, slot, data.bounds, data.spacing, data.curl, data.chalo, cb):
            return 10
        value = twist_density(b, cb, norm)
        if not isfinite(value):
            return 9
        out[3] = value
    if data.extra:
        slot = data.islots[leaf]
        if slot < 0:
            context.requested = leaf
            context.misses += 1
            return 7
        if not interpolate(p, leaf, slot, data.bounds, data.spacing, data.integrands, data.ihalo, out+offset):
            return 10
        for a in range(data.extra):
            if not isfinite(out[offset+a]):
                return 9
    for a in range(3):
        out[a] = context.direction*(b[a]/norm)
    return 0


cdef int _stage(void* context, const double* state, double* out,
                double* h, int stage) noexcept nogil:
    return _evaluate(<_Inputs>(<_Context*>context).inputs, <_Context*>context,
                     state, out, h, stage)


cdef void _advance_range(_Inputs data, _Batch batch, int64_t first, int64_t last,
                         double* scratch, double step, double fraction,
                         int64_t max_steps, double max_length, double null_threshold,
                         int direction, int64_t path_start) noexcept nogil:
    cdef int size = 3+data.twist+data.extra
    cdef int offset = 3+data.twist
    cdef double* y = scratch
    cdef double* trial = scratch+size
    cdef double* point = scratch+2*size
    cdef double* slopes = scratch+3*size
    cdef _Context context
    cdef int64_t seed, stage
    cdef int a, j, code
    cdef double h
    cdef bint save_paths = batch.paths.shape[0] > 0
    context.inputs = <void*>data
    context.cap, context.fraction = step, fraction
    context.null_threshold, context.direction = null_threshold, direction
    for seed in range(first,last):
        context.node_hint = -1
        context.requested, context.samples, context.misses = -1, 0, 0
        stage, h = batch.stages[seed], batch.step_size[seed]
        for a in range(3):
            y[a] = batch.positions[seed,a]
        if data.twist:
            y[3] = batch.twist[seed]
        for a in range(data.extra):
            y[offset+a] = batch.integral_values[seed,a]
        for j in range(stage):
            for a in range(3):
                slopes[j*size+a] = batch.tangents[seed,j,a]
            if data.twist:
                slopes[j*size+3] = batch.alpha[seed,j]
            for a in range(data.extra):
                slopes[j*size+offset+a] = batch.integral_slopes[seed,j,a]
        while batch.status[seed] == 0:
            if batch.steps[seed] >= max_steps:
                batch.status[seed] = 1
                break
            if batch.length[seed] >= max_length:
                batch.status[seed] = 2
                break
            if save_paths and batch.steps[seed]-path_start >= batch.paths.shape[1]-1:
                break
            context.length = batch.length[seed]
            context.remaining = max_length-context.length
            code = rk4_trial(&context, _stage, y, size, &h, &stage, slopes, point, trial)
            if code:
                # Missing coverage is a resumable request, not a committed stop.
                if code != 7:
                    batch.status[seed] = code
                break
            if not contains_point(trial, &data.lo[0], &data.hi[0]):
                batch.status[seed] = 3
                break
            if trial[0] == y[0] and trial[1] == y[1] and trial[2] == y[2]:
                batch.status[seed] = 11
                break
            # Commit all requested quantities together only after admission.
            for a in range(size):
                y[a] = trial[a]
            for a in range(3):
                batch.positions[seed,a] = y[a]
            if data.twist:
                batch.twist[seed] = y[3]
            for a in range(data.extra):
                batch.integral_values[seed,a] = y[offset+a]
            batch.length[seed] += h
            batch.steps[seed] += 1
            if save_paths:
                for a in range(3):
                    batch.paths[seed,batch.steps[seed]-path_start,a] = y[a]
            stage, h = 0, 0.
        batch.stages[seed], batch.step_size[seed] = stage, h
        for j in range(stage):
            for a in range(3):
                batch.tangents[seed,j,a] = slopes[j*size+a]
            if data.twist:
                batch.alpha[seed,j] = slopes[j*size+3]
            for a in range(data.extra):
                batch.integral_slopes[seed,j,a] = slopes[j*size+offset+a]
        batch.requested[seed] = context.requested
        batch.samples[seed] += context.samples
        batch.misses[seed] += context.misses


def advance_lines(fields, companion, integrands, state, int64_t first, int64_t last,
                  double step, double step_fraction, int64_t max_steps,
                  double max_length, double null_threshold, int direction,
                  int workers=1, int dispatch=0, int64_t path_start=0):
    from .native import openmp_build_info
    if workers < 1 or dispatch not in (0,1):
        raise ValueError("invalid native worker/dispatch setting")
    if workers > 1 and not openmp_build_info()['enabled']:
        raise RuntimeError("native analysis was built without OpenMP")
    cdef _Inputs data = _Inputs(fields, companion, integrands)
    cdef _Batch batch = _Batch(state)
    cdef double[:, ::1] scratch = np.empty((workers, 7*(3+data.twist+data.extra)))
    cdef int64_t task, start, stop, count = last-first
    with nogil:
        if workers == 1:
            _advance_range(data,batch,first,last,&scratch[0,0],step,step_fraction,max_steps,
                           max_length,null_threshold,direction,path_start)
        elif dispatch == 0:
            for task in prange(workers, schedule='static', num_threads=workers):
                start, stop = first+count*task//workers, first+count*(task+1)//workers
                _advance_range(data,batch,start,stop,&scratch[threadid(),0],step,step_fraction,max_steps,
                               max_length,null_threshold,direction,path_start)
        else:
            for task in prange((count+7)//8, schedule='dynamic', chunksize=1, num_threads=workers):
                start = first+task*8
                stop = min(start+8,last)
                _advance_range(data,batch,start,stop,&scratch[threadid(),0],step,step_fraction,max_steps,
                               max_length,null_threshold,direction,path_start)
