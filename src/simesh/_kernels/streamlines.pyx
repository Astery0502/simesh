# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Prepared-field adapter and accepted-prefix driver for the shared RK kernel."""

import numpy as np
from libc.math cimport hypot, isfinite
from libc.stdint cimport int64_t, uintptr_t
from libc.string cimport memcpy
from cython.parallel cimport prange, threadid
from .native cimport owner_node, interpolate
from .cartesian cimport contains_point
from .tracing_step cimport cell_width, trace_step, finer_step
from .rk4 cimport rk4_trial, RK_RETRY
from .line_quantities cimport twist_density


cdef void _copy_prefixes(const double[:, :, :] paths, const int64_t[::1] offsets,
                        double[:, ::1] output, const double[:, ::1] initial,
                        int64_t first, int64_t last) noexcept nogil:
    cdef int64_t seed, count, cursor
    cdef int axis
    for seed in range(first,last):
        cursor, count = offsets[seed], offsets[seed+1]-offsets[seed]
        if count:
            memcpy(&output[cursor,0], &paths[seed,0,0], count*3*sizeof(double))
            if initial is not None:
                for axis in range(3):
                    output[cursor,axis] = initial[seed,axis]


def _copy_path_segment(const double[:, :, :] paths, const int64_t[::1] offsets,
                       double[:, ::1] output, const double[:, ::1] initial,
                       int64_t first, int64_t last, int workers=1):
    """Copy borrowed accepted prefixes into disjoint, already admitted ranges."""
    cdef int64_t seed, count, task, begin, end, n = paths.shape[0]
    if (paths.shape[2] != 3 or paths.strides[2] != sizeof(double) or
            paths.strides[1] != 3*sizeof(double) or offsets.shape[0] != n+1 or
            output.shape[1] != 3 or offsets[0] != 0 or offsets[n] != output.shape[0] or
            first < 0 or last < first or last > n or workers < 1 or
            (initial is not None and
            (initial.shape[0] != n or initial.shape[1] != 3))):
        raise ValueError("invalid path segment layout")
    if offsets[first] < 0 or offsets[last] > output.shape[0]:
        raise ValueError("path prefix is outside its output range")
    for seed in range(first,last):
        count = offsets[seed+1]-offsets[seed]
        if count < 0 or count > paths.shape[1]:
            raise ValueError("invalid accepted path prefix")
    if workers > 1:
        from .native import openmp_build_info
        if not openmp_build_info()['enabled']:
            raise RuntimeError("native analysis was built without OpenMP")
    with nogil:
        if workers == 1:
            _copy_prefixes(paths,offsets,output,initial,first,last)
        else:
            for task in prange(workers, schedule='static', num_threads=workers):
                begin = first+(last-first)*task//workers
                end = first+(last-first)*(task+1)//workers
                _copy_prefixes(paths,offsets,output,initial,begin,end)


cdef class _PathSegments:
    """Retain segment arrays while disjoint workers gather complete branches."""
    cdef object _owners
    cdef uintptr_t[::1] _values, _offsets
    cdef const int64_t[::1] _branches
    cdef int64_t _count, _seeds

    def __init__(self, values, offsets, const int64_t[::1] branches):
        cdef const double[:, ::1] data
        cdef const int64_t[::1] starts
        cdef int64_t segment, seed
        self._count, self._seeds = len(values), branches.shape[0]
        if len(offsets) != self._count:
            raise ValueError("path segment arrays must have matching offsets")
        # The pointers borrow storage from these retained NumPy owners.
        self._owners = (tuple(values), tuple(offsets))
        self._branches = branches
        self._values = np.empty(self._count, dtype=np.uintp)
        self._offsets = np.empty(self._count, dtype=np.uintp)
        for segment in range(self._count):
            data, starts = self._owners[0][segment], self._owners[1][segment]
            if (data.shape[1] != 3 or starts.shape[0] != self._seeds+1 or
                    starts[0] != 0 or starts[self._seeds] != data.shape[0]):
                raise ValueError("invalid packed path segment")
            for seed in range(self._seeds):
                if starts[seed+1] < starts[seed]:
                    raise ValueError("path offsets must be nondecreasing")
            self._values[segment] = <uintptr_t>&data[0,0] if data.shape[0] else 0
            self._offsets[segment] = <uintptr_t>&starts[0]

    cdef void _copy(self, double[:, ::1] output, const int64_t[::1] offsets,
                    int64_t first, int64_t last) noexcept nogil:
        cdef int64_t seed, segment, cursor, count
        cdef const int64_t* starts
        cdef const double* values
        for seed in range(first,last):
            cursor = offsets[self._branches[seed]]
            for segment in range(self._count):
                starts = <const int64_t*>self._offsets[segment]
                count = starts[seed+1]-starts[seed]
                if count:
                    values = <const double*>self._values[segment]
                    memcpy(&output[cursor,0], values+3*starts[seed], count*3*sizeof(double))
                    cursor += count

    def copy_range(self, double[:, ::1] output, const int64_t[::1] offsets,
                   int64_t first, int64_t last, int workers=1):
        """Gather whole branches into already admitted, disjoint output ranges."""
        cdef int64_t seed, segment, branch, count, task, begin, end
        cdef const int64_t* starts
        if (first < 0 or last < first or last > self._seeds or workers < 1 or
                output.shape[1] != 3 or offsets.shape[0] < 1 or
                offsets[0] != 0 or offsets[offsets.shape[0]-1] != output.shape[0]):
            raise ValueError("invalid final path output layout")
        for seed in range(first,last):
            branch = self._branches[seed]
            if branch < 0 or branch+1 >= offsets.shape[0]:
                raise ValueError("path branch is outside the final output")
            count = 0
            for segment in range(self._count):
                starts = <const int64_t*>self._offsets[segment]
                count += starts[seed+1]-starts[seed]
            if (offsets[branch] < 0 or offsets[branch+1] > output.shape[0] or
                    offsets[branch+1]-offsets[branch] != count):
                raise ValueError("path branch count differs from its output range")
        if workers > 1:
            from .native import openmp_build_info
            if not openmp_build_info()['enabled']:
                raise RuntimeError("native analysis was built without OpenMP")
        with nogil:
            if workers == 1:
                self._copy(output,offsets,first,last)
            else:
                for task in prange(workers, schedule='static', num_threads=workers):
                    begin = first+(last-first)*task//workers
                    end = first+(last-first)*(task+1)//workers
                    self._copy(output,offsets,begin,end)


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
                         int direction, int64_t path_start,
                         const int64_t[::1] active) noexcept nogil:
    cdef int size = 3+data.twist+data.extra
    cdef int offset = 3+data.twist
    cdef double* y = scratch
    cdef double* trial = scratch+size
    cdef double* point = scratch+2*size
    cdef double* slopes = scratch+3*size
    cdef _Context context
    cdef int64_t index, seed, stage
    cdef int a, j, code
    cdef double h
    cdef bint save_paths = batch.paths.shape[0] > 0
    context.inputs = <void*>data
    context.cap, context.fraction = step, fraction
    context.null_threshold, context.direction = null_threshold, direction
    for index in range(first,last):
        seed = index if active is None else active[index]
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


cdef class _AdvanceLines:
    """Bind shared inputs and disjoint state rows before worker dispatch."""
    cdef _Inputs _data
    cdef _Batch _batch
    cdef const int64_t[::1] _active
    cdef int64_t _count

    def __init__(self, fields, companion, integrands, state, const int64_t[::1] active=None):
        cdef int64_t row
        self._data = _Inputs(fields, companion, integrands)
        self._batch = _Batch(state)
        self._active = active
        self._count = self._batch.positions.shape[0] if active is None else active.shape[0]
        if active is not None:
            for row in range(self._count):
                if active[row] < 0 or active[row] >= self._batch.positions.shape[0]:
                    raise ValueError("active seed is outside the line state")

    def advance(self, int64_t first, int64_t last, double step, double step_fraction,
                int64_t max_steps, double max_length, double null_threshold, int direction,
                int workers=1, int dispatch=0, int64_t path_start=0):
        if workers < 1 or dispatch not in (0,1):
            raise ValueError("invalid native worker/dispatch setting")
        if first < 0 or last < first or last > self._count:
            raise ValueError("invalid active seed range")
        if workers > 1:
            from .native import openmp_build_info
            if not openmp_build_info()['enabled']:
                raise RuntimeError("native analysis was built without OpenMP")
        cdef _Inputs data = self._data
        cdef _Batch batch = self._batch
        cdef const int64_t[::1] active = self._active
        cdef double[:, ::1] scratch = np.empty((workers, 7*(3+data.twist+data.extra)))
        cdef int64_t task, start, stop, count = last-first
        with nogil:
            if workers == 1:
                _advance_range(data,batch,first,last,&scratch[0,0],step,step_fraction,max_steps,
                               max_length,null_threshold,direction,path_start,active)
            elif dispatch == 0:
                for task in prange(workers, schedule='static', num_threads=workers):
                    start, stop = first+count*task//workers, first+count*(task+1)//workers
                    _advance_range(data,batch,start,stop,&scratch[threadid(),0],step,step_fraction,max_steps,
                                   max_length,null_threshold,direction,path_start,active)
            else:
                for task in prange((count+7)//8, schedule='dynamic', chunksize=1, num_threads=workers):
                    start = first+task*8
                    stop = min(start+8,last)
                    _advance_range(data,batch,start,stop,&scratch[threadid(),0],step,step_fraction,max_steps,
                                   max_length,null_threshold,direction,path_start,active)
