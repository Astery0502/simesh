"""Accepted-prefix RK4 tracing with explicit retained diagnostic dependencies."""

from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from .fields import require_fields, require_continuous
from ._validation import admit, workers_count, remaining
from ._execution import worker_context, run_ranges, native_dispatch


class Termination(IntEnum):
    """Native trace stop reasons. MAX_STEPS/MAX_LENGTH describe accepted prefixes; DOMAIN_EXIT is a different outcome.
    """
    RUNNING = 0
    MAX_STEPS = 1
    MAX_LENGTH = 2
    DOMAIN_EXIT = 3
    NULL_FIELD = 4
    NONFINITE_FIELD = 5
    OUTSIDE_SEED = 6
    MISSING_COVERAGE = 7
    UNREPRESENTABLE_NORM = 8
    NONFINITE_DIAGNOSTIC = 9
    UNREPRESENTABLE_SAMPLE = 10
    UNREPRESENTABLE_STEP = 11


@dataclass(frozen=True)
class TraceResult:
    """Owned raw accepted-prefix tracing result; not a localized footpoint product.

    Attributes
    ----------
    seed_ids, seeds : ndarray
        Input IDs and (n, 3) seed positions.
    positions, length, steps : ndarray
        Final accepted positions, coordinate lengths and step counts.
    termination : ndarray
        Per-seed Termination codes; limits differ from physical exit.
    samples, misses : ndarray
        Sampling and missing-coverage counts.
    localized_endpoint : bool
        False for this accepted-prefix tracer.
    trajectories : ndarray, optional
        Dense paths; point_counts gives accepted prefixes.
    twist : ndarray, optional
        May describe an incomplete prefix; inspect termination.
    integrals : ndarray, optional
        Per-component RK integrals shaped (n, k), in integrand units times
        coordinate length. Both branch directions accumulate positive arc length;
        these may be partial integrals. Inspect termination before combining them.
    integral_fields : tuple of FieldDefinition
        Definitions of the supplied scalar rates, in integral-column order.
    """
    seed_ids: np.ndarray
    seeds: np.ndarray
    positions: np.ndarray
    length: np.ndarray
    steps: np.ndarray
    termination: np.ndarray
    samples: np.ndarray
    misses: np.ndarray
    localized_endpoint: bool = False
    trajectories: np.ndarray | None = None
    twist: np.ndarray | None = None
    integrals: np.ndarray | None = None
    integral_fields: tuple = ()

    @property
    def point_counts(self):
        """Accepted trajectory prefix sizes; outside seeds have zero stored points."""
        return np.where(self.termination == Termination.OUTSIDE_SEED, 0, self.steps+1)


@dataclass
class _LineState:
    seed_ids: np.ndarray
    seeds: np.ndarray
    positions: np.ndarray
    length: np.ndarray
    steps: np.ndarray
    status: np.ndarray
    stages: np.ndarray
    tangents: np.ndarray
    requested: np.ndarray
    samples: np.ndarray
    misses: np.ndarray
    alpha: np.ndarray
    twist: np.ndarray
    paths: np.ndarray
    step_size: np.ndarray
    integrals: np.ndarray
    integral_slopes: np.ndarray


def _new_state(mesh, seeds, ids, max_steps, max_length, trajectories, twist, *, path_capacity=None, integral_count=0):
    n = len(seeds)
    capacity = max_steps+1 if path_capacity is None else path_capacity
    state = _LineState(ids.copy(), seeds.copy(), seeds.copy(), np.zeros(n),
        np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64),
        np.empty((n,4,3)), np.full(n,-1,np.int64), np.zeros(n,np.int64), np.zeros(n,np.int64),
        np.empty((n if twist else 0,4)), np.zeros(n if twist else 0),
        np.full((n,capacity,3),np.nan) if trajectories else np.empty((0,0,3)), np.zeros(n),
        np.zeros((n,integral_count)), np.empty((n,4,integral_count)))
    owners = mesh.locate(seeds)
    state.status[owners < 0] = Termination.OUTSIDE_SEED
    if max_steps == 0:
        state.status[owners >= 0] = Termination.MAX_STEPS
    elif max_length == 0:
        state.status[owners >= 0] = Termination.MAX_LENGTH
    if trajectories:
        state.paths[owners >= 0,0] = seeds[owners >= 0]
    return state


def _advance_state(fields, companion, state, *, step, step_fraction, max_steps, max_length,
                   null_threshold, direction, workers, executor, native=False, dispatch=0,
                   path_start=0, integrands=None):
    """Advance private RK state; missing data is returned to the coordinator."""
    from ._kernels.streamlines import advance_lines
    def advance(first, last):
        advance_lines(fields, companion, integrands, state, first, last,
            np.inf if step is None else step, 0. if step_fraction is None else step_fraction,
            max_steps, max_length, null_threshold, direction,
            workers if native else 1, dispatch, path_start)
    run_ranges(len(state.seeds), 1 if native else workers*(8 if dispatch and workers>1 else 1),
               advance, None if native else executor)


# A delivery segment is independent of the integration limit. Slot zero holds
# the initial seed only in the first segment; later segments skip that slot.
_PATH_SEGMENT_STEPS = 64


def _trace_batch_bytes(seeds, seed_ids, count, max_steps, trajectories, integral_count=0, workers=1):
    return (seeds.nbytes+3*seed_ids.nbytes+count*(640+40*integral_count+
            (48*(max_steps+1) if trajectories else 0))+workers*56*(4+integral_count))


def _trace_output_bytes(count, max_steps, trajectories, twist, integral_count=0):
    return count*(96+8*integral_count+(8 if twist else 0)+(24*(max_steps+1) if trajectories else 0))


def _iter_path_segments(fields, seeds, seed_ids, *, step, step_fraction, max_steps, max_length,
                        null_threshold, direction, workers, backend, schedule,
                        memory_limit=None):
    """Yield borrowed path views and persistent state for one admitted seed batch.

    The consumer copies accepted points before advancing this iterator. Missing
    coverage terminates a line; a full segment only pauses it between RK steps.
    """
    native, dispatch = native_dispatch(backend, schedule)
    capacity = min(max_steps, _PATH_SEGMENT_STEPS)+1
    reserve = seeds.nbytes+seed_ids.nbytes+len(seeds)*(640+24*capacity)+workers*56*3
    admit(fields.mesh.nbytes+fields.nbytes+reserve,memory_limit,"trace segment")
    state = _new_state(fields.mesh,seeds,seed_ids,max_steps,max_length,True,False,
                       path_capacity=capacity)
    context = nullcontext(None) if native else worker_context(workers)
    path_start = 0
    with context as executor:
        while True:
            fields._check()
            _advance_state(fields,None,state,step=step,step_fraction=step_fraction,max_steps=max_steps,max_length=max_length,
                null_threshold=null_threshold,direction=direction,workers=workers,executor=executor,
                native=native,dispatch=dispatch,path_start=path_start)
            pending = state.status == Termination.RUNNING
            missing = pending & (state.requested >= 0)
            state.status[missing] = Termination.MISSING_COVERAGE
            pending &= ~missing
            if np.any(pending & (state.steps != path_start+_PATH_SEGMENT_STEPS)):
                raise RuntimeError("tracer made no progress and requested no coverage")
            if path_start == 0:
                counts = np.where(state.status == Termination.OUTSIDE_SEED,0,state.steps+1)
                paths = state.paths
            else:
                counts = np.maximum(state.steps-path_start,0)
                paths = state.paths[:,1:]
            yield state, paths, counts
            if not np.any(pending):
                return
            path_start += _PATH_SEGMENT_STEPS


def _result(state, trajectories, twist, integrands=None):
    return TraceResult(state.seed_ids,state.seeds,state.positions,state.length,state.steps,
                       state.status,state.samples,state.misses,
                       trajectories=state.paths if trajectories else None,
                       twist=state.twist if twist else None,
                       integrals=state.integrals if integrands is not None else None,
                       integral_fields=() if integrands is None else tuple(integrands.fields))


def _step_controls(step, step_fraction):
    """Validate common fixed or local-cell tracing controls without allocating."""
    if step is not None and (not np.isfinite(step) or step <= 0):
        raise ValueError("step must be a positive finite coordinate-length cap")
    if step_fraction is not None and (not np.isfinite(step_fraction) or not 0 < step_fraction <= 1):
        raise ValueError("step_fraction must be in (0, 1] or None for fixed steps")
    if step is None and step_fraction is None:
        raise ValueError("fixed-step tracing requires step when step_fraction is None")


def _validate_inputs(seeds, seed_ids, step, max_steps, max_length, null_threshold,
                     direction, workers, seed_batch, trajectories, twist, step_fraction=.25):
    _step_controls(step, step_fraction)
    workers_count(workers)
    seeds = np.asarray(seeds)
    if (seeds.ndim != 2 or seeds.shape[1] != 3 or seeds.dtype != np.float64 or
            not seeds.flags.c_contiguous or not np.isfinite(seeds).all()):
        raise ValueError("seeds must be finite C-contiguous float64 (n,3)")
    if seed_ids is None:
        seed_ids = np.arange(len(seeds),dtype=np.int64)
    else:
        seed_ids = np.asarray(seed_ids)
        if (seed_ids.shape != (len(seeds),) or seed_ids.dtype != np.int64 or
                len(np.unique(seed_ids)) != len(seeds)):
            raise ValueError("seed_ids must be unique int64 IDs matching seeds")
    if (type(max_steps) is not int or
            not 0 <= max_steps <= np.iinfo(np.int64).max or np.isnan(max_length) or
            max_length < 0 or not np.isfinite(null_threshold) or null_threshold < 0 or
            direction not in (-1,1) or type(seed_batch) is not int or seed_batch < 1 or
            type(trajectories) is not bool or type(twist) is not bool):
        raise ValueError("invalid trace step, limits, direction or output settings")
    return seeds, seed_ids


def _validate_vector(fields):
    require_continuous(fields, operation="tracing")
    if len(fields.fields) != 3 or len({f.units for f in fields.fields}) != 1:
        raise ValueError("tracing requires three ordered vector components with common units")


def _resolve_curl(fields, companion, want_twist, memory_limit):
    if not want_twist:
        if companion is not None:
            raise ValueError("curl_field is only consumed when twist=True")
        return None
    if companion is None:
        require_fields(fields, halo=2, operation="twist with automatic curl")
        from .operators.derivatives import curl
        companion = curl(fields,memory_limit=memory_limit)
    require_continuous(companion,operation="curl interpolation")
    expected = ("curl",fields.value_identity,tuple(fields.fields),(0,1,2),fields.scheme)
    if (companion.mesh is not fields.mesh or companion.source is not fields.source or
            companion.derivation != expected or len(companion.fields) != 3 or
            np.any(companion.slot_of_leaf[fields.leaf_ids] < 0)):
        raise ValueError("curl_field must derive from this vector group and cover its valid domain")
    return companion


def _integrand_count(fields, integrands):
    if integrands is None:
        return 0
    require_continuous(integrands, operation="line integrals")
    if integrands.mesh is not fields.mesh:
        raise ValueError("integrands must use the same Mesh as the traced field")
    return len(integrands.fields)


def iter_traces(fields, seeds, *, seed_ids=None, step=None, step_fraction=.25, max_steps=1000, max_length=np.inf,
                null_threshold=0., direction=1, workers=1, seed_batch=256,
                trajectories=False, twist=False, curl_field=None, integrands=None, memory_limit=None,
                backend="threadpool", schedule="static"):
    """Yield owned results; diagnostic integration uses accepted RK segments.

    Parameters
    ----------
    fields : Fields
        Exactly three continuous vector components with common units and one valid halo.
    seeds : array-like
        Seed positions with shape (n, 3).
    seed_ids : array-like, optional
        Unique seed IDs in input order.
    step : float, optional
        Positive coordinate-length cap; required only for fixed-step tracing.
    step_fraction : float, optional
        Local cell-size fraction in (0, 1], default 0.25. RK stages entering finer
        cells reduce the step and restart it. None selects fixed steps using step.
    max_steps : int
        Maximum accepted integration steps per branch.
    max_length : float
        Maximum integrated coordinate length per branch.
    null_threshold : float
        Vector norm at or below which tracing reports a null field.
    direction : int
        Trace along (+1) or against (-1) the vector field.
    workers : int
        Number of workers over disjoint ranges.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    trajectories : bool
        Retain dense accepted path prefixes; allocated capacity grows with max_steps.
    twist : bool
        Compute twist; automatic curl preparation requires two primary halo layers.
    curl_field : Fields, optional
        Matching raw curl result with valid interpolation support and derivation identity.
    integrands : Fields, optional
        Prepared continuous scalar rates on the same Mesh, with one valid halo.
        Each component is integrated against positive branch arc length using
        the trajectory RK stages. Missing or nonfinite samples stop the branch;
        results then retain the accepted partial integrals. The caller establishes
        compatible coordinates, snapshot and units; no Source is read.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    backend : str
        Execution backend: threadpool or explicitly built openmp.
    schedule : str
        Native scheduling policy: static or dynamic.

    Returns
    -------
    iterator of TraceResult
        Owned accepted-prefix results; inspect TraceResult.termination. Optional
        trajectories use dense storage.
    """
    _validate_vector(fields)
    seeds, seed_ids = _validate_inputs(seeds,seed_ids,step,max_steps,max_length,
        null_threshold,direction,workers,seed_batch,trajectories,twist,step_fraction)
    native,dispatch = native_dispatch(backend,schedule)
    count = min(seed_batch,len(seeds))
    integral_count = _integrand_count(fields, integrands)
    reserve = _trace_batch_bytes(seeds,seed_ids,count,max_steps,trajectories,integral_count,workers)
    reserve += 0 if integrands is None else integrands.nbytes
    if not len(seeds):
        return
    need_curl = twist and max_steps > 0 and max_length > 0
    companion = (_resolve_curl(fields,curl_field,twist,remaining(memory_limit,reserve))
                 if need_curl or curl_field is not None else None)
    footprint = fields.mesh.nbytes+fields.nbytes+(0 if companion is None else companion.nbytes)
    admit(footprint+reserve,memory_limit,"trace batches")
    context = nullcontext(None) if native else worker_context(workers)
    with context as executor:
        for first in range(0,len(seeds),seed_batch):
            fields._check()
            last = min(first+seed_batch,len(seeds))
            state = _new_state(fields.mesh,seeds[first:last],seed_ids[first:last],
                               max_steps,max_length,trajectories,twist,integral_count=integral_count)
            _advance_state(fields,companion,state,step=step,step_fraction=step_fraction,max_steps=max_steps,max_length=max_length,
                null_threshold=null_threshold,direction=direction,workers=workers,executor=executor,
                native=native,dispatch=dispatch,integrands=integrands)
            pending = state.status == Termination.RUNNING
            if np.any(pending & (state.requested < 0)):
                raise RuntimeError("tracer made no progress and requested no coverage")
            state.status[pending] = Termination.MISSING_COVERAGE
            yield _result(state,trajectories,twist,integrands)
            del state


def _collect(batches, n, max_steps, trajectories, twist, integrands=None):
    first = next(batches,None)  # Trigger validation before allocating outputs.
    try:
        result = TraceResult(np.empty(n,np.int64),np.empty((n,3)),np.empty((n,3)),
            np.empty(n),*(np.empty(n,np.int64) for _ in range(4)),
            trajectories=np.empty((n,max_steps+1,3)) if trajectories else None,
            twist=np.empty(n) if twist else None,
            integrals=None if integrands is None else np.empty((n,len(integrands.fields))),
            integral_fields=() if integrands is None else tuple(integrands.fields))
        names = ("seed_ids","seeds","positions","length","steps","termination","samples","misses")
        offset, batch = 0, first
        del first
        while batch is not None:
            last = offset+len(batch.seed_ids)
            for name in names:
                getattr(result,name)[offset:last] = getattr(batch,name)
            if trajectories:
                result.trajectories[offset:last] = batch.trajectories
            if twist:
                result.twist[offset:last] = batch.twist
            if integrands is not None:
                result.integrals[offset:last] = batch.integrals
            offset = last
            del batch
            batch = next(batches,None)
        return result
    finally:
        batches.close()


def trace(fields, seeds, *, seed_ids=None, step=None, step_fraction=.25, max_steps=1000, max_length=np.inf,
          null_threshold=0., direction=1, workers=1, seed_batch=256,
          trajectories=False, twist=False, curl_field=None, integrands=None, memory_limit=None,
          backend="threadpool", schedule="static"):
    """Collect an admitted complete result without a full concatenation copy.

    Parameters
    ----------
    fields : Fields
        Exactly three continuous vector components with common units and one valid halo.
    seeds : array-like
        Seed positions with shape (n, 3).
    seed_ids : array-like, optional
        Unique seed IDs in input order.
    step : float, optional
        Positive coordinate-length cap; required only for fixed-step tracing.
    step_fraction : float, optional
        Local cell-size fraction in (0, 1], default 0.25. RK stages entering finer
        cells reduce the step and restart it. None selects fixed steps using step.
    max_steps : int
        Maximum accepted integration steps per branch.
    max_length : float
        Maximum integrated coordinate length per branch.
    null_threshold : float
        Vector norm at or below which tracing reports a null field.
    direction : int
        Trace along (+1) or against (-1) the vector field.
    workers : int
        Number of workers over disjoint ranges.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    trajectories : bool
        Retain dense accepted path prefixes; allocated capacity grows with max_steps.
    twist : bool
        Compute twist; automatic curl preparation requires two primary halo layers.
    curl_field : Fields, optional
        Matching raw curl result with valid interpolation support and derivation identity.
    integrands : Fields, optional
        Prepared continuous scalar rates on the same Mesh, with one valid halo.
        Each component is integrated against positive branch arc length using
        the trajectory RK stages. Missing or nonfinite samples stop the branch;
        results then retain the accepted partial integrals. The caller establishes
        compatible coordinates, snapshot and units; no Source is read.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    backend : str
        Execution backend: threadpool or explicitly built openmp.
    schedule : str
        Native scheduling policy: static or dynamic.

    Returns
    -------
    TraceResult
        Owned accepted-prefix results; inspect TraceResult.termination. Optional
        trajectories use dense storage.
    """
    _validate_vector(fields)
    if twist and curl_field is None and max_steps>0 and max_length>0:
        require_fields(fields,halo=2,operation="twist with automatic curl")
    seeds,seed_ids = _validate_inputs(seeds,seed_ids,step,max_steps,max_length,null_threshold,
                                    direction,workers,seed_batch,trajectories,twist,step_fraction)
    integral_count = _integrand_count(fields, integrands)
    output = _trace_output_bytes(len(seeds),max_steps,trajectories,twist,integral_count)
    batches = iter_traces(fields,seeds,seed_ids=seed_ids,step=step,step_fraction=step_fraction,max_steps=max_steps,max_length=max_length,
        null_threshold=null_threshold,direction=direction,workers=workers,seed_batch=seed_batch,
        trajectories=trajectories,twist=twist,curl_field=curl_field,integrands=integrands,
        memory_limit=remaining(memory_limit,output),backend=backend,schedule=schedule)
    return _collect(batches,len(seeds),max_steps,trajectories,twist,integrands)


def _retrace_inputs(result, selected_seed_ids, memory_limit):
    selected = np.asarray(selected_seed_ids)
    if (selected.ndim != 1 or selected.dtype.kind not in "iu" or
            len(np.unique(selected)) != len(selected) or
            (selected.dtype.kind == "u" and np.any(selected > np.iinfo(np.int64).max))):
        raise ValueError("selected seed IDs must be unique representable integers")
    old_bytes = sum(value.nbytes for value in vars(result).values() if isinstance(value,np.ndarray))
    mapping = 2*result.seed_ids.nbytes+4*selected.nbytes
    limit = remaining(memory_limit,old_bytes+mapping)
    order = np.argsort(result.seed_ids)
    sorted_ids = result.seed_ids[order]
    positions = np.searchsorted(sorted_ids,selected.astype(np.int64))
    if np.any(positions >= len(order)) or not np.array_equal(sorted_ids[positions],selected):
        raise ValueError("selected seed ID is absent from the original result")
    rows = order[positions]
    return np.ascontiguousarray(result.seeds[rows]),np.ascontiguousarray(result.seed_ids[rows]),limit


def retrace(fields, result, selected_seed_ids, **kwargs):
    """Explicitly trace selected original seed IDs again, retaining trajectories.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    result : TraceResult
        Previous raw trace supplying seed positions and IDs.
    selected_seed_ids : sequence of int
        Existing seed IDs to integrate again.
    **kwargs : object
        Controls forwarded to simesh.trace, including step and step_fraction.

    Returns
    -------
    TraceResult
        New integration from selected original seeds; not a checkpoint resume.
    """
    _validate_vector(fields)
    if (kwargs.get("twist",False) and kwargs.get("curl_field") is None and
            kwargs.get("max_steps",1000)>0 and kwargs.get("max_length",np.inf)>0):
        require_fields(fields,halo=2,operation="twist with automatic curl")
    seeds,ids,limit=_retrace_inputs(result,selected_seed_ids,kwargs.pop("memory_limit",None))
    kwargs.pop("trajectories",None)
    return trace(fields,seeds,seed_ids=ids,
                 trajectories=True,memory_limit=limit,**kwargs)
