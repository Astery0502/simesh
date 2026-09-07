"""Independent arclength RK4 seeds with persistent stage state on pool misses."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .fields import PreparedPool
from .diagnostics import WithCurl, CurlPool, with_curl


class Termination(IntEnum):
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


@dataclass(frozen=True)
class TraceResult:
    seed_ids: np.ndarray
    seeds: np.ndarray
    positions: np.ndarray
    length: np.ndarray
    steps: np.ndarray
    termination: np.ndarray
    samples: np.ndarray
    misses: np.ndarray
    localized_endpoint: bool = False
    twist: np.ndarray | None = None
    trajectories: np.ndarray | None = None

    @property
    def point_counts(self):
        return np.where(self.termination==Termination.OUTSIDE_SEED,0,self.steps+1)


def _run_chunk(fields, seeds, seed_ids, step, max_steps, max_length, null_threshold,
               direction, workers, executor, want_twist, trajectories):
    from simesh.utils.lib.analysis.native import advance_lines
    n = len(seeds)
    positions = seeds.copy()
    length = np.zeros(n, dtype=float)
    steps = np.zeros(n, dtype=np.int64)
    status = np.zeros(n, dtype=np.int64)
    stages = np.zeros(n, dtype=np.int64)
    tangents = np.empty((n,4,3), dtype=float)
    requested = np.full(n, -1, dtype=np.int64)
    samples = np.zeros(n, dtype=np.int64)
    misses = np.zeros(n, dtype=np.int64)
    alpha = np.empty((n if want_twist else 0,4),dtype=float)
    twist = np.zeros(n if want_twist else 0,dtype=float)
    paths = np.full((n,max_steps+1,3),np.nan) if trajectories else np.empty((0,0,3))
    empty = np.empty((0,0,0,0,0))
    is_pool = isinstance(fields, (PreparedPool,CurlPool))
    base = fields.primary if isinstance(fields,WithCurl) else fields
    mesh = fields.source.mesh if is_pool else base.mesh
    initial = mesh.locate(seeds)
    status[initial < 0] = Termination.OUTSIDE_SEED
    if max_steps == 0:
        status[initial >= 0] = Termination.MAX_STEPS
    elif max_length == 0:
        status[initial >= 0] = Termination.MAX_LENGTH
    if trajectories:
        paths[initial>=0,0] = seeds[initial>=0]
    partitions = [part for part in np.array_split(np.arange(n), workers) if len(part)]

    def advance(product, part):
        companion = product.curl if want_twist else None
        product = product.primary if isinstance(product,WithCurl) else product
        span = slice(int(part[0]), int(part[-1])+1)
        advance_lines(mesh.lower, mesh.upper, mesh.roots, mesh.children, mesh.node_leaves,
            mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
            product.slot_of_leaf, product.values, product.halo,
            step, max_steps, max_length, null_threshold, direction,
            positions[span], length[span], steps[span], status[span], stages[span],
            tangents[span], requested[span], samples[span], misses[span],
            companion.slot_of_leaf if want_twist else product.slot_of_leaf,
            companion.values if want_twist else empty,1,
            alpha[span] if want_twist else alpha,twist[span] if want_twist else twist,
            paths[span] if trajectories else paths)

    while np.any(status == Termination.RUNNING):
        context = fields.borrow(fields.resident_leaf_ids) if is_pool else nullcontext(fields)
        with context as product:
            if executor is None:
                advance(product, partitions[0])
            else:
                futures = [executor.submit(advance, product, part) for part in partitions]
                for future in futures:
                    future.result()
        missing = np.unique(requested[(status == Termination.RUNNING) & (requested >= 0)])
        if missing.size:
            if is_pool:
                # Stage data is private and survives eviction of prior stage owners.
                with fields.borrow(missing):
                    pass
            else:
                status[(status == Termination.RUNNING) & (requested >= 0)] = Termination.MISSING_COVERAGE
        elif np.any(status == Termination.RUNNING):
            raise RuntimeError("tracer made no progress and requested no preparation")
    return TraceResult(seed_ids.copy(), seeds.copy(), positions, length, steps, status, samples, misses,
                       twist=twist if want_twist else None,trajectories=paths if trajectories else None)


def iter_traces(fields, seeds, *, seed_ids=None, step, max_steps=1000,
                max_length=np.inf, null_threshold=0., direction=1, workers=1,
                seed_batch=256, budget_bytes=2*1024**3, twist=False, trajectories=False):
    """Yield owned summary batches; retain or write them under caller ownership.

    Trajectories are optional admitted accepted prefixes; unused tails are NaN.
    Pool stage state resumes on misses; detached fields report missing coverage.
    Twist samples curl at RK stages. Endpoints remain last accepted points.
    """
    seeds = np.asarray(seeds)
    if (seeds.ndim != 2 or seeds.shape[1] != 3 or seeds.dtype != np.float64 or
            not seeds.flags.c_contiguous or not np.isfinite(seeds).all()):
        raise ValueError("seeds must be finite C-contiguous float64 (n,3)")
    if seed_ids is None:
        seed_ids = np.arange(len(seeds), dtype=np.int64)
    else:
        seed_ids = np.asarray(seed_ids)
        if (seed_ids.shape != (len(seeds),) or seed_ids.dtype != np.int64 or
                len(np.unique(seed_ids)) != len(seeds)):
            raise ValueError("seed_ids must be unique int64 IDs matching seeds")
    if (not np.isfinite(step) or step <= 0 or type(max_steps) is not int or
            not 0 <= max_steps <= np.iinfo(np.int64).max or
            np.isnan(max_length) or max_length < 0 or not np.isfinite(null_threshold) or
            null_threshold < 0 or direction not in (-1,1) or type(workers) is not int or
            not 1 <= workers <= 4 or type(seed_batch) is not int or seed_batch < 1):
        raise ValueError("invalid trace step, limits, direction or worker settings")
    if type(twist) is not bool or type(trajectories) is not bool:
        raise ValueError("twist and trajectories must be booleans")
    pool = isinstance(fields,(PreparedPool,CurlPool))
    if isinstance(fields,CurlPool) and fields._values is None:
        raise RuntimeError("curl companion is closed")
    if pool and (fields.primary._closed if isinstance(fields,CurlPool) else fields._closed):
        raise RuntimeError("tracing pool is closed")
    base = fields.primary if isinstance(fields,WithCurl) else fields
    definitions = tuple(fields.source.fields[i] for i in fields.field_ids) if pool else base.fields
    if len(definitions) != 3 or len({f.units for f in definitions}) != 1:
        raise ValueError("tracing requires three ordered magnetic components in common units")
    if base.halo < 1:
        raise ValueError("tracing requires at least one valid interpolation layer")
    if not len(seeds):
        return
    seed_batch = min(seed_batch, fields.capacity) if pool else seed_batch
    private = seed_batch*(576+(24*(max_steps+1) if trajectories else 0))
    transient_reserve = seeds.nbytes+3*seed_ids.nbytes+private
    temporary = None
    if twist and max_steps > 0 and max_length > 0:
        if isinstance(fields,PreparedPool):
            fields = temporary = CurlPool(fields,budget_bytes=budget_bytes-transient_reserve)
        elif not isinstance(fields,(CurlPool,WithCurl)):
            fields = with_curl(fields,budget_bytes=budget_bytes-transient_reserve)
        if isinstance(fields,WithCurl) and (fields.curl.mesh is not fields.primary.mesh or
                fields.curl.source is not fields.primary.source or fields.curl.halo<1):
            raise ValueError("curl group must share the primary source/mesh and valid support")
    else:
        fields = fields.primary if isinstance(fields,(CurlPool,WithCurl)) else fields
    pool = isinstance(fields,(PreparedPool,CurlPool))
    if isinstance(fields,WithCurl):
        footprint = fields.primary.mesh.nbytes+fields.primary.nbytes+fields.curl.nbytes
    else:
        footprint = fields.controlled_bytes if pool else fields.mesh.nbytes+fields.nbytes
    # Source seeds/IDs, private stage/partition/miss arrays, one result batch.
    required = footprint + transient_reserve
    if required > budget_bytes:
        raise MemoryError(f"trace needs {required} controlled bytes, budget {budget_bytes}")
    manager = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    try:
        with manager as executor:
            for first in range(0, len(seeds), seed_batch):
                stop = min(first+seed_batch, len(seeds))
                yield _run_chunk(fields, seeds[first:stop], seed_ids[first:stop], step,
                                 max_steps, max_length, null_threshold, direction, workers, executor,
                                 twist,trajectories)
    finally:
        if temporary is not None:
            temporary.close()


def trace(fields, seeds, **kwargs):
    """Collect an explicitly admitted full summary. Use iter_traces for sinks."""
    budget = kwargs.get("budget_bytes", 2*1024**3)
    # Result fields: IDs, two positions, length and four integer counters.
    output_bytes = len(seeds)*(96+(8 if kwargs.get("twist",False) else 0) +
        (24*(kwargs.get("max_steps",1000)+1) if kwargs.get("trajectories",False) else 0))
    kwargs = dict(kwargs, budget_bytes=budget-2*output_bytes)
    if kwargs["budget_bytes"] < 0:
        raise MemoryError("full trace output exceeds budget")
    batches = list(iter_traces(fields, seeds, **kwargs))
    if not batches:
        return TraceResult(np.empty(0,np.int64),np.empty((0,3)),np.empty((0,3)),
            np.empty(0),*(np.empty(0,np.int64) for _ in range(4)),
            twist=np.empty(0) if kwargs.get("twist",False) else None,
            trajectories=np.empty((0,kwargs.get("max_steps",1000)+1,3)) if kwargs.get("trajectories",False) else None)
    arrays = [np.concatenate([getattr(batch, name) for batch in batches]) for name in
              ("seed_ids","seeds","positions","length","steps","termination","samples","misses")]
    twists = np.concatenate([b.twist for b in batches]) if batches[0].twist is not None else None
    paths = np.concatenate([b.trajectories for b in batches]) if batches[0].trajectories is not None else None
    return TraceResult(*arrays,twist=twists,trajectories=paths)


def retrace(fields, result, selected_seed_ids, **kwargs):
    """Explicitly retrace selected original seeds, retaining accepted trajectories."""
    selected = np.asarray(selected_seed_ids)
    if selected.ndim!=1 or selected.dtype.kind not in "iu" or len(np.unique(selected))!=len(selected):
        raise ValueError("selected seed IDs must be a unique integer vector")
    if selected.dtype.kind == "u" and np.any(selected>np.iinfo(np.int64).max):
        raise ValueError("selected seed ID is absent from the original result")
    previous_bytes = sum(value.nbytes for value in vars(result).values() if isinstance(value,np.ndarray))
    mapping_bytes = result.seed_ids.nbytes*2+selected.nbytes*4
    remaining = kwargs.get("budget_bytes",2*1024**3)-previous_bytes-mapping_bytes
    if remaining < 0:
        raise MemoryError("retained prior result and selection mapping exceed retrace budget")
    order = np.argsort(result.seed_ids)
    sorted_ids = result.seed_ids[order]
    positions = np.searchsorted(sorted_ids,selected.astype(np.int64))
    if np.any(positions>=len(order)) or not np.array_equal(sorted_ids[positions],selected):
        raise ValueError("selected seed ID is absent from the original result")
    rows = order[positions]
    return trace(fields,np.ascontiguousarray(result.seeds[rows]),
                 seed_ids=np.ascontiguousarray(result.seed_ids[rows]),
                 **dict(kwargs,trajectories=True,budget_bytes=remaining))
