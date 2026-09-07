"""Independent arclength RK4 seeds with persistent stage state on pool misses."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .fields import PreparedPool


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


def _run_chunk(fields, seeds, seed_ids, step, max_steps, max_length, null_threshold,
               direction, workers, executor):
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
    is_pool = isinstance(fields, PreparedPool)
    mesh = fields.source.mesh if is_pool else fields.mesh
    initial = mesh.locate(seeds)
    status[initial < 0] = Termination.OUTSIDE_SEED
    partitions = [part for part in np.array_split(np.arange(n), workers) if len(part)]

    def advance(product, part):
        span = slice(int(part[0]), int(part[-1])+1)
        advance_lines(mesh.lower, mesh.upper, mesh.roots, mesh.children, mesh.node_leaves,
            mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
            product.slot_of_leaf, product.values, product.halo,
            step, max_steps, max_length, null_threshold, direction,
            positions[span], length[span], steps[span], status[span], stages[span],
            tangents[span], requested[span], samples[span], misses[span])

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
    return TraceResult(seed_ids.copy(), seeds.copy(), positions, length, steps, status, samples, misses)


def iter_traces(fields, seeds, *, seed_ids=None, step, max_steps=1000,
                max_length=np.inf, null_threshold=0., direction=1, workers=1,
                seed_batch=256, budget_bytes=2*1024**3):
    """Yield owned summary batches; retain or write them under caller ownership.

    No trajectories are allocated. Pool stage state resumes on misses; detached
    fields report missing coverage. Endpoints are last accepted points.
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
    pool = isinstance(fields, PreparedPool)
    definitions = tuple(fields.source.fields[i] for i in fields.field_ids) if pool else fields.fields
    if len(definitions) != 3 or len({f.units for f in definitions}) != 1:
        raise ValueError("tracing requires three ordered magnetic components in common units")
    if fields.halo < 1:
        raise ValueError("tracing requires at least one valid interpolation layer")
    seed_batch = min(seed_batch, fields.capacity) if pool else seed_batch
    footprint = (fields.controlled_bytes if pool else fields.mesh.nbytes+fields.nbytes)
    # Source seeds/IDs, private stage/partition/miss arrays, one result batch.
    required = footprint + seeds.nbytes + seed_ids.nbytes + seed_batch*512
    if required > budget_bytes:
        raise MemoryError(f"trace needs {required} controlled bytes, budget {budget_bytes}")
    manager = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with manager as executor:
        for first in range(0, len(seeds), seed_batch):
            stop = min(first+seed_batch, len(seeds))
            yield _run_chunk(fields, seeds[first:stop], seed_ids[first:stop], step,
                             max_steps, max_length, null_threshold, direction, workers, executor)


def trace(fields, seeds, **kwargs):
    """Collect an explicitly admitted full summary. Use iter_traces for sinks."""
    budget = kwargs.get("budget_bytes", 2*1024**3)
    # Result fields: IDs, two positions, length and four integer counters.
    output_bytes = len(seeds)*96
    kwargs = dict(kwargs, budget_bytes=budget-2*output_bytes)
    if kwargs["budget_bytes"] < 0:
        raise MemoryError("full trace output exceeds budget")
    batches = list(iter_traces(fields, seeds, **kwargs))
    if not batches:
        return TraceResult(np.empty(0,np.int64),np.empty((0,3)),np.empty((0,3)),
            np.empty(0),*(np.empty(0,np.int64) for _ in range(4)))
    arrays = [np.concatenate([getattr(batch, name) for batch in batches]) for name in
              ("seed_ids","seeds","positions","length","steps","termination","samples","misses")]
    return TraceResult(*arrays)
