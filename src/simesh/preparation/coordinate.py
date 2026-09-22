"""Whole-domain coordinate-phase preparation with independent value storage."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
import numpy as np

from .._validation import admit, workers_count
from .._execution import worker_context, run_ranges
from ..mesh import resolve_selection
from ..fields import Fields

SCHEME = "canonical-coordinatephase-boundary-v3"


@dataclass(frozen=True, eq=False)
class Geometry:
    """Field-independent adjacency and coordinate-phase geometric facts."""

    block: np.ndarray
    neighbor_type: np.ndarray
    neighbor_index: np.ndarray
    neighbor_children: np.ndarray
    rnode: np.ndarray
    coordinates: np.ndarray
    physical_sides: np.ndarray
    needs_coarse: bool

    @property
    def nbytes(self):
        return sum(a.nbytes for a in vars(self).values() if isinstance(a, np.ndarray))


def build_geometry(mesh):
    """Build from the existing flat forest, without another tree object."""
    if any(mesh.periodic):
        raise ValueError("coordinate-phase does not support periodic halo preparation; use exact-phase")
    from .._kernels.coordinate import fill_geometry, first_unrepresentable_prolongation
    n = mesh.leaf_count
    block = np.asarray(mesh.block_shape, dtype=np.uint32)
    types = np.zeros((n, 27), dtype=np.uint32)
    neighbors = np.zeros_like(types)
    children = np.zeros((n, 64), dtype=np.uint32)
    rnode = np.empty((n, 9), dtype=float)
    coordinates = np.empty((n, 3), dtype=np.int64)
    sides = np.empty((n, 3), dtype=np.int32)
    f = mesh.forest
    fill_geometry(mesh.root_shape, mesh.coord_to_rank, f.root_node_ids, f.node_levels,
                  f.node_coords, f.child_node_ids, f.node_leaf_ids, f.leaf_node_ids,
                  block, mesh.lower, mesh.upper, types, neighbors, children, rnode, coordinates, sides)
    bad = first_unrepresentable_prolongation(rnode, block, types)
    if bad >= 0:
        raise ValueError(f"coordinate-phase prolongation coordinates are not representable at leaf {bad}")
    needs_coarse = bool(np.any((types == 2) | (types == 4)))
    for array in (block, types, neighbors, children, rnode, coordinates, sides):
        array.flags.writeable = False
    return Geometry(block, types, neighbors, children, rnode, coordinates, sides, needs_coarse)


@dataclass
class WorkBuffers:
    """Temporary numerical buffers, never retained by the published Fields."""

    coarse: np.ndarray
    boundary_modes: np.ndarray

    @property
    def nbytes(self):
        return self.coarse.nbytes+self.boundary_modes.nbytes


def allocate_workspace(geometry, field_count):
    n = len(geometry.rnode) if geometry.needs_coarse else 0
    coarse = np.zeros((n, *(int(b)//2+4 for b in geometry.block), field_count), dtype=float)
    return WorkBuffers(coarse, np.zeros((field_count, 6), dtype=np.int32))


def execute(geometry, workspace, values, *, workers, backend):
    """Borrow explicit arrays and preserve the original exchange stage order."""
    from .._kernels.coordinate import ExchangeBindings
    bindings = ExchangeBindings(geometry.block, geometry.neighbor_type, geometry.neighbor_index,
        geometry.neighbor_children, geometry.coordinates, geometry.rnode, geometry.physical_sides,
        values, workspace.coarse, workspace.boundary_modes)
    workers = min(workers, len(values))
    context = worker_context(workers) if backend == "threadpool" else nullcontext(None)
    timings = {}
    with context as executor:
        def phase(number, name):
            start = time.perf_counter()
            if backend == "openmp":
                bindings.openmp_phase(number, workers)
            else:
                run_ranges(len(values), workers, lambda a, b: bindings.local_phase(number, a, b), executor)
            timings[name] = time.perf_counter()-start
        phase(0, "physical_before_seconds")
        if geometry.needs_coarse:
            phase(1, "local_coarsen_seconds")
        start = time.perf_counter()
        bindings.exchange_same_and_restrict()
        timings["same_and_restrict_seconds"] = time.perf_counter()-start
        if geometry.needs_coarse:
            start = time.perf_counter()
            bindings.exchange_coarse_support()
            timings["coarse_support_seconds"] = time.perf_counter()-start
            phase(2, "local_prolong_seconds")
        phase(3, "physical_after_seconds")
    # No binding or workspace becomes a NumPy base or part of the result.
    return timings


def prepare(source, fields=None, *, region=None, leaf_ids=None, workers=1,
            backend="threadpool", memory_limit=None):
    from .._kernels.coordinate import openmp_build_info
    source.validate()
    if any(source.mesh.periodic):
        raise ValueError("coordinate-phase does not support periodic halo preparation; use exact-phase")
    workers_count(workers)
    if backend not in ("threadpool", "openmp"):
        raise ValueError("coordinate backend must be threadpool or openmp")
    if backend == "openmp" and not openmp_build_info()["enabled"]:
        raise RuntimeError("OpenMP preparation requires an OpenMP build")
    mesh = source.mesh
    selected = resolve_selection(mesh, region, leaf_ids)
    if len(selected.leaf_ids) != mesh.leaf_count:
        raise ValueError("coordinate-phase preparation requires full-domain coverage")
    chosen = source.field_ids(fields)
    n, k = mesh.leaf_count, len(chosen)
    block = np.asarray(mesh.block_shape, dtype=np.int64)
    if (np.any(block < 4) or np.any(block % 2) or
            np.any(mesh.root_shape > np.iinfo(np.uint32).max//block) or
            n > np.iinfo(np.uint32).max or
            np.any(mesh.forest.node_coords > np.iinfo(np.int32).max)):
        raise ValueError("coordinate-phase requires even blocks >=4 and representable canonical indices")
    final_bytes = n*k*math.prod(int(b)+4 for b in block)*8
    coarse_bound = n*k*math.prod(int(b)//2+4 for b in block)*8
    geometry_bound = n*640+4096
    required = (source.nbytes+mesh.nbytes+geometry_bound+n*24+k*32 + final_bytes +
                max(source.read_footprint(n,k)+k*math.prod(map(int,block))*8, coarse_bound))
    admit(required, memory_limit, "whole-domain preparation")
    started = time.perf_counter()
    geometry = build_geometry(mesh)
    planned = time.perf_counter()
    ids = np.arange(n, dtype=np.int64)
    values = np.zeros((n, *(int(b)+4 for b in block), k), dtype=float)
    allocated = time.perf_counter()
    read_stats = source.read_native_into(ids, chosen, values, storage_halo=2)
    loaded = time.perf_counter()
    workspace = allocate_workspace(geometry, k)
    workspace.boundary_modes[:] = source._boundary_modes[chosen]
    scratch_bytes = workspace.nbytes
    scratch_ready = time.perf_counter()
    stages = execute(geometry, workspace, values, workers=workers, backend=backend)
    exchanged = time.perf_counter()
    geometry_bytes = geometry.nbytes
    del workspace, geometry
    released = time.perf_counter()
    source.validate()
    stats = {"provider": "coordinate-phase", "workers": workers, "backend": backend,
             "geometry_seconds": planned-started, "read_seconds": loaded-allocated,
             "allocation_seconds": allocated-planned, "interior_copy_seconds": 0.,
             "scratch_allocation_seconds": scratch_ready-loaded,
             "ghost_seconds": exchanged-scratch_ready, "release_seconds": released-exchanged,
             "total_seconds": released-started, "controlled_upper_bytes": required,
             "geometry_bytes": geometry_bytes, "scratch_bytes": scratch_bytes,
             "retained_scratch_bytes": 0, "publication_copy_bytes": 0,
             "read": read_stats, "stages": stages}
    # Geometry and storage use original SFC order. A caller's target order stays
    # in Selection and is addressed through this directory, without a full copy.
    return Fields(mesh, values, selected, ids, tuple(source.fields[i] for i in chosen),
                  2, 2, SCHEME, source.identity, stats,
                  value_identity=source.value_identity(chosen,SCHEME))
