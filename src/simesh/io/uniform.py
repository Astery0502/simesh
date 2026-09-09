"""Bounded uniform exports from immutable sources or AMRVAC files."""

from contextlib import closing, nullcontext
import math
import os
import numpy as np

from .source import Source
from .amrvac import open_amrvac
from .._uniform import geometry, output_arrays, write_blocks
from .._validation import remaining, workers_count
from .._execution import worker_context


def export_uniform(source, resolution, *, fields=None, bounds=None, interpolation="zero",
                   output=None, batch_size=64, workers=1, memory_limit=None,
                   scheme=None, support_capacity=128, tile_rows=64):
    """Export stored fields with bounded input storage and explicit reconstruction.

    Parameters
    ----------
    source : Source or str or Path
        Open immutable source, or a Cartesian 3D ordinary AMRVAC v5 file opened and
        closed by this call. A supplied Source remains open.
    resolution : sequence of int
        Positive uniform-grid cell counts (nx, ny, nz).
    fields : str or sequence, optional
        Stored field names or local Source indices in output order.
    bounds : array-like, optional
        Lower and upper physical bounds (2, 3); defaults to the original domain.
    interpolation : {"zero", "native", "linear"}
        Containing-cell zero-order sampling, exact block placement, or trilinear
        cell-center sampling. Native requires matching spacing on every leaf and
        cell-aligned bounds. Zero/native read interiors only. Linear prepares
        halos using the explicit scheme; none of these modes performs conservative
        coarse-cell averaging.
    output : tuple of ndarray, optional
        Writable contiguous float64 values (nx, ny, nz, k) and bool validity
        (nx, ny, nz), including NumPy memory maps. No input aliases. Failure may
        leave partial writes; callers manage persistence and flushing.
    batch_size : int
        Maximum interior read batch, or resident prepared-leaf capacity for linear
        sampling. Preparation also uses its separate support workspace. Output
        remains a full volume.
    workers : int
        Workers writing disjoint leaf regions or sampling linear tiles; source reads
        and exact-phase preparation are serial.
    memory_limit : int, optional
        Accounted-array budget, including output, input backing and batch/read
        scratch; not a process RSS limit, even with memory-mapped output.
    scheme : {"exact-phase"}, optional
        Required explicitly for linear sampling; omit for zero/native. Bounded
        linear export supports exact-phase only. For coordinate-phase, use prepare
        followed by applications.uniform_grid with resident Fields.
    support_capacity : int
        Requested preparation support capacity in leaves for linear sampling;
        exact-phase applies its minimum working capacity.
    tile_rows : int
        Maximum rows per linear-sampling tile; must be positive.

    Returns
    -------
    UniformResult
        Owned or caller-backed component-last values, definitions, physical bounds
        and coverage. Uncovered cells are NaN with false validity. Coverage does not
        imply finite values; zero/native preserve covered nonfinite values. The
        result survives source closure.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    workers_count(workers)
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    if interpolation == "linear":
        if scheme != "exact-phase":
            raise ValueError("linear export requires explicit scheme='exact-phase'; "
                             "use prepare and uniform_grid for coordinate-phase")
    elif scheme is not None:
        raise ValueError("scheme is only used by linear export")
    if isinstance(source, (str, os.PathLike)):
        context = open_amrvac(source, memory_limit=memory_limit)
    elif isinstance(source, Source):
        context = nullcontext(source)
    else:
        raise TypeError("source must be a Source or AMRVAC path")
    with context as current:
        current.validate()
        mesh = current.mesh
        selected = current.field_ids(fields)
        shape, lower, upper, step = geometry(mesh, resolution, bounds, interpolation)
        inputs = (*current._memory_arrays,
                  *(a for a in vars(mesh).values() if isinstance(a, np.ndarray)),
                  *(a for a in mesh.forest if isinstance(a, np.ndarray)))
        value_scheme = "interior"
        if interpolation == "linear":
            from ..bounded import PreparedPool, iter_uniform_bounded
            if any(current.fields[i].interpretation.startswith("categorical") for i in selected):
                raise ValueError("linear export requires continuous components")
            nx, ny, nz = shape
            count = len(selected)
            volume_bytes = math.prod(shape)*(8*count+1)+48
            slab_bytes = nx*ny*(8*count+9)
            # Reserve both the yielded slab and its successor, plus tile scratch.
            stream_bytes = 2*slab_bytes+min(nx,tile_rows)*ny*(8*count+128)+512
            with PreparedPool(current, selected, capacity=batch_size, scheme=scheme,
                    support_capacity=support_capacity,
                    memory_limit=remaining(memory_limit, volume_bytes+stream_bytes)) as pool:
                values, valid = output_arrays(shape, count, output, inputs,
                    pool.controlled_bytes+stream_bytes, memory_limit)
                with closing(iter_uniform_bounded(pool, shape, bounds=(lower, upper),
                        tile_rows=tile_rows, workers=workers,
                        memory_limit=remaining(memory_limit, volume_bytes+512))) as slabs:
                    for iz, slab in slabs:
                        values[:, :, iz] = slab.values
                        valid[:, :, iz] = slab.valid
                        del slab
                value_scheme = pool.scheme
        else:
            capacity = min(batch_size, mesh.leaf_count)
            leaf_bytes = math.prod(mesh.block_shape)*len(selected)*8
            footprint = (mesh.nbytes+current.nbytes+(capacity+1)*leaf_bytes+64*capacity+4*selected.nbytes+
                         current.read_footprint(capacity, len(selected)))
            values, valid = output_arrays(shape, len(selected), output, inputs, footprint, memory_limit)
            values.fill(np.nan)
            valid.fill(False)
            batch = np.empty((capacity, *mesh.block_shape, len(selected)))
            columns = np.arange(len(selected), dtype=np.int64)
            slots = np.arange(capacity, dtype=np.int64)
            first_center, last_center = lower+.5*step, lower+(np.asarray(shape)-.5)*step
            with worker_context(workers) as executor:
                for first in range(0, mesh.leaf_count, capacity):
                    # Keep only leaves containing at least one possible target center.
                    # Candidate metadata and values are bounded by batch_size.
                    ids = np.arange(first, min(first+capacity, mesh.leaf_count), dtype=np.int64)
                    ids = np.asarray([leaf for leaf in ids
                        if np.all(mesh.node_upper[mesh.leaf_nodes[leaf]] > first_center)
                        and np.all(mesh.node_lower[mesh.leaf_nodes[leaf]] <= last_center)], dtype=np.int64)
                    if not len(ids):
                        continue
                    current.read_native_into(ids, selected, batch[:len(ids)])
                    write_blocks(mesh, ids, slots[:len(ids)], batch, 0, columns, lower, step,
                                 interpolation, values, valid, workers=workers, executor=executor)
        current.validate()
        from ..applications import UniformResult
        return UniformResult(values, valid, lower, upper,
            tuple(current.fields[i] for i in selected), current.value_identity(selected, value_scheme))


def export_uniform_vtk(source, path, resolution, *, overwrite=False, **uniform_options):
    """Resample stored fields and publish a uniform binary legacy VTK file.

    Parameters
    ----------
    source : Source or str or Path
        Open immutable source or Cartesian 3D ordinary AMRVAC v5 file. A supplied
        Source remains open; a file opened by this call is closed before writing.
    path : str or Path
        Destination .vtk file; its parent must exist.
    resolution : sequence of int
        Positive uniform-grid cell counts (nx, ny, nz).
    overwrite : bool
        Replace an existing destination atomically; otherwise refuse it before sampling.
    **uniform_options : dict
        Keyword controls forwarded unchanged to [export_uniform][simesh.export_uniform],
        including fields, bounds, interpolation, output, batch_size, workers,
        memory_limit, scheme, support_capacity and tile_rows. Linear interpolation
        requires scheme="exact-phase" explicitly.

    Returns
    -------
    Path
        Published uniform VTK file with scalar values and coverage, as described by
        [write_uniform_vtk][simesh.write_uniform_vtk]. No AMR hierarchy is written.

    Notes
    -----
    Sampling retains a complete output volume, optionally in caller-provided memory
    maps. The memory budget covers each phase, including a supplied Source during
    serialization. Failure may leave caller output arrays changed but preserves
    an existing destination file.
    """
    from .vtk import _destination, write_uniform_vtk

    destination = _destination(path, overwrite)
    grid = export_uniform(source, resolution, **uniform_options)
    retained = source.mesh.nbytes + source.nbytes if isinstance(source, Source) else 0
    return write_uniform_vtk(destination, grid, overwrite=overwrite,
        memory_limit=remaining(uniform_options.get("memory_limit"), retained))
