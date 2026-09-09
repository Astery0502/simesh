"""Uniform output validation and interior delivery shared by resident and source workflows."""

import math
import numpy as np

from ._validation import admit, workers_count
from ._execution import worker_context, run_ranges
from .fields import require_fields, _component_indices, _input_arrays


def geometry(mesh, resolution, bounds, interpolation):
    from .slices import _uniform_geometry
    if interpolation not in ("zero", "native", "linear"):
        raise ValueError("interpolation must be 'zero', 'native', or 'linear'")
    shape, lower, upper = _uniform_geometry(mesh, resolution, bounds)
    lower, upper = lower.copy(), upper.copy()
    step = (upper-lower)/shape
    if not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("uniform spacing must be finite and positive")
    if interpolation == "native":
        # Placement accepts only the original cell lattice, never a near-resolution
        # resampling. Tolerance accounts for floating coordinate arithmetic only.
        tolerance = 32*np.finfo(float).eps
        spacing_range = np.array((mesh.spacing.min(axis=0), mesh.spacing.max(axis=0)))
        if not np.allclose(spacing_range, step, rtol=tolerance, atol=0):
            raise ValueError("native placement requires matching cell spacing on every leaf")
        for face in (lower, upper):
            offset = (face-mesh.lower)/step
            if not np.isfinite(offset).all() or np.any(np.abs(offset) >= 2**52):
                raise ValueError("native placement requires representable cell offsets")
            if np.any(np.abs(offset-np.rint(offset)) > tolerance*np.maximum(1, np.abs(offset))):
                raise ValueError("native placement requires cell-aligned bounds")
    return shape, lower, upper, step


def output_arrays(shape, count, output, inputs, footprint, memory_limit):
    admit(footprint+math.prod(shape)*(8*count+1)+48, memory_limit, "uniform volume")
    if output is None:
        values = np.empty((*shape, count))
        valid = np.empty(shape, dtype=bool)
    else:
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError("output must be (values, valid)")
        values, valid = output
        for array, expected, dtype in ((values, (*shape, count), np.float64), (valid, shape, np.bool_)):
            if (not isinstance(array, np.ndarray) or array.shape != expected or array.dtype != dtype
                    or not array.flags.c_contiguous or not array.flags.writeable):
                raise ValueError("output must match writable contiguous volume values and validity")
        if np.shares_memory(values, valid) or any(np.shares_memory(a, b) for a in output for b in inputs):
            raise ValueError("output must not alias input or other output")
    return values, valid


def write_blocks(mesh, ids, slots, data, halo, selected, lower, step, interpolation,
                 values, valid, *, workers=1, executor=None, owners=None, z_offset=0):
    from ._kernels.native import uniform_zero_blocks

    def fill(first, last):
        if interpolation == "zero":
            uniform_zero_blocks(mesh.leaf_nodes, mesh.node_lower, mesh.node_upper,
                mesh.bounds, mesh.spacing, ids[first:last], slots[first:last], data,
                halo, selected, lower, step, values, valid.view(np.uint8), owners, z_offset)
            return
        for position in range(first, last):
            leaf, slot = ids[position], slots[position]
            start = np.rint((mesh.node_lower[mesh.leaf_nodes[leaf]]-lower)/step).astype(np.int64)
            start[2] -= z_offset
            stop = start+mesh.block_shape
            lo, hi = np.maximum(start, 0), np.minimum(stop, values.shape[:3])
            if np.any(lo >= hi):
                continue
            destination = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            source = tuple(slice(int(a-s)+halo, int(b-s)+halo) for a, b, s in zip(lo, hi, start))
            # Copy components separately: advanced indexing would allocate a block.
            for column, component in enumerate(selected):
                values[(*destination, column)] = data[(slot, *source, component)]
            valid[destination] = True
            if owners is not None:
                owners[destination] = leaf

    run_ranges(len(ids), workers, fill, executor)


def resident(fields, resolution, components, output, bounds, interpolation, workers, memory_limit):
    require_fields(fields)
    selected = np.asarray(_component_indices(fields, components), dtype=np.int64)
    workers_count(workers)
    shape, lower, upper, step = geometry(fields.mesh, resolution, bounds, interpolation)
    slots = fields.slot_of_leaf[fields.leaf_ids]
    values, valid = output_arrays(shape, len(selected), output, (*_input_arrays(fields), lower, upper),
        fields.nbytes+fields.mesh.nbytes+slots.nbytes+8*len(selected), memory_limit)
    values.fill(np.nan)
    valid.fill(False)
    with worker_context(workers) as executor:
        write_blocks(fields.mesh, fields.leaf_ids, slots, fields.values, fields.storage_halo,
                     selected, lower, step, interpolation, values, valid,
                     workers=workers, executor=executor)
    from .applications import UniformResult
    return UniformResult(values, valid, lower, upper, tuple(fields.fields[i] for i in selected), fields.value_identity)


def slabs(fields, resolution, components, bounds, interpolation, workers, memory_limit):
    from .slices import SliceResult, Plane
    require_fields(fields)
    selected = np.asarray(_component_indices(fields, components), dtype=np.int64)
    workers_count(workers)
    shape, lower, upper, step = geometry(fields.mesh, resolution, bounds, interpolation)
    nx, ny, nz = shape
    slots = fields.slot_of_leaf[fields.leaf_ids]
    # Include the previously yielded slab while constructing its successor.
    retained = nx*ny*(8*len(selected)+9)
    footprint = fields.nbytes+fields.mesh.nbytes+slots.nbytes+8*len(selected)+retained+nx*ny*8+128*ny+144
    with worker_context(workers) as executor:
        for iz in range(nz):
            fields._check()
            values, valid = output_arrays((nx, ny, 1), len(selected), None, (), footprint, memory_limit)
            values.fill(np.nan)
            valid.fill(False)
            owners = np.full((nx, ny, 1), -1, dtype=np.int64)
            write_blocks(fields.mesh, fields.leaf_ids, slots, fields.values, fields.storage_halo,
                         selected, lower, step, interpolation, values, valid,
                         workers=workers, executor=executor, owners=owners, z_offset=iz)
            origin = lower.copy()
            origin[2] = lower[2]+(iz+.5)*step[2]
            plane = Plane(origin, [upper[0]-lower[0], 0., 0.], [0., upper[1]-lower[1], 0.], (nx, ny))
            # Owners describe geometry even when the supplied Fields lack coverage.
            # Fill missing owners using geometry alone, one row at a time.
            for ix in range(nx):
                missing = np.flatnonzero(~valid[ix, :, 0])
                if len(missing):
                    points = np.empty((len(missing), 3))
                    points[:, 0] = lower[0]+(ix+.5)*step[0]
                    points[:, 1] = lower[1]+(missing+.5)*step[1]
                    points[:, 2] = origin[2]
                    owners[ix, missing, 0] = fields.mesh.locate(points)
            yield iz, SliceResult(plane, values[:, :, 0], valid[:, :, 0], owners[:, :, 0])
            del values, valid, owners
