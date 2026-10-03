"""Uniform output validation and interior delivery shared by resident and source workflows."""

from dataclasses import dataclass
import math
import numpy as np

from ._validation import admit, workers_count
from ._execution import worker_context, run_ranges
from .geometry import _uniform_geometry
from .fields import require_fields, _component_indices, _input_arrays


@dataclass(frozen=True)
class UniformResult:
    """Collected cell-center uniform volume with coverage.

    Attributes
    ----------
    values : ndarray
        Float64 values (nx, ny, nz, component).
    valid : ndarray
        Coverage mask (nx, ny, nz).
    lower, upper : ndarray
        Physical sampling bounds.
    definitions : tuple of FieldDefinition
        Output components and units.
    source_identity : object
        In-memory value association.
    """
    values: np.ndarray
    valid: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    definitions: tuple
    source_identity: object

    @property
    def usable(self):
        """Coverage-valid cells whose sampled components are all finite."""
        return self.valid & np.isfinite(self.values).all(axis=-1)

    @property
    def spacing(self):
        """Physical uniform-cell spacing along x, y and z.
        """
        return (self.upper-self.lower)/np.asarray(self.valid.shape)

    @property
    def axes(self):
        """Cell-center coordinate arrays along x, y and z.
        """
        return tuple(lo+(np.arange(count)+.5)*step
                     for lo,count,step in zip(self.lower,self.valid.shape,self.spacing))




def geometry(mesh, resolution, bounds, interpolation):
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
    bounds = None if bounds is None else tuple(bounds)
    shape, lower, upper, step = geometry(fields.mesh, resolution, bounds, interpolation)
    slots = fields.slot_of_leaf[fields.leaf_ids]
    values, valid = output_arrays(shape, len(selected), output, (*_input_arrays(fields), *(bounds or ()), lower, upper),
        fields.nbytes+fields.mesh.nbytes+slots.nbytes+8*len(selected), memory_limit)
    values.fill(np.nan)
    valid.fill(False)
    with worker_context(workers) as executor:
        write_blocks(fields.mesh, fields.leaf_ids, slots, fields.values, fields.storage_halo,
                     selected, lower, step, interpolation, values, valid,
                     workers=workers, executor=executor)
    return UniformResult(values, valid, lower, upper, tuple(fields.fields[i] for i in selected), fields.value_identity)
