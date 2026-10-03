"""Pixel-center samples and axis-aligned native-cell slices of retained fields."""

from dataclasses import dataclass, field
from math import prod
import numpy as np

from ._validation import admit, workers_count
from .fields import FieldDefinition, require_fields, require_continuous, _input_arrays, _component_indices
from .spatial import Plane
from .geometry import AxisSlice, _uniform_geometry
from ._execution import worker_context
from .operators.sampling import _sample, _sample_output


@dataclass(frozen=True, eq=False)
class AMRSliceResult:
    """Native-cell values associated with block geometry; no plotting dependency.

    Attributes
    ----------
    geometry : AxisSlice
        Full-domain section with original block identity and refinement geometry.
    values : ndarray
        Read-only float64 (nblocks, nu, nv, k) values in transverse XYZ order.
        Missing blocks contain NaN; supplied nonfinite values are preserved.
    valid : ndarray
        Read-only bool (nblocks,) supplied-block coverage, independent of finiteness.
    definitions : tuple of FieldDefinition
        Selected component definitions in output order, without unit conversion.
    source_identity : object
        In-memory field-value association, not persistent snapshot verification.
    representation : str
        Piecewise-constant interpretation of original three-dimensional interiors;
        values are not interpolated point samples or reconstructed face averages.

    Notes
    -----
    slice_axis owns output arrays independently of input leases. Direct construction
    marks arrays read-only but does not remove external writable aliases.
    save_result retains values, coverage and the original mesh topology; load_result
    reconstructs the geometry without accessing a Source.
    """

    geometry: AxisSlice
    values: np.ndarray
    valid: np.ndarray
    definitions: tuple[FieldDefinition, ...]
    source_identity: object
    representation: str = field(default="piecewise-constant leaf interiors", init=False)

    def __post_init__(self):
        if not isinstance(self.geometry, AxisSlice):
            raise TypeError("AMR slice result requires AxisSlice geometry")
        definitions = tuple(self.definitions)
        shape = (len(self.geometry.leaf_ids), *self.geometry.block_shape, len(definitions))
        if (not definitions or not all(isinstance(d, FieldDefinition) for d in definitions)
                or not isinstance(self.values, np.ndarray) or self.values.shape != shape
                or self.values.dtype != np.float64 or not self.values.flags.c_contiguous
                or not isinstance(self.valid, np.ndarray) or self.valid.shape != shape[:1]
                or self.valid.dtype != np.bool_ or not self.valid.flags.c_contiguous):
            raise ValueError("AMR slice arrays and definitions must match geometry")
        self.values.flags.writeable = self.valid.flags.writeable = False
        object.__setattr__(self, "definitions", definitions)

    @property
    def usable(self):
        """Cell mask (nblocks, nu, nv): covered with all selected components finite."""
        return self.valid[:, None, None] & np.isfinite(self.values).all(axis=-1)

    @property
    def nbytes(self):
        """Accounted output and geometry arrays, including the shared Mesh."""
        return self.geometry.nbytes + self.values.nbytes + self.valid.nbytes


def slice_axis(fields, axis, coordinate, *, components=None, side="positive", memory_limit=None):
    """Extract an owned axis-aligned AMR slice without interpolation or source access.

    Parameters
    ----------
    fields : Fields
        Completed interiors; no valid halo is required, including categorical fields.
        Missing leaves remain in the geometry with false coverage and NaN values.
    axis : int or str
        Normal axis: 0, 1, 2 or x, y, z.
    coordinate : float
        Plane position in original mesh coordinates, including domain faces.
    components : str or int or sequence, optional
        Names or local component indices, in output order; omission selects all.
    side : str
        Positive or negative coordinate-side cell at interfaces, spelled 'positive'
        or 'negative'; physical domain faces always take the interior cell.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    AMRSliceResult
        Complete block sections using piecewise-constant interior values, with
        owned output arrays and shared immutable Mesh. Input region bounds do not
        clip blocks or create physical boundaries.
    """
    require_fields(fields, operation="slice_axis")
    selected = _component_indices(fields, components)
    mesh = fields.mesh
    footprint = mesh.nbytes + fields.nbytes
    # Reserve cell-edge work and the small index arrays used by Fields.window.
    scratch = 1024 + (max(mesh.block_shape)+1)*32
    admit(footprint + mesh.leaf_count*32 + scratch,
          memory_limit, "axis slice geometry")
    geometry = AxisSlice(mesh, axis, coordinate, side)
    shape = (len(geometry.leaf_ids), *geometry.block_shape, len(selected))
    required = fields.nbytes + geometry.nbytes + prod(shape)*8 + shape[0] + scratch
    admit(required, memory_limit, "axis slice")
    values = np.empty(shape)
    valid = np.empty(shape[0], dtype=bool)
    for row, (leaf, index) in enumerate(zip(geometry.leaf_ids, geometry.cell_indices)):
        valid[row] = fields.slot_of_leaf[leaf] >= 0
        if not valid[row]:
            values[row].fill(np.nan)
            continue
        lo, hi = [0, 0, 0], list(mesh.block_shape)
        lo[geometry.axis], hi[geometry.axis] = int(index), int(index)+1
        section = fields.window(leaf, lo, hi).squeeze(axis=geometry.axis)
        for column, component in enumerate(selected):
            values[row, ..., column] = section[..., component]
    return AMRSliceResult(geometry, values, valid, tuple(fields.fields[i] for i in selected),
                          fields.value_identity)


@dataclass(frozen=True)
class SliceResult:
    """Raw plane samples; not directly accepted by save_result.

    Attributes
    ----------
    plane : Plane
        Sampling geometry.
    values : ndarray
        Component-last image values (*plane.shape, k).
    valid, owners : ndarray
        Image-shaped coverage mask and original owner IDs.
    """
    plane: Plane
    values: np.ndarray
    valid: np.ndarray
    owners: np.ndarray


def _plane_result(plane,count,footprint,sampler,*,tile_rows,workers,executor,memory_limit,output=None,direct=False):
    if not isinstance(plane,Plane) or type(tile_rows) is not int or tile_rows<1:
        raise ValueError("Plane and positive tile_rows required")
    nx,ny=plane.shape
    required=footprint+nx*ny*(count*8+9)+min(nx,tile_rows)*ny*(128 if direct else count*8+128)
    admit(required,memory_limit,"slice")
    if output is None:
        output=(np.empty((nx,ny,count)),np.empty((nx,ny),dtype=np.int64),np.empty((nx,ny),dtype=bool))
    output,owners,valid=output
    v=(np.arange(ny)+.5)/ny
    for first in range(0,nx,tile_rows):
        last=min(first+tile_rows,nx)
        u=(np.arange(first,last)+.5)/nx
        points=plane.origin+u[:,None,None]*plane.u+v[None,:,None]*plane.v
        positions=np.ascontiguousarray(points.reshape(-1,3))
        if direct:
            sampler(positions,workers,executor,(output[first:last].reshape(-1,count),
                    owners[first:last].reshape(-1),valid[first:last].reshape(-1)))
        else:
            values,ids,okay=sampler(positions,workers,executor)
            output[first:last]=values.reshape(last-first,ny,count)
            valid[first:last]=okay.reshape(last-first,ny)
            owners[first:last]=ids.reshape(last-first,ny)
    return SliceResult(plane,output,valid,owners)


def sample_plane(fields,plane,*,components=None,output=None,tile_rows=64,workers=1,memory_limit=None):
    """Sample a plane directly; no input preparation or source access occurs.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    plane : Plane
        Pixel-center sampling plane.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    output : tuple of ndarray, optional
        Writable contiguous (values, owners, valid) on plane.shape, with component-last
        values; no input aliases. Failure may leave partial writes.
    tile_rows : int
        Maximum rows sampled per tile; collected output still occupies memory.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    SliceResult
        Pixel-center values, original owner IDs and coverage on the plane layout.
    """
    selected=np.asarray(require_continuous(fields,components,operation="sample_plane"),dtype=np.int64)
    if not isinstance(plane,Plane):
        raise TypeError("plane must be Plane")
    if output is not None:
        if not isinstance(output,(tuple,list)) or len(output)!=3:
            raise ValueError("output must be (values, owners, valid)")
        shapes=((*plane.shape,len(selected)),plane.shape,plane.shape)
        if any(not isinstance(a,np.ndarray) or a.shape!=shape or not a.flags.c_contiguous
               for a,shape in zip(output,shapes)):
            raise ValueError("plane output must match contiguous plane shapes")
        _sample_output(tuple(a.reshape((-1,len(selected)) if i==0 else (-1,))
                             for i,a in enumerate(output)), int(np.prod(plane.shape)),
                       len(selected),(*_input_arrays(fields),plane.origin,plane.u,plane.v))
    with worker_context(workers) as executor:
        return _plane_result(plane,len(selected),fields.mesh.nbytes+fields.nbytes,
            lambda points,w,e,out:_sample(fields,points,w,e,selected,out),tile_rows=tile_rows,
            workers=workers,executor=executor,memory_limit=memory_limit,output=output,direct=True)


def iter_uniform(fields,resolution,*,components=None,bounds=None,interpolation="linear",tile_rows=64,workers=1,memory_limit=None):
    """Yield owned (z index, SliceResult) slabs without materializing a volume.

    Parameters
    ----------
    fields : Fields
        Selected components; linear sampling requires continuous values and one valid
        halo. Zero/native output reads interiors only, including categorical values.
    resolution : sequence of int
        Positive uniform-grid cell counts (nx, ny, nz).
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    bounds : array-like, optional
        Lower and upper physical bounds; defaults to the original domain.
    interpolation : {"linear", "zero", "native"}
        Linear sampling, containing-cell zero-order sampling, or exact placement.
        Native requires matching spacing on every leaf and cell-aligned bounds.
        Zero-order coarsening is not conservative averaging.
    tile_rows : int
        Maximum rows per linear-sampling tile; zero/native modes write by leaf.
        Collected output still occupies memory.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    iterator of tuple
        Owned (z_index, SliceResult) XY slices in z order; input Fields remain resident
        and no complete volume is collected.
    """
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    if interpolation != "linear":
        yield from _interior_slabs(fields, resolution, components, bounds, interpolation, workers, memory_limit)
        return
    from ._validation import remaining
    selected=np.asarray(require_continuous(fields,components,operation="iter_uniform"),dtype=np.int64)
    (nx,ny,nz),lower,upper=_uniform_geometry(fields.mesh,resolution,bounds)
    width=upper-lower
    previous=nx*ny*(8*len(selected)+9)
    limit=remaining(memory_limit,previous)
    with worker_context(workers) as executor:
        for iz in range(nz):
            fields._check()
            origin=lower.copy()
            origin[2]=lower[2]+(iz+.5)*(width[2]/nz)
            plane=Plane(origin,[width[0],0.,0.],[0.,width[1],0.],(nx,ny))
            yield iz,_plane_result(plane,len(selected),fields.mesh.nbytes+fields.nbytes,
                lambda points,w,e,out:_sample(fields,points,w,e,selected,out),tile_rows=tile_rows,
                workers=workers,executor=executor,memory_limit=limit,direct=True)


def _interior_slabs(fields, resolution, components, bounds, interpolation, workers, memory_limit):
    from ._uniform import geometry, output_arrays, write_blocks
    require_fields(fields)
    selected = np.asarray(_component_indices(fields, components), dtype=np.int64)
    workers_count(workers)
    shape, lower, upper, step = geometry(fields.mesh, resolution, bounds, interpolation)
    nx, ny, nz = shape
    slots = fields.slot_of_leaf[fields.leaf_ids]
    # Include the previously yielded slab while constructing its successor.
    retained = nx*ny*(8*len(selected)+9)
    native_scratch = 48*len(slots) if interpolation == "native" else 0
    footprint = (fields.nbytes+fields.mesh.nbytes+slots.nbytes+8*len(selected)
                 + retained+nx*ny*8+128*ny+144+native_scratch)
    admit(footprint, memory_limit, "uniform slab geometry")
    if interpolation == "native":
        nodes = fields.mesh.leaf_nodes[fields.leaf_ids]
        starts = fields.mesh.node_lower[nodes, 2]
        starts = np.rint((starts-lower[2])/step[2])
        stops = starts+fields.mesh.block_shape[2]
        del nodes
    with worker_context(workers) as executor:
        for iz in range(nz):
            fields._check()
            values, valid = output_arrays((nx, ny, 1), len(selected), None, (), footprint, memory_limit)
            values.fill(np.nan)
            valid.fill(False)
            owners = np.full((nx, ny, 1), -1, dtype=np.int64)
            ids, active_slots = fields.leaf_ids, slots
            if interpolation == "native":
                active = (starts <= iz) & (iz < stops)
                ids, active_slots = ids[active], slots[active]
            # Zero-order ownership stays in the kernel: fused coordinate arithmetic
            # can round differently from a Python prefilter at a block face.
            write_blocks(fields.mesh, ids, active_slots, fields.values, fields.storage_halo,
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
            del values, valid, owners, ids, active_slots
