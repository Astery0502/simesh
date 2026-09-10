"""Identified geometry and result associations over explicit native consumers.

These functions leave source preparation and numerical choices with the caller.
Connectivity diagnostics do not retain trajectories; selected points can be
traced separately using the existing accepted-prefix integration contract.
"""

from dataclasses import dataclass, field
import math
import numpy as np

from .geometry import PointSet, RaySet, LineSet
from .fields import require_fields, require_continuous, _input_arrays
from .operators.sampling import sample as sample_values
from .connectivity import line_diagnostics, iter_line_diagnostics, QSLResult
from .slices import Plane, _uniform_geometry
from .tracing import _iter_path_segments, Termination, _validate_vector, _validate_inputs
from ._validation import admit, remaining, workers_count, array_bytes
from ._execution import worker_context, run_ranges, native_dispatch
from .projection import LOSStatus

__all__ = ["PointSet","RaySet","LineSet","SampledPoints","ConnectivityMap","RayResult",
           "UniformResult","sample","field_map","uniform_grid","surface_diagnostics","bottom_diagnostics",
           "connectivity","iter_connectivity","trace","los","thermal_los"]


def _points(points):
    if not isinstance(points,PointSet):
        raise TypeError("points must be a PointSet")
    return points


@dataclass(frozen=True)
class SampledPoints:
    """Sampled values associated with identified geometry.

    Attributes
    ----------
    points : PointSet
        Original sampling geometry and IDs.
    values : ndarray
        Values (n, k); definitions supplies component meanings.
    owners, valid : ndarray
        Original owner IDs and coverage mask (n,).
    definitions : tuple of FieldDefinition
        Selected component definitions.
    source_identity : object
        In-memory value token, not snapshot verification.
    """
    points: PointSet
    values: np.ndarray
    owners: np.ndarray
    valid: np.ndarray
    definitions: tuple
    source_identity: object

    @property
    def usable(self):
        """Coverage-valid points whose sampled components are all finite."""
        return self.valid & np.isfinite(self.values).all(axis=-1)

    @property
    def image(self):
        """Return values in the original point/image layout; no invalid values are filled.
        """
        return self.points.reshape(self.values)

    def select(self, mask):
        """Return selected geometry preserving original IDs, rather than a filtered result object.
        """
        return self.points.select(mask)


@dataclass(frozen=True)
class ConnectivityMap:
    """Identified connectivity diagnostics with validity-aware selection.

    Attributes
    ----------
    points : PointSet
        Diagnostic seeds and image layout.
    data : QSLResult
        Raw Q/twist, endpoint and termination data.
    source_identity : object
        In-memory value association.
    """
    points: PointSet
    data: QSLResult
    source_identity: object

    @property
    def quantities(self):
        """Names of diagnostics actually present in the result.
        """
        return tuple(name for name in ("q","twist") if getattr(self.data,name) is not None)

    @property
    def q_valid(self):
        """Q validity mask; all false when Q was not requested.
        """
        return self.data.valid if self.data.q is not None else np.zeros(len(self.points),dtype=bool)

    @property
    def twist_valid(self):
        """Complete-line and finite-twist mask; all false when twist was not requested.
        """
        return (self.data.complete & np.isfinite(self.data.twist) if self.data.twist is not None
                else np.zeros(len(self.points),dtype=bool))

    def image(self, name):
        """Return values in the original point/image layout; no invalid values are filled.
        """
        if name in ("q_valid","twist_valid"):
            return self.points.reshape(getattr(self,name))
        if name not in ("q","log10_q","q_perp","log10_q_perp","twist","length","valid","complete"):
            raise ValueError("select a pointwise diagnostic quantity")
        values = getattr(self.data,name)
        if values is None:
            raise ValueError(f"{name} was not computed")
        return self.points.reshape(values)

    def select(self, mask):
        """Return selected geometry preserving original IDs, rather than a filtered result object.
        """
        return self.points.select(mask)

    def threshold(self, *, q_min=None, abs_twist_min=None, mode="any"):
        """Select valid Q or complete finite twist; thresholds are inclusive."""
        if mode not in ("any","all") or (q_min is None and abs_twist_min is None):
            raise ValueError("provide thresholds and choose any/all")
        masks = []
        for value in (q_min,abs_twist_min):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("thresholds must be finite and nonnegative")
        if q_min is not None:
            if self.data.q is None:
                raise ValueError("Q was not computed")
            masks.append(self.q_valid & (self.data.q >= q_min))
        if abs_twist_min is not None:
            if self.data.twist is None:
                raise ValueError("twist was not computed")
            masks.append(self.twist_valid & (np.abs(self.data.twist) >= abs_twist_min))
        mask = np.logical_or.reduce(masks) if mode == "any" else np.logical_and.reduce(masks)
        return self.points.select(mask)


@dataclass(frozen=True)
class RayResult:
    """Identified scalar or thermal ray integrals.

    Attributes
    ----------
    rays : RaySet
        Origin IDs, directions and requested clipping.
    values, entry, exit : ndarray
        Per-ray integral and clipped coordinate depths.
    status, samples, misses : ndarray
        LOSStatus and diagnostic counts per ray.
    units, quadrature : str
        Physical output unit and selected quadrature/reconstruction.
    metadata : dict
        Recorded model and physical choices.
    source_identity : object
        In-memory value association.
    """
    rays: RaySet
    values: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    status: np.ndarray
    samples: np.ndarray
    misses: np.ndarray
    units: str
    quadrature: str
    source_identity: object
    metadata: dict = field(default_factory=dict)

    @property
    def valid(self):
        """Rays with COMPLETE or EMPTY status.
        """
        return np.isin(self.status,(LOSStatus.COMPLETE,LOSStatus.EMPTY))

    @property
    def complete(self):
        """Whether every ray has a valid COMPLETE or EMPTY status.
        """
        return bool(self.valid.all())

    @property
    def image(self):
        """Return values in the original point/image layout; no invalid values are filled.
        """
        return self.rays.origins.reshape(self.values)

    def select(self, mask):
        """Return selected geometry preserving original IDs, rather than a filtered result object.
        """
        return self.rays.select(mask)


def sample(fields, points, *, components=None, output=None, workers=1, memory_limit=None):
    """Sample physical points and retain their IDs and image layout.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    points : PointSet
        Owned finite positions and stable IDs, optionally with a plane/image layout.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    output : tuple of ndarray, optional
        Writable contiguous values (n, k), owners (n,) and valid (n,) arrays; no input
        aliases. Failure may leave partial writes.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    SampledPoints
        Flat sampled arrays with original geometry, definitions and coverage; image
        restores the layout and usable also checks finiteness.
    """
    _points(points)
    selected = require_continuous(fields,components)
    values,owners,valid = sample_values(fields,points.positions,components=selected,output=output,workers=workers,
                                       memory_limit=remaining(memory_limit,points.nbytes))
    return SampledPoints(points,values,owners,valid,tuple(fields.fields[i] for i in selected),fields.value_identity)


def _diagnostic_support(fields, quantities, controls):
    from .connectivity import _require_support, _quantities
    if "twist" in controls:
        if quantities is not None:
            raise ValueError("use quantities alone to select diagnostics; do not also pass twist")
        twist = controls.pop("twist")
        if type(twist) is not bool:
            raise ValueError("twist must be boolean")
        quantities = ("q", "twist") if twist else ("q",)
    names = _quantities(("q", "twist") if quantities is None else quantities)
    _require_support(fields, twist="twist" in names, curl_field=controls.get("curl_field"))
    return names


def connectivity(fields, points, *, quantities=None, memory_limit=None, **controls):
    """Compute diagnostics without retaining paths; select and trace afterward.

    Parameters
    ----------
    fields : Fields
        Three-component vector field meeting the [qsl][simesh.qsl] support requirements.
    points : PointSet
        Owned finite positions and stable IDs, optionally with a plane/image layout.
    quantities : sequence of str, optional
        q, twist, or both. None selects both unless the compatibility twist control is supplied.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    **controls : object
        Forwarded [qsl][simesh.qsl] controls. The boolean twist control is accepted only
        when quantities is None; prefer quantities. backend/schedule are not accepted.

    Returns
    -------
    ConnectivityMap
        Identified diagnostics with validity-aware selection; full trajectories are not
        retained.
    """
    quantities=_diagnostic_support(fields,quantities,controls)
    _points(points)
    limit = remaining(memory_limit,points.nbytes)
    data = line_diagnostics(fields,points.positions,quantities=quantities,memory_limit=limit,**controls)
    return ConnectivityMap(points,data,fields.value_identity)


def iter_connectivity(fields, points, *, quantities=None, seed_batch=256, memory_limit=None, **controls):
    """Yield owned maps whose point IDs remain global across batches.

    Parameters
    ----------
    fields : Fields
        Three-component vector field meeting the [qsl][simesh.qsl] support requirements.
    points : PointSet
        Owned finite positions and stable IDs, optionally with a plane/image layout.
    quantities : sequence of str, optional
        q, twist, or both. None selects both unless the compatibility twist control is supplied.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    **controls : object
        Forwarded [qsl][simesh.qsl] controls. The boolean twist control is accepted only
        when quantities is None; prefer quantities. backend/schedule are not accepted.

    Returns
    -------
    iterator of ConnectivityMap
        Identified diagnostics with validity-aware selection; full trajectories are not
        retained. Batches own their results, while input fields remain resident.
    """
    quantities=_diagnostic_support(fields,quantities,controls)
    _points(points)
    reserved = points.nbytes+min(seed_batch,len(points))*64 if type(seed_batch) is int and seed_batch>0 else points.nbytes
    limit = remaining(memory_limit,reserved)
    batches = iter_line_diagnostics(fields,points.positions,quantities=quantities,
                                   seed_batch=seed_batch,memory_limit=limit,**controls)
    start = 0
    try:
        for data in batches:
            stop = start+len(data.seeds)
            normals = points.normals
            if normals is not None and normals.ndim == 2:
                normals = normals[start:stop]
            subset = PointSet(data.seeds,points.ids[start:stop],normals=normals,plane=points.plane)
            yield ConnectivityMap(subset,data,fields.value_identity)
            start = stop
    finally:
        batches.close()


def _surface(fields, surface, memory_limit):
    require_fields(fields)
    if isinstance(surface,Plane):
        return PointSet.from_plane(surface,memory_limit=remaining(memory_limit,fields.mesh.nbytes+fields.nbytes))
    return _points(surface)


def field_map(fields, surface, *, components=None, output=None, workers=1, memory_limit=None):
    """Sample a field or precomputed diagnostic on a Plane or PointSet.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    surface : Plane or PointSet
        Pixel-center plane or identified sampling positions.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    output : tuple of ndarray, optional
        Writable contiguous values (n, k), owners (n,) and valid (n,) arrays; no input
        aliases. Failure may leave partial writes.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    SampledPoints
        Flat sampled arrays with original geometry, definitions and coverage; image
        restores the layout and usable also checks finiteness.
    """
    selected=require_continuous(fields,components,operation="field_map")
    return sample(fields,_surface(fields,surface,memory_limit),components=selected,output=output,workers=workers,memory_limit=memory_limit)


def surface_diagnostics(fields, surface, *, quantities=None, memory_limit=None, **controls):
    """Compute selected Q/twist products on an arbitrary sampling surface.

    Parameters
    ----------
    fields : Fields
        Three-component vector field meeting the [qsl][simesh.qsl] support requirements.
    surface : Plane or PointSet
        Surface positions at which to start diagnostics.
    quantities : sequence of str, optional
        q, twist, or both. None selects both unless the compatibility twist control is supplied.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    **controls : object
        Forwarded [qsl][simesh.qsl] controls. The boolean twist control is accepted only
        when quantities is None; prefer quantities. backend/schedule are not accepted.

    Returns
    -------
    ConnectivityMap
        Identified diagnostics with validity-aware selection; full trajectories are not
        retained.
    """
    names=_diagnostic_support(fields,quantities,controls)
    return connectivity(fields,_surface(fields,surface,memory_limit),quantities=names,
                        memory_limit=memory_limit,**controls)


def bottom_diagnostics(fields, shape=(128,128), *, quantities=None, ids=None,
                       memory_limit=None, **controls):
    """Compute diagnostics on the physical z-min face with pixel-center seeds.

    Parameters
    ----------
    fields : Fields
        Three-component vector field meeting the [qsl][simesh.qsl] support requirements.
    shape : tuple of int
        Pixel counts on the physical z-min face.
    quantities : sequence of str, optional
        q, twist, or both. None selects both unless the compatibility twist control is supplied.
    ids : array-like, optional
        Unique IDs for the generated boundary seeds.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    **controls : object
        Forwarded [qsl][simesh.qsl] controls. The boolean twist control is accepted only
        when quantities is None; prefer quantities. backend/schedule are not accepted.

    Returns
    -------
    ConnectivityMap
        Identified diagnostics with validity-aware selection; full trajectories are not
        retained.
    """
    names=_diagnostic_support(fields,quantities,controls)
    points = PointSet.boundary(fields.mesh,"zmin",shape,ids=ids,
                              memory_limit=remaining(memory_limit,fields.mesh.nbytes+fields.nbytes))
    return connectivity(fields,points,quantities=names,memory_limit=memory_limit,**controls)


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


def uniform_grid(fields, resolution, *, components=None, output=None, bounds=None,
                 interpolation="linear", workers=1, tile_rows=64, memory_limit=None):
    """Write a sampled volume directly to final arrays, optionally caller-owned.

    Parameters
    ----------
    fields : Fields
        Selected components; linear sampling requires continuous values and one valid
        halo. Zero/native output reads interiors only, including categorical values.
    resolution : sequence of int
        Positive uniform-grid cell counts (nx, ny, nz).
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    output : tuple of ndarray, optional
        Writable contiguous float64 values (nx, ny, nz, k) and bool valid (nx, ny, nz);
        no input aliases. Failure may leave partial writes.
    bounds : array-like, optional
        Lower and upper physical bounds; defaults to the original domain.
    interpolation : {"linear", "zero", "native"}
        Linear cell-center sampling (default), containing-cell zero-order sampling,
        or exact block placement. Native requires matching spacing on every leaf
        and cell-aligned bounds. Zero-order coarsening is not conservative averaging.
    workers : int
        Number of workers over disjoint ranges.
    tile_rows : int
        Maximum rows per linear-sampling tile; zero/native modes write by leaf.
        Collected output still occupies memory.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    UniformResult
        Complete cell-center volume with component definitions and coverage, even when
        computation is tiled.
    """
    from ._uniform import resident, output_arrays
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    if interpolation != "linear":
        return resident(fields, resolution, components, output, bounds, interpolation, workers, memory_limit)
    from .operators.sampling import _sample
    selected=np.asarray(require_continuous(fields,components,operation="uniform_grid"),dtype=np.int64)
    resolution,lower,upper = _uniform_geometry(fields.mesh,resolution,bounds)
    nx,ny,nz=resolution
    count=len(selected)
    scratch=min(nx,tile_rows)*ny*128
    values,valid=output_arrays(resolution,count,output,(*_input_arrays(fields),lower,upper),
        fields.nbytes+fields.mesh.nbytes+scratch,memory_limit)
    width=upper-lower
    v=(np.arange(ny)+.5)/ny
    with worker_context(workers) as executor:
        for iz in range(nz):
            origin=lower.copy()
            origin[2]=lower[2]+(iz+.5)*(width[2]/nz)
            for first in range(0,nx,tile_rows):
                last=min(nx,first+tile_rows)
                u=(np.arange(first,last)+.5)/nx
                positions=origin+u[:,None,None]*np.array([width[0],0.,0.])+v[None,:,None]*np.array([0.,width[1],0.])
                owners=np.empty((last-first)*ny,dtype=np.int64)
                target=(values[first:last,:,iz].reshape(-1,count),owners,valid[first:last,:,iz].reshape(-1))
                _sample(fields,positions.reshape(-1,3),workers,executor,selected,target)
    return UniformResult(values,valid,lower.copy(),upper.copy(),
                         tuple(fields.fields[i] for i in selected),fields.value_identity)


def trace(fields, points, *, direction="both", step, max_steps=1000, max_length=np.inf,
          null_threshold=0., workers=1, backend="threadpool", schedule="static",
          seed_batch=128, memory_limit=None):
    """Trace selected points into compact branches, preserving IDs.

    Parameters
    ----------
    fields : Fields
        Exactly three ordered continuous vector components with common units and one valid halo.
    points : PointSet
        Owned finite positions and stable IDs, optionally with a plane/image layout.
    direction : str
        both, along, against or inward; inward needs seeds on exactly one physical domain
        face.
    step : float
        Positive integration step in coordinate-length units.
    max_steps : int
        Maximum accepted integration steps per branch.
    max_length : float
        Maximum integrated coordinate length per branch.
    null_threshold : float
        Vector norm at or below which tracing reports a null field.
    workers : int
        Number of workers over disjoint ranges.
    backend : str
        Execution backend: threadpool or explicitly built openmp.
    schedule : str
        Native scheduling policy: static or dynamic.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    LineSet
        Compact against/along branches; LineSet defines unrequested/tangent codes and
        tracing termination.

    Notes
    -----
    Exact upper-face seeds are evaluated at the nearest interior representable coordinate; stored initial positions remain unchanged.
    """
    _points(points)
    _validate_vector(fields)
    if direction not in ("both","along","against","inward"):
        raise ValueError("direction must be both/along/against/inward")
    if type(seed_batch) is not int or seed_batch < 1:
        raise ValueError("seed_batch must be positive")
    _validate_inputs(points.positions,points.ids,step,max_steps,max_length,null_threshold,
                     1,workers,seed_batch,True,False)
    native_dispatch(backend,schedule)
    base = fields.mesh.nbytes+fields.nbytes+points.nbytes+len(points)*256
    admit(base,memory_limit,"trace geometry")
    positions = points.positions.copy()
    for axis in range(3):
        upper = positions[:,axis] == fields.mesh.upper[axis]
        positions[upper,axis] = np.nextafter(fields.mesh.upper[axis],fields.mesh.lower[axis])
    requested = np.zeros((len(points),2),dtype=bool)
    status = np.full((len(points),2),LineSet.NOT_REQUESTED,dtype=np.int64)
    if direction == "both":
        requested[:] = True
    elif direction in ("along","against"):
        requested[:,int(direction == "along")] = True
    else:
        lo,hi = fields.mesh.lower,fields.mesh.upper
        tolerance = 32*np.finfo(float).eps*max(np.max(np.abs([lo,hi])),np.max(hi-lo))
        near_lo = np.abs(points.positions-lo) <= tolerance
        near_hi = np.abs(points.positions-hi) <= tolerance
        if np.any(points.positions<lo) or np.any(points.positions>hi) or np.any((near_lo|near_hi).sum(axis=1) != 1):
            raise ValueError("inward requires points on exactly one physical box face")
        normals = near_hi.astype(float)-near_lo.astype(float)
        b,_,valid = sample_values(fields,positions)
        with np.errstate(over="ignore",invalid="ignore"):
            dot = np.sum(normals*b,axis=1)
            norm = np.hypot(np.hypot(b[:,0],b[:,1]),b[:,2])
        tangent = valid & np.isfinite(norm) & (norm > null_threshold) & (dot == 0)
        status[tangent] = LineSet.TANGENT_SEED
        requested[:,0] = (dot > 0) & ~tangent
        requested[:,1] = (~(dot > 0)) & ~tangent
    pieces = [[] for _ in range(2*len(points))]
    counts = np.zeros(2*len(points),dtype=np.int64)
    retained = 0
    for side in range(2):
        rows = np.flatnonzero(requested[:,side])
        for start in range(0,len(rows),seed_batch):
            selected = rows[start:start+seed_batch]
            segments = _iter_path_segments(fields,np.ascontiguousarray(positions[selected]),points.ids[selected],
                step=step,max_steps=max_steps,direction=2*side-1,
                max_length=max_length,null_threshold=null_threshold,workers=workers,backend=backend,schedule=schedule,
                memory_limit=remaining(memory_limit,base-fields.nbytes-fields.mesh.nbytes+retained))
            try:
                for state,paths,point_counts in segments:
                    raw_bytes = array_bytes(vars(state).values())
                    new_bytes = int(point_counts.sum())*24
                    admit(base+retained+raw_bytes+new_bytes,memory_limit,"packed trace output")
                    for index,row in enumerate(selected):
                        count = int(point_counts[index])
                        branch = 2*int(row)+side
                        if count:
                            path = paths[index,:count].copy()
                            if counts[branch] == 0:
                                path[0] = points.positions[row]
                            pieces[branch].append(path)
                            counts[branch] += count
                        status[row,side] = state.status[index]
                    retained += new_bytes
            finally:
                segments.close()
            del state,paths
    # Packing overlaps retained segments; validation starts after their release.
    admit(base+2*retained+retained//8,memory_limit,"collected trace output")
    offsets = np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    packed = np.empty((int(offsets[-1]),3))
    for index,segments in enumerate(pieces):
        cursor = int(offsets[index])
        for path in segments:
            packed[cursor:cursor+len(path)] = path
            cursor += len(path)
        segments.clear()
    pieces.clear()
    path = None
    return LineSet(points,packed,offsets,status,fields.value_identity)


def los(fields, rays, *, component=0, quadrature="gauss2", step_fraction=.5,
        max_samples=1000000, workers=1, ray_batch=4096, memory_limit=None):
    """Integrate identified rays with shared or per-ray directions.

    Parameters
    ----------
    fields : Fields
        Selected continuous scalar component with at least one valid halo.
    rays : RaySet
        Identified origins, normalized shared/per-ray directions and near/far distances.
    component : str or int
        Continuous scalar field name or local component index.
    quadrature : str
        gauss2 splits at interpolation knots; midpoint uses step_fraction.
    step_fraction : float
        Positive local step fraction; its role depends on the selected integration method.
    max_samples : int
        Per-ray sampling limit; reaching the limit is not a complete integral.
    workers : int
        Number of workers over disjoint ranges.
    ray_batch : int
        Maximum rays per computation batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    RayResult
        Identified values and clipping intervals in field units times coordinate length;
        RayResult.valid interprets COMPLETE/EMPTY statuses.
    """
    from ._kernels.native import integrate_ray_set
    component, = require_continuous(fields,(component,),operation="LOS")
    if not isinstance(rays,RaySet):
        raise TypeError("rays must be a RaySet")
    workers_count(workers)
    if (quadrature not in ("gauss2","midpoint") or type(component) is not int or
            not 0 <= component < len(fields.fields) or not np.isfinite(step_fraction) or step_fraction <= 0 or
            type(max_samples) is not int or not 1 <= max_samples <= np.iinfo(np.int64).max or
            type(ray_batch) is not int or ray_batch < 1):
        raise ValueError("invalid LOS component, quadrature or batch controls")
    n = len(rays.origins)
    admit(fields.mesh.nbytes+fields.nbytes+rays.nbytes+n*48+min(n,ray_batch)*384,
          memory_limit,"ray-set LOS")
    values,entry,exit = (np.empty(n) for _ in range(3))
    status,samples,misses = (np.empty(n,dtype=np.int64) for _ in range(3))
    m = fields.mesh
    all_directions = np.broadcast_to(rays.directions,(n,3))
    with worker_context(workers) as executor:
        for start in range(0,n,ray_batch):
            stop = min(start+ray_batch,n)
            directions = np.ascontiguousarray(all_directions[start:stop])
            def run(first,last):
                s = slice(start+first,start+last)
                integrate_ray_set(m.lower,m.upper,m.roots,m.children,m.node_leaves,m.node_lower,m.node_upper,
                    m.bounds,m.spacing,fields.slot_of_leaf,fields.values,fields.storage_halo,
                    rays.origins.positions[s],directions[first:last],rays.near[s],rays.far[s],
                    component,step_fraction,int(quadrature=="gauss2"),max_samples,
                    values[s],entry[s],exit[s],status[s],samples[s],misses[s])
            run_ranges(stop-start,workers,run,executor)
    return RayResult(rays,values,entry,exit,status,samples,misses,
                     fields.fields[component].units+" * coordinate-length",quadrature,fields.value_identity)


def thermal_los(thermodynamics, rays, *, length_unit_cm, model=None,
                order="thermodynamics-first", subdivisions=4, max_samples=1000000,
                workers=1, ray_batch=4096, memory_limit=None):
    """Apply the existing native AIA171 reconstruction to identified rays.

    Parameters
    ----------
    thermodynamics : Fields
        Matching number-density/kelvin fields from thermal_fields, with valid
        interpolation support.
    rays : RaySet
        Identified rays and coordinate-distance clipping.
    length_unit_cm : float
        Centimeters per coordinate-length unit.
    model : AIA171, optional
        Model matching the thermal fields; None selects the default AIA171 model.
    order : str
        thermodynamics-first interpolates n,T before n²R(T); emissivity-first applies
        response at nodes before interpolation.
    subdivisions : int
        Composite Gauss2 subdivisions for nonlinear thermal response integration.
    max_samples : int
        Per-ray sampling limit; reaching the limit is not a complete integral.
    workers : int
        Number of workers over disjoint ranges.
    ray_batch : int
        Maximum rays per computation batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    RayResult
        Historical AIA171 brightness in DN s^-1 pixel^-1, with identified rays and status.

    Notes
    -----
    These orders define different reconstructions. Composite Gauss2 for nonlinear response is an approximation; check convergence and result status.
    """
    from dataclasses import replace
    from .physics.thermal import (AIA171, _check_thermal, emissivity_fields,
                                  _RESPONSE_GRID, _LOG_RESPONSE, _RESPONSE_SLOPES, _scale_thermal_values)
    from ._kernels.native import initialize_ray_set
    from ._kernels.thermal_rays import integrate_ray_set_ready
    model = AIA171() if model is None else model
    _check_thermal(thermodynamics,model)
    if not isinstance(rays,RaySet):
        raise TypeError("rays must be a RaySet")
    workers_count(workers)
    if (type(model) is not AIA171 or order not in ("thermodynamics-first","emissivity-first") or
            not np.isfinite(length_unit_cm) or length_unit_cm <= 0 or
            type(subdivisions) is not int or not 1 <= subdivisions <= np.iinfo(np.int64).max//2 or
            type(max_samples) is not int or not 1 <= max_samples <= np.iinfo(np.int64).max or
            type(ray_batch) is not int or ray_batch < 1):
        raise ValueError("native AIA171, positive units and valid quadrature/batch controls are required")
    if order == "emissivity-first":
        emissivity = emissivity_fields(thermodynamics,model=model,
                                      memory_limit=remaining(memory_limit,rays.nbytes+len(rays.origins)*32))
        result = los(emissivity,rays,max_samples=max_samples,workers=workers,ray_batch=ray_batch,
                     memory_limit=remaining(memory_limit,thermodynamics.nbytes+len(rays.origins)*32))
        result = replace(result,source_identity=thermodynamics.value_identity,
                         quadrature="emissivity-first/gauss2")
    else:
        n = len(rays.origins)
        mesh = thermodynamics.mesh
        admit(mesh.nbytes+thermodynamics.nbytes+rays.nbytes+n*96+min(n,ray_batch)*128+workers*65536,
              memory_limit,"ray-set thermal LOS")
        values,entry,exit = (np.zeros(n) for _ in range(3))
        status,samples,misses = (np.zeros(n,dtype=np.int64) for _ in range(3))
        all_directions = np.broadcast_to(rays.directions,(n,3))
        with worker_context(workers) as executor:
            for start in range(0,n,ray_batch):
                stop = min(start+ray_batch,n)
                directions = np.ascontiguousarray(all_directions[start:stop])
                def run(first,last):
                    s = slice(start+first,start+last)
                    origins = rays.origins.positions[s]
                    direction = directions[first:last]
                    initialize_ray_set(mesh.lower,mesh.upper,origins,direction,rays.near[s],rays.far[s],
                                       entry[s],exit[s],status[s])
                    output = values[s]
                    output[status[s] >= LOSStatus.MISSING_COVERAGE] = np.nan
                    integrate_ray_set_ready(mesh.roots,mesh.children,mesh.node_leaves,mesh.node_lower,mesh.node_upper,
                        mesh.bounds,mesh.spacing,thermodynamics.slot_of_leaf,thermodynamics.values,
                        thermodynamics.storage_halo,origins,direction,entry[s],exit[s],subdivisions,max_samples,
                        _RESPONSE_GRID,_LOG_RESPONSE,_RESPONSE_SLOPES,values[s],status[s],samples[s])
                run_ranges(stop-start,workers,run,executor)
        result = RayResult(rays,values,entry,exit,status,samples,misses,"DN s^-1 pixel^-1",
                           f"thermodynamics-first/composite-gauss2/{subdivisions}",thermodynamics.value_identity)
    values,flags = _scale_thermal_values(result.values,result.status,result.valid,length_unit_cm)
    return replace(result,values=values,status=flags,units="DN s^-1 pixel^-1",
                   metadata={"model":model.identity,"temperature_label":thermodynamics.source[2],
                             "density_unit_g_cm3":thermodynamics.preparation_stats["density_unit_g_cm3"],
                             "length_unit_cm":float(length_unit_cm)})


def iter_lines(fields, points, *, seed_batch=128, memory_limit=None, **controls):
    """Yield compact LineSet shards in input order, preserving global seed IDs.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    points : PointSet
        Owned finite positions and stable IDs, optionally with a plane/image layout.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    **controls : object
        Forwarded controls from simesh.applications.trace, including required step.

    Returns
    -------
    iterator of LineSet
        Owned compact seed batches with global IDs; input fields remain resident.
    """
    _validate_vector(fields)
    _points(points)
    if type(seed_batch) is not int or seed_batch<1:
        raise ValueError("seed_batch must be positive")
    previous=0
    for first in range(0,len(points),seed_batch):
        last=min(first+seed_batch,len(points))
        normals=points.normals
        if normals is not None and normals.ndim==2:
            normals=normals[first:last]
        subset=PointSet(points.positions[first:last],points.ids[first:last],normals=normals,plane=points.plane)
        result=trace(fields,subset,seed_batch=seed_batch,
                     memory_limit=remaining(memory_limit,points.nbytes+previous),**controls)
        previous=result.nbytes
        yield result
        del result

__all__ += ["iter_lines"]
