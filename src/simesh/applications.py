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
from .connectivity import qsl, iter_qsl, line_diagnostics, iter_line_diagnostics, QSLResult
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
        return self.points.reshape(self.values)

    def select(self, mask):
        return self.points.select(mask)


@dataclass(frozen=True)
class ConnectivityMap:
    points: PointSet
    data: QSLResult
    source_identity: object

    @property
    def quantities(self):
        return tuple(name for name in ("q","twist") if getattr(self.data,name) is not None)

    @property
    def q_valid(self):
        return self.data.valid if self.data.q is not None else np.zeros(len(self.points),dtype=bool)

    @property
    def twist_valid(self):
        return (self.data.complete & np.isfinite(self.data.twist) if self.data.twist is not None
                else np.zeros(len(self.points),dtype=bool))

    def image(self, name):
        if name in ("q_valid","twist_valid"):
            return self.points.reshape(getattr(self,name))
        if name not in ("q","log10_q","q_perp","log10_q_perp","twist","length","valid","complete"):
            raise ValueError("select a pointwise diagnostic quantity")
        values = getattr(self.data,name)
        if values is None:
            raise ValueError(f"{name} was not computed")
        return self.points.reshape(values)

    def select(self, mask):
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
        return np.isin(self.status,(LOSStatus.COMPLETE,LOSStatus.EMPTY))

    @property
    def complete(self):
        return bool(self.valid.all())

    @property
    def image(self):
        return self.rays.origins.reshape(self.values)

    def select(self, mask):
        return self.rays.select(mask)


def sample(fields, points, *, components=None, output=None, workers=1, memory_limit=None):
    """Sample physical points and retain their IDs and image layout."""
    _points(points)
    selected = require_continuous(fields,components)
    values,owners,valid = sample_values(fields,points.positions,components=selected,output=output,workers=workers,
                                       memory_limit=remaining(memory_limit,points.nbytes))
    return SampledPoints(points,values,owners,valid,tuple(fields.fields[i] for i in selected),fields.value_identity)


def _diagnostic_support(fields, quantities, controls):
    from .connectivity import _require_support, _quantities
    names=None if quantities is None else _quantities(quantities)
    _require_support(fields,compute_q=names is None or "q" in names,
        twist=controls.get("twist",True) if names is None else "twist" in names,
        method=controls.get("method","finite-difference"),curl_field=controls.get("curl_field"))
    return names


def connectivity(fields, points, *, quantities=None, memory_limit=None, **controls):
    """Compute diagnostics without retaining paths; select and trace afterward."""
    quantities=_diagnostic_support(fields,quantities,controls)
    _points(points)
    limit = remaining(memory_limit,points.nbytes)
    data = (qsl(fields,points.positions,memory_limit=limit,**controls) if quantities is None else
            line_diagnostics(fields,points.positions,quantities=quantities,memory_limit=limit,**controls))
    return ConnectivityMap(points,data,fields.value_identity)


def iter_connectivity(fields, points, *, quantities=None, seed_batch=256, memory_limit=None, **controls):
    """Yield owned maps whose point IDs remain global across batches."""
    quantities=_diagnostic_support(fields,quantities,controls)
    _points(points)
    reserved = points.nbytes+min(seed_batch,len(points))*64 if type(seed_batch) is int and seed_batch>0 else points.nbytes
    limit = remaining(memory_limit,reserved)
    batches = (iter_qsl(fields,points.positions,seed_batch=seed_batch,memory_limit=limit,**controls)
               if quantities is None else iter_line_diagnostics(fields,points.positions,quantities=quantities,
                   seed_batch=seed_batch,memory_limit=limit,**controls))
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
    """Sample a field or precomputed diagnostic on a Plane or PointSet."""
    selected=require_continuous(fields,components,operation="field_map")
    return sample(fields,_surface(fields,surface,memory_limit),components=selected,output=output,workers=workers,memory_limit=memory_limit)


def surface_diagnostics(fields, surface, *, quantities=("q","twist"), memory_limit=None, **controls):
    """Compute selected Q/twist products on an arbitrary sampling surface."""
    names=_diagnostic_support(fields,quantities,controls)
    return connectivity(fields,_surface(fields,surface,memory_limit),quantities=names,
                        memory_limit=memory_limit,**controls)


def bottom_diagnostics(fields, shape=(128,128), *, quantities=("q","twist"), ids=None,
                       memory_limit=None, **controls):
    """Compute diagnostics on the physical z-min face with pixel-center seeds."""
    names=_diagnostic_support(fields,quantities,controls)
    points = PointSet.boundary(fields.mesh,"zmin",shape,ids=ids,
                              memory_limit=remaining(memory_limit,fields.mesh.nbytes+fields.nbytes))
    return connectivity(fields,points,quantities=names,memory_limit=memory_limit,**controls)


@dataclass(frozen=True)
class UniformResult:
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
        return (self.upper-self.lower)/np.asarray(self.valid.shape)

    @property
    def axes(self):
        return tuple(lo+(np.arange(count)+.5)*step
                     for lo,count,step in zip(self.lower,self.valid.shape,self.spacing))


def uniform_grid(fields, resolution, *, components=None, output=None, bounds=None,
                 workers=1, tile_rows=64, memory_limit=None):
    """Write a sampled volume directly to final arrays, optionally caller-owned.

    output=(values, valid) must match the contiguous volume layout. Failure may
    leave partial writes; only a successful return publishes a UniformResult.
    """
    from .operators.sampling import _sample
    from ._execution import worker_context
    selected=np.asarray(require_continuous(fields,components,operation="uniform_grid"),dtype=np.int64)
    resolution,lower,upper = _uniform_geometry(fields.mesh,resolution,bounds)
    nx,ny,nz=resolution
    if type(tile_rows) is not int or tile_rows<1:
        raise ValueError("tile_rows must be positive")
    count=len(selected)
    output_bytes=math.prod(resolution)*(8*count+1)+48
    scratch=min(nx,tile_rows)*ny*128
    admit(fields.nbytes+fields.mesh.nbytes+output_bytes+scratch,memory_limit,"uniform volume")
    if output is None:
        values=np.empty((*resolution,count))
        valid=np.empty(resolution,dtype=bool)
    else:
        if not isinstance(output,(tuple,list)) or len(output)!=2:
            raise ValueError("output must be (values, valid)")
        values,valid=output
        if any(not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype!=dtype or
               not a.flags.c_contiguous or not a.flags.writeable
               for a,shape,dtype in ((values,(*resolution,count),np.float64),(valid,resolution,np.bool_))):
            raise ValueError("output must match writable contiguous volume values and validity")
        if any(np.shares_memory(a,b) for a in output for b in (*_input_arrays(fields),lower,upper)) or np.shares_memory(values,valid):
            raise ValueError("output must not alias input or other output")
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

    Direction is both/along/against/inward. Inward is defined only on a single
    physical box face per seed. Status -1 denotes an unrequested branch; -2 a
    tangent seed for which inward is undefined. Other statuses are Termination.
    Upper-face seeds enter via the nearest interior representable coordinate;
    displayed initial points keep the caller's exact boundary positions.
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
    # Reserve the packed copy and finite-value validation mask as well as the
    # retained segments. No intermediate complete per-branch arrays are built.
    admit(base+2*retained+retained//8,memory_limit,"collected trace output")
    offsets = np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    packed = np.empty((int(offsets[-1]),3))
    for index,segments in enumerate(pieces):
        cursor = int(offsets[index])
        for path in segments:
            packed[cursor:cursor+len(path)] = path
            cursor += len(path)
    return LineSet(points,packed,offsets,status,fields.value_identity)


def los(fields, rays, *, component=0, quadrature="gauss2", step_fraction=.5,
        max_samples=1000000, workers=1, ray_batch=4096, memory_limit=None):
    """Integrate identified rays with shared or per-ray directions."""
    from ._kernels.native import integrate_ray_set
    require_continuous(fields,(component,),operation="LOS")
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
    values,entry,exit = (np.zeros(n) for _ in range(3))
    status,samples,misses = (np.zeros(n,dtype=np.int64) for _ in range(3))
    m = fields.mesh
    with worker_context(workers) as executor:
        for start in range(0,n,ray_batch):
            stop = min(start+ray_batch,n)
            directions = np.ascontiguousarray(np.broadcast_to(rays.directions,(n,3))[start:stop])
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
    """Apply the existing native AIA171 reconstruction to identified rays."""
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
        with worker_context(workers) as executor:
            for start in range(0,n,ray_batch):
                stop = min(start+ray_batch,n)
                directions = np.ascontiguousarray(np.broadcast_to(rays.directions,(n,3))[start:stop])
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
    """Yield compact LineSet shards in input order, preserving global seed IDs."""
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
