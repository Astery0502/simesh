"""Pixel-owned scalar LOS integration over explicitly completed fields."""

from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from .fields import require_continuous, _field_index
from .slices import Plane
from ._validation import admit, workers_count
from ._execution import worker_context, run_ranges, native_dispatch

class LOSStatus(IntEnum):
    """Ray states. COMPLETE and EMPTY are valid; missing coverage, sample limits and numerical failures are not.
    """
    RUNNING = 0
    COMPLETE = 1
    EMPTY = 2
    MISSING_COVERAGE = 3
    NONFINITE_SCALAR = 4
    GEOMETRY_FAILURE = 5
    SAMPLE_LIMIT = 6
    UNREPRESENTABLE_INTEGRAL = 7
    UNREPRESENTABLE_SAMPLE = 8


@dataclass(frozen=True)
class LOSResult:
    """Raw plane LOS product; use application LOS for save_result support.

    Attributes
    ----------
    plane, direction : object
        Sampling geometry and normalized direction.
    values, entry, exit : ndarray
        Image-shaped integrals and coordinate clipping depths.
    status, samples, misses : ndarray
        LOSStatus codes and diagnostic counts.
    scalar_units, quadrature : str
        Field units times coordinate length, and integration method.
    """
    plane: Plane
    direction: np.ndarray
    values: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    status: np.ndarray
    samples: np.ndarray
    misses: np.ndarray
    scalar_units: str
    quadrature: str

    @property
    def depth(self):
        """Clipped coordinate-distance interval length for each ray."""
        return self.exit-self.entry

    @property
    def valid(self):
        """Coverage/status mask accepting COMPLETE and EMPTY rays."""
        return (self.status==LOSStatus.COMPLETE)|(self.status==LOSStatus.EMPTY)

    @property
    def complete(self):
        """Whether every ray is valid, including empty rays."""
        return bool(self.valid.all())


def orthographic_plane(lower,upper,direction,shape):
    """Enclose the projection of the entire physical box in a perpendicular Plane.

    Parameters
    ----------
    lower : array-like
        Finite domain lower bounds (3,).
    upper : array-like
        Finite strictly ordered upper bounds (3,).
    direction : array-like
        Finite nonzero viewing direction (3,).
    shape : tuple of int
        Positive plane pixel counts.

    Returns
    -------
    Plane
        Plane enclosing the projection of the whole box, perpendicular to the direction.
    """
    lower,upper = np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    d = np.asarray(direction,dtype=float)
    if (lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or
            np.any(upper<=lower) or d.shape!=(3,) or not np.isfinite(d).all() or not np.any(d)):
        raise ValueError("finite ordered bounds and a nonzero direction are required")
    d = d/np.max(np.abs(d))
    d = d/np.sqrt(np.dot(d,d))
    from .slices import _transverse_basis
    u,v = _transverse_basis(d)
    corners = lower+(upper-lower)*np.indices((2,2,2)).reshape(3,-1).T
    pu,pv,pd = corners@u,corners@v,corners@d
    origin = u*pu.min()+v*pv.min()+d*(pd.min()-.1*np.linalg.norm(upper-lower))
    return Plane(origin,u*np.ptp(pu),v*np.ptp(pv),shape)



@dataclass
class _RayState:
    origins: np.ndarray
    direction: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    progress: np.ndarray
    values: np.ndarray
    status: np.ndarray
    requested: np.ndarray
    samples: np.ndarray
    misses: np.ndarray


def _new_rays(mesh,origins,direction,near,far):
    from ._kernels.native import initialize_rays
    n=len(origins)
    starts,ends = np.empty(n),np.empty(n)
    status=np.zeros(n,np.int64)
    initialize_rays(mesh.lower,mesh.upper,origins,direction,near,far,starts,ends,status)
    values=np.zeros(n)
    values[status>=LOSStatus.MISSING_COVERAGE]=np.nan
    return _RayState(origins,direction,starts,ends,starts.copy(),values,status,
                     np.full(n,-1,np.int64),np.zeros(n,np.int64),np.zeros(n,np.int64))


def _advance_rays(fields,state,component,step_fraction,quadrature,max_samples,workers,executor,native,dispatch):
    from ._kernels.native import advance_rays
    m=fields.mesh
    values=fields.values
    def advance(first,last):
        s=slice(first,last)
        advance_rays(m.lower,m.upper,m.roots,m.children,m.node_leaves,m.node_lower,m.node_upper,
            m.bounds,m.spacing,fields.slot_of_leaf,values,fields.storage_halo,component,
            state.origins[s],state.direction,state.progress[s],state.ends[s],step_fraction,
            quadrature,max_samples,state.values[s],state.status[s],state.requested[s],
            state.samples[s],state.misses[s],workers if native else 1,dispatch)
    run_ranges(len(state.origins),1 if native else workers*(8 if dispatch and workers>1 else 1),
               advance,None if native else executor)


def _ray_arrays(state):
    return state.values,state.starts,state.ends,state.status,state.samples,state.misses


def _tile_ready(fields,origins,direction,near,far,component,step_fraction,quadrature,max_samples,
                workers,executor,native,dispatch,touch):
    state=_new_rays(fields.mesh,origins,direction,near,far)
    _advance_rays(fields,state,component,step_fraction,quadrature,max_samples,workers,executor,native,dispatch)
    pending=state.status==LOSStatus.RUNNING
    if np.any(pending & (state.requested<0)):
        raise RuntimeError("LOS made no progress without a preparation request")
    state.status[pending]=LOSStatus.MISSING_COVERAGE
    state.values[pending]=np.nan
    return _ray_arrays(state)


def _integrate_views(planes,directions,definitions,footprint,runner,*,component=0,near=0.,far=np.inf,
                     step_fraction=.5,max_samples=1000000,workers=1,tile_shape=(16,16),
                     memory_limit=None,quadrature="gauss2",backend="threadpool",schedule="static",
                     view_order="tile",capacity=None):
    planes=tuple(planes)
    if not planes or any(not isinstance(p,Plane) for p in planes):
        raise ValueError("LOS requires one or more Planes")
    plane=planes[0]
    if any(p.shape!=plane.shape for p in planes) or view_order not in ("tile","view"):
        raise ValueError("grouped views require equal image shapes and tile/view order")
    if quadrature not in ("midpoint","gauss2"):
        raise ValueError("quadrature must be midpoint or gauss2")
    native,dispatch=native_dispatch(backend,schedule)
    workers_count(workers)
    if isinstance(component, str):
        component = _field_index(definitions, component)
    elif isinstance(component, np.integer) and not isinstance(component, np.bool_):
        component = int(component)
    if (type(component) is not int or not 0<=component<len(definitions) or
            not np.isfinite(step_fraction) or step_fraction<=0 or type(max_samples) is not int or
            not 1<=max_samples<=np.iinfo(np.int64).max):
        raise ValueError("invalid component or LOS quadrature limits")
    directions=np.array(directions,dtype=float,order="C",copy=True)
    if directions.shape!=(len(planes),3) or not np.isfinite(directions).all() or np.any(~directions.any(axis=1)):
        raise ValueError("finite nonzero directions must match the views")
    for direction in directions:
        direction/=np.max(np.abs(direction))
        direction/=np.sqrt(np.dot(direction,direction))
    tile_shape=tuple(tile_shape)
    if len(tile_shape)!=2 or any(type(n) is not int or n<1 for n in tile_shape):
        raise ValueError("tile_shape requires two positive integer extents")
    tx,ty=tile_shape
    if capacity is not None:
        ty=min(ty,capacity)
        tx=min(tx,max(1,capacity//ty))
    nx,ny=plane.shape
    required=footprint+len(planes)*(nx*ny*96+256)+min(tx,nx)*min(ty,ny)*384
    admit(required,memory_limit,"LOS images")
    near=np.array(np.broadcast_to(near,plane.shape),dtype=float,order="C",copy=True)
    far=np.array(np.broadcast_to(far,plane.shape),dtype=float,order="C",copy=True)
    if not np.isfinite(near).all() or np.any(near<0) or np.isnan(far).any() or np.any(far<near):
        raise ValueError("near/far must be ordered nonnegative arclength bounds")
    arrays=[[np.empty(plane.shape,dtype=float if i<3 else np.int64) for i in range(6)] for _ in planes]
    def tasks():
        if view_order=="view":
            for view in range(len(planes)):
                for x in range(0,nx,tx):
                    for y in range(0,ny,ty):
                        yield view,x,y
        else:
            for x in range(0,nx,tx):
                for y in range(0,ny,ty):
                    for view in range(len(planes)):
                        yield view,x,y
    context=nullcontext(None) if native else worker_context(workers)
    with context as executor:
        for view,x,y in tasks():
            plane,direction=planes[view],directions[view]
            stopx,stopy=min(x+tx,nx),min(y+ty,ny)
            u,v=(np.arange(x,stopx)+.5)/nx,(np.arange(y,stopy)+.5)/ny
            origins=plane.origin+u[:,None,None]*plane.u+v[None,:,None]*plane.v
            part=np.s_[x:stopx,y:stopy]
            rows=runner(np.ascontiguousarray(origins.reshape(-1,3)),direction,
                np.ascontiguousarray(near[part].ravel()),np.ascontiguousarray(far[part].ravel()),
                component,step_fraction,int(quadrature=="gauss2"),max_samples,workers,executor,native,dispatch,
                len(planes)==1 or view_order=="view")
            for output,row in zip(arrays[view],rows):
                output[part]=row.reshape(stopx-x,stopy-y)
    directions.flags.writeable=False
    return [LOSResult(p,d,*values,definitions[component].units+" * coordinate-length",quadrature)
            for p,d,values in zip(planes,directions,arrays)]


def integrate_los_views(fields,planes,directions,**kwargs):
    """Integrate retained scalar fields over requested equal-shaped views.

    Parameters
    ----------
    fields : Fields
        Selected continuous scalar component with at least one valid halo.
    planes : sequence of Plane
        One or more equal-shaped planes.
    directions : array-like
        One finite nonzero (3,) vector per plane.
    **kwargs : object
        Forwarded keyword controls listed below.

    Other Parameters
    ----------------
    component : str or int
        Scalar field name or local column index (default 0).
    near, far : float or array-like
        Nonnegative plane-shaped clipping distances (defaults 0 and infinity).
    quadrature : str
        gauss2 splits at interpolation knots/block boundaries; midpoint uses
        step_fraction (default gauss2).
    step_fraction : float
        Positive midpoint step fraction (default 0.5).
    max_samples : int
        Per-ray limit (default 1000000).
    workers : int
        Worker count (default 1).
    tile_shape : tuple of int
        Image tile dimensions (default (16, 16)).
    backend, schedule : str
        threadpool/openmp and static/dynamic (defaults threadpool and static).
    view_order : str
        tile or view traversal (default tile).
    memory_limit : int, optional
        Per-call accounted-array budget, not a process RSS cap.
    capacity : int, optional
        Optional cap on tile positions; bounded consumers supply their pool capacity.

    Returns
    -------
    list of LOSResult
        One raw result per equal-shaped plane, in field units times coordinate length;
        LOSResult defines validity and completion.
    """
    kwargs["component"], = require_continuous(fields, (kwargs.get("component", 0),), operation="LOS")
    return _integrate_views(planes,directions,fields.fields,fields.nbytes+fields.mesh.nbytes,
                           lambda *args:_tile_ready(fields,*args),**kwargs)


def integrate_los(fields,plane,direction,**kwargs):
    """Integrate a supplied scalar along clipped, pixel-owned rays.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    plane : Plane
        Pixel-center sampling plane.
    direction : array-like
        Finite nonzero viewing direction (3,).
    **kwargs : object
        Forwarded controls listed in [integrate_los_views][simesh.integrate_los_views].

    Returns
    -------
    LOSResult
        One raw plane result; use valid/status as well as values.
    """
    return integrate_los_views(fields,[plane],[direction],**kwargs)[0]
