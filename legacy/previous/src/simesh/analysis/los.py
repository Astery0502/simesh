"""Deterministic pixel-owned scalar LOS integration on prepared native fields.

Physical response/EOS is a separate required input definition. This module
integrates the supplied scalar; it does not infer a thermal or instrument model.
"""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from .fields import PreparedPool
from .slices import Plane


class LOSStatus(IntEnum):
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
        return self.exit-self.entry

    @property
    def valid(self):
        return (self.status==LOSStatus.COMPLETE)|(self.status==LOSStatus.EMPTY)

    @property
    def complete(self):
        return bool(self.valid.all())


def _tile(fields,origins,direction,near,far,component,step_fraction,quadrature,max_samples,workers,executor,
          native_workers=1,dispatch=0,touch_coverage=True):
    from simesh.utils.lib.analysis.native import initialize_rays,advance_rays
    n = len(origins)
    mesh = fields.source.mesh if isinstance(fields,PreparedPool) else fields.mesh
    starts,ends = np.empty(n),np.empty(n)
    status = np.zeros(n,dtype=np.int64)
    initialize_rays(mesh.lower,mesh.upper,origins,direction,near,far,starts,ends,status)
    progress = starts.copy()
    values = np.zeros(n)
    requested = np.full(n,-1,dtype=np.int64)
    samples,misses = np.zeros(n,dtype=np.int64),np.zeros(n,dtype=np.int64)
    values[status>=3] = np.nan
    partitions = [p for p in np.array_split(np.arange(n),workers) if len(p)]
    def advance(product,part):
        span = slice(int(part[0]),int(part[-1])+1)
        advance_rays(mesh.lower,mesh.upper,mesh.roots,mesh.children,mesh.node_leaves,
            mesh.node_lower,mesh.node_upper,mesh.bounds,mesh.spacing,product.slot_of_leaf,
            product.values,product.halo,component,origins[span],direction,progress[span],ends[span],
            step_fraction,quadrature,max_samples,values[span],status[span],requested[span],samples[span],misses[span],
            native_workers,dispatch)
    while np.any(status==LOSStatus.RUNNING):
        context = fields.borrow(fields.resident_leaf_ids,touch=touch_coverage) if isinstance(fields,PreparedPool) else nullcontext(fields)
        with context as product:
            if executor is None:
                advance(product,partitions[0])
            else:
                futures = [executor.submit(advance,product,p) for p in partitions]
                for future in futures:
                    future.result()
        missing = np.unique(requested[(status==0)&(requested>=0)])
        if len(missing):
            if isinstance(fields,PreparedPool):
                with fields.borrow(missing):
                    pass
            else:
                selected = (status==0)&(requested>=0)
                status[selected] = LOSStatus.MISSING_COVERAGE
                values[selected] = np.nan
        elif np.any(status==0):
            raise RuntimeError("LOS made no progress without a preparation request")
    return values,starts,ends,status,samples,misses


def integrate_los(fields,plane,direction,*,component=0,near=0.,far=np.inf,
                  step_fraction=.5,max_samples=1000000,workers=1,tile_shape=(16,16),
                  budget_bytes=2*1024**3,quadrature="gauss2",backend='auto',schedule='static'):
    """Integrate a supplied scalar over each clipped ray in physical arclength.

    Pixel centers follow Plane geometry; near/far can be scalars or plane-shaped
    arrays. Gauss2 splits at cell-center planes and integrates the trilinear
    scalar's piecewise cubic restriction to each ray. step_fraction is used
    only for midpoint quadrature.
    Invalid pixels are explicit; a sample limit never certifies partial coverage.
    """
    return integrate_los_views(fields,[plane],[direction],component=component,near=near,far=far,
        step_fraction=step_fraction,max_samples=max_samples,workers=workers,tile_shape=tile_shape,
        budget_bytes=budget_bytes,quadrature=quadrature,backend=backend,schedule=schedule)[0]


def integrate_los_views(fields,planes,directions,*,component=0,near=0.,far=np.inf,
                        step_fraction=.5,max_samples=1000000,workers=1,tile_shape=(16,16),
                        budget_bytes=2*1024**3,quadrature='gauss2',backend='auto',
                        schedule='static',view_order='tile'):
    """Integrate equal-shaped views with one pool and one execution team.

    Tile order interleaves matching pixel tiles across views, retaining nearby
    numerical neighborhoods. View order is the sequential reference. Results
    keep input view order and independent pixel sums; near/far are common scalar
    or image-shaped bounds. Every requested image is included in admission.
    """
    planes=tuple(planes)
    if not planes or any(not isinstance(p,Plane) for p in planes):
        raise ValueError('LOS requires one or more Planes of ray origins')
    plane=planes[0]
    if any(p.shape!=plane.shape for p in planes) or view_order not in ('tile','view'):
        raise ValueError('LOS view grouping requires equal image shapes and tile/view order')
    if quadrature not in ("midpoint","gauss2"):
        raise ValueError("quadrature must be midpoint or gauss2")
    from .execution import native_dispatch
    native,dispatch = native_dispatch(backend,schedule)
    pool = isinstance(fields,PreparedPool)
    if pool and fields._closed:
        raise RuntimeError("LOS pool is closed")
    definitions = tuple(fields.source.fields[i] for i in fields.field_ids) if pool else fields.fields
    if (type(component) is not int or not 0<=component<len(definitions) or fields.halo<1 or
            not np.isfinite(step_fraction) or step_fraction<=0 or type(max_samples) is not int or
            not 1<=max_samples<=np.iinfo(np.int64).max or type(workers) is not int or not 1<=workers<=4):
        raise ValueError("invalid LOS component, support, quadrature or worker parameters")
    directions = np.array(directions,dtype=float,order="C",copy=True)
    if directions.shape!=(len(planes),3) or not np.isfinite(directions).all() or np.any(~directions.any(axis=1)):
        raise ValueError("LOS directions must be finite nonzero triplets matching the views")
    for direction in directions:
        direction /= np.max(np.abs(direction))
        direction /= np.sqrt(np.dot(direction,direction))
    tile_shape = tuple(tile_shape)
    if len(tile_shape)!=2 or any(type(n) is not int or n<1 for n in tile_shape):
        raise ValueError("tile_shape requires two positive integer extents")
    tx,ty = tile_shape
    if pool:
        ty = min(ty,fields.capacity)
        tx = min(tx,max(1,fields.capacity//ty))
    nx,ny = plane.shape
    footprint = fields.controlled_bytes if pool else fields.nbytes+fields.mesh.nbytes
    required = footprint+len(planes)*(nx*ny*96+256)+min(tx,nx)*min(ty,ny)*384
    if required>budget_bytes:
        raise MemoryError(f"LOS needs {required} controlled bytes, budget {budget_bytes}")
    near = np.array(np.broadcast_to(near,plane.shape),dtype=float,order="C",copy=True)
    far = np.array(np.broadcast_to(far,plane.shape),dtype=float,order="C",copy=True)
    if not np.isfinite(near).all() or np.any(near<0) or np.isnan(far).any() or np.any(far<near):
        raise ValueError("near/far must be ordered nonnegative arclength bounds")
    arrays = [[np.empty(plane.shape,dtype=float if i<3 else np.int64) for i in range(6)] for p in planes]
    def tasks():
        if view_order=='view':
            for view in range(len(planes)):
                for x in range(0,nx,tx):
                    for y in range(0,ny,ty):
                        yield view,x,y
        else:
            for x in range(0,nx,tx):
                for y in range(0,ny,ty):
                    for view in range(len(planes)):
                        yield view,x,y
    manager = ThreadPoolExecutor(max_workers=workers) if workers>1 and not native else nullcontext(None)
    with manager as executor:
        for view,x,y in tasks():
            plane,direction=planes[view],directions[view]
            stopx,stopy = min(x+tx,nx),min(y+ty,ny)
            u = (np.arange(x,stopx)+.5)/nx
            v = (np.arange(y,stopy)+.5)/ny
            origins = plane.origin+u[:,None,None]*plane.u+v[None,:,None]*plane.v
            part = np.s_[x:stopx,y:stopy]
            rows = _tile(fields,np.ascontiguousarray(origins.reshape(-1,3)),direction,
                np.ascontiguousarray(near[part].ravel()),np.ascontiguousarray(far[part].ravel()),
                component,step_fraction,int(quadrature=="gauss2"),max_samples,
                1 if native else workers*(8 if dispatch and workers>1 else 1),
                executor,workers if native else 1,dispatch,
                touch_coverage=len(planes)==1 or view_order=='view')
            for output,row in zip(arrays[view],rows):
                output[part] = row.reshape(stopx-x,stopy-y)
    directions.flags.writeable = False
    return [LOSResult(p,d,*values,definitions[component].units+" * coordinate-length",quadrature)
            for p,d,values in zip(planes,directions,arrays)]


def orthographic_plane(lower,upper,direction,shape):
    """Enclose the projection of the entire physical box in a perpendicular Plane."""
    lower,upper = np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    d = np.asarray(direction,dtype=float)
    if (lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or
            np.any(upper<=lower) or d.shape!=(3,) or not np.isfinite(d).all() or not np.any(d)):
        raise ValueError("finite ordered bounds and a nonzero direction are required")
    d = d/np.max(np.abs(d))
    d = d/np.sqrt(np.dot(d,d))
    basis = np.eye(3)[np.argmin(np.abs(d))]
    u = basis-np.dot(basis,d)*d
    u /= np.linalg.norm(u)
    v = np.cross(d,u)
    corners = lower+(upper-lower)*np.indices((2,2,2)).reshape(3,-1).T
    pu,pv,pd = corners@u,corners@v,corners@d
    origin = u*pu.min()+v*pv.min()+d*(pd.min()-.1*np.linalg.norm(upper-lower))
    return Plane(origin,u*np.ptp(pu),v*np.ptp(pv),shape)
