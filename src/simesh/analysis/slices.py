"""Reusable plane geometry and pixel-center samples from retained native fields."""

from dataclasses import dataclass
import numpy as np

from .mesh import frozen_array
from .sampling import sample
from .fields import PreparedPool


def _sample_pool(pool,points):
    """Prepare bounded owner batches, preserving arbitrary point order."""
    owners=pool.source.mesh.locate(points)
    values=np.empty((len(points),len(pool.field_ids)))
    valid=np.empty(len(points),dtype=bool)
    pending=[(0,len(points))]
    while pending:
        first,last=pending.pop()
        ids=np.unique(owners[first:last])
        ids=ids[ids>=0]
        if len(ids)>pool.capacity:
            middle=(first+last)//2
            pending.extend(((middle,last),(first,middle)))
            continue
        with pool.borrow(ids) as ready:
            result,sampled_owners,okay=sample(ready,points[first:last])
            values[first:last]=result
            valid[first:last]=okay
            del result,sampled_owners,okay
    return values,owners,valid


@dataclass(frozen=True)
class Plane:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    shape: tuple

    def __post_init__(self):
        for name in ("origin","u","v"):
            value = frozen_array(getattr(self,name),float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("plane vectors must be finite triplets")
            object.__setattr__(self,name,value)
        if np.linalg.norm(np.cross(self.u,self.v)) == 0:
            raise ValueError("plane spans must be independent")
        shape = tuple(self.shape)
        if len(shape)!=2 or any(type(n) is not int or n<1 for n in shape):
            raise ValueError("plane shape must have two positive integer extents")
        object.__setattr__(self,"shape",shape)


@dataclass(frozen=True)
class SliceResult:
    plane: Plane
    values: np.ndarray
    valid: np.ndarray
    owners: np.ndarray


def sample_plane(prepared, plane, *, tile_rows=64, budget_bytes=2*1024**3):
    """Sample pixel centers; retain source native data for future moved slices."""
    if not isinstance(plane,Plane) or type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("a Plane and positive tile_rows are required")
    nx,ny = plane.shape
    pooled=isinstance(prepared,PreparedPool)
    if pooled and prepared._closed:
        raise RuntimeError("slice pool is closed")
    fields = len(prepared.field_ids) if pooled else len(prepared.fields)
    footprint=prepared.controlled_bytes if pooled else prepared.nbytes+prepared.mesh.nbytes
    required = (footprint + nx*ny*(fields*8+9) +
                min(nx,tile_rows)*ny*(fields*8+128))
    if required > budget_bytes:
        raise MemoryError(f"slice needs {required} controlled bytes, budget {budget_bytes}")
    output = np.empty((nx,ny,fields),dtype=float)
    valid = np.empty((nx,ny),dtype=bool)
    owners = np.empty((nx,ny),dtype=np.int64)
    v = (np.arange(ny)+.5)/ny
    for start in range(0,nx,tile_rows):
        stop = min(start+tile_rows,nx)
        u = (np.arange(start,stop)+.5)/nx
        points = (plane.origin + u[:,None,None]*plane.u + v[None,:,None]*plane.v)
        sampler=_sample_pool if pooled else sample
        values, ids, okay = sampler(prepared,np.ascontiguousarray(points.reshape(-1,3)))
        output[start:stop] = values.reshape(stop-start,ny,fields)
        valid[start:stop] = okay.reshape(stop-start,ny)
        owners[start:stop] = ids.reshape(stop-start,ny)
    return SliceResult(plane,output,valid,owners)


def iter_uniform(fields,resolution,*,bounds=None,tile_rows=64,budget_bytes=2*1024**3):
    """Yield (z index, owned SliceResult) at uniform-grid cell centers.

    Each slab has values (nx,ny,field). Consume/write slabs incrementally; no
    complete volume or coordinate cube is allocated. Retaining slabs is an
    explicit caller output choice, separately accounted like iter_traces.
    """
    resolution=tuple(resolution)
    if len(resolution)!=3 or any(not isinstance(n,(int,np.integer)) or n<1 for n in resolution):
        raise ValueError("resolution needs three positive integer extents")
    nx,ny,nz=map(int,resolution)
    if isinstance(fields,PreparedPool) and fields._closed:
        raise RuntimeError("uniform source pool is closed")
    mesh=fields.source.mesh if isinstance(fields,PreparedPool) else fields.mesh
    lower,upper=(mesh.lower,mesh.upper) if bounds is None else bounds
    lower,upper=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or np.any(upper<=lower):
        raise ValueError("uniform bounds must be finite ordered triplets")
    width=upper-lower
    count=len(fields.field_ids) if isinstance(fields,PreparedPool) else len(fields.fields)
    # A normal for-loop still holds its previous yielded slab while advancing.
    previous_slab_bytes=nx*ny*(8*count+9)
    for iz in range(nz):
        origin=lower.copy()
        origin[2]=lower[2]+(iz+.5)*(width[2]/nz)
        plane=Plane(origin,[width[0],0.,0.],[0.,width[1],0.],(nx,ny))
        yield iz,sample_plane(fields,plane,tile_rows=tile_rows,budget_bytes=budget_bytes-previous_slab_bytes)
