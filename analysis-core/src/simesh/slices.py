"""Pixel-center plane samples of retained native fields."""

from dataclasses import dataclass
import numpy as np

from ._validation import frozen_array, admit
from .fields import require_fields
from ._execution import worker_context
from .operators.sampling import _sample


@dataclass(frozen=True)
class Plane:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    shape: tuple

    def __post_init__(self):
        for name in ("origin", "u", "v"):
            value = frozen_array(getattr(self, name), float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("plane vectors must be finite triplets")
            object.__setattr__(self, name, value)
        if np.linalg.norm(np.cross(self.u, self.v)) == 0:
            raise ValueError("plane spans must be independent")
        shape = tuple(self.shape)
        if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("plane shape requires two positive integer extents")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class SliceResult:
    plane: Plane
    values: np.ndarray
    valid: np.ndarray
    owners: np.ndarray


def _plane_result(plane,count,footprint,sampler,*,tile_rows,workers,executor,memory_limit):
    if not isinstance(plane,Plane) or type(tile_rows) is not int or tile_rows<1:
        raise ValueError("Plane and positive tile_rows required")
    nx,ny=plane.shape
    required=footprint+nx*ny*(count*8+9)+min(nx,tile_rows)*ny*(count*8+128)
    admit(required,memory_limit,"slice")
    output=np.empty((nx,ny,count))
    valid=np.empty((nx,ny),dtype=bool)
    owners=np.empty((nx,ny),dtype=np.int64)
    v=(np.arange(ny)+.5)/ny
    for first in range(0,nx,tile_rows):
        last=min(first+tile_rows,nx)
        u=(np.arange(first,last)+.5)/nx
        points=plane.origin+u[:,None,None]*plane.u+v[None,:,None]*plane.v
        values,ids,okay=sampler(np.ascontiguousarray(points.reshape(-1,3)),workers,executor)
        output[first:last]=values.reshape(last-first,ny,count)
        valid[first:last]=okay.reshape(last-first,ny)
        owners[first:last]=ids.reshape(last-first,ny)
    return SliceResult(plane,output,valid,owners)


def sample_plane(fields,plane,*,tile_rows=64,workers=1,memory_limit=None):
    """Sample a plane directly; no input preparation or source access occurs."""
    require_fields(fields,halo=1)
    with worker_context(workers) as executor:
        return _plane_result(plane,len(fields.fields),fields.mesh.nbytes+fields.nbytes,
            lambda points,w,e:_sample(fields,points,w,e),tile_rows=tile_rows,
            workers=workers,executor=executor,memory_limit=memory_limit)


def _uniform_geometry(mesh,resolution,bounds):
    resolution=tuple(resolution)
    if len(resolution)!=3 or any(not isinstance(n,(int,np.integer)) or n<1 for n in resolution):
        raise ValueError("resolution needs three positive integer extents")
    lower,upper=(mesh.lower,mesh.upper) if bounds is None else bounds
    lower,upper=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or np.any(upper<=lower):
        raise ValueError("uniform bounds must be finite ordered triplets")
    return tuple(map(int,resolution)),lower,upper


def iter_uniform(fields,resolution,*,bounds=None,tile_rows=64,workers=1,memory_limit=None):
    """Yield owned (z index, SliceResult) slabs without materializing a volume."""
    from ._validation import remaining
    require_fields(fields,halo=1)
    (nx,ny,nz),lower,upper=_uniform_geometry(fields.mesh,resolution,bounds)
    width=upper-lower
    previous=nx*ny*(8*len(fields.fields)+9)
    limit=remaining(memory_limit,previous)
    with worker_context(workers) as executor:
        for iz in range(nz):
            fields._check()
            origin=lower.copy()
            origin[2]=lower[2]+(iz+.5)*(width[2]/nz)
            plane=Plane(origin,[width[0],0.,0.],[0.,width[1],0.],(nx,ny))
            yield iz,_plane_result(plane,len(fields.fields),fields.mesh.nbytes+fields.nbytes,
                lambda points,w,e:_sample(fields,points,w,e),tile_rows=tile_rows,
                workers=workers,executor=executor,memory_limit=limit)
