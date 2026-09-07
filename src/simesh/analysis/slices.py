"""Reusable plane geometry and pixel-center samples from retained native fields."""

from dataclasses import dataclass
import numpy as np

from .mesh import frozen_array
from .sampling import sample


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
    fields = len(prepared.fields)
    required = (prepared.nbytes+prepared.mesh.nbytes + nx*ny*(fields*8+9) +
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
        values, ids, okay = sample(prepared,np.ascontiguousarray(points.reshape(-1,3)))
        output[start:stop] = values.reshape(stop-start,ny,fields)
        valid[start:stop] = okay.reshape(stop-start,ny)
        owners[start:stop] = ids.reshape(stop-start,ny)
    return SliceResult(plane,output,valid,owners)
