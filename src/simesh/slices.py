"""Pixel-center plane samples of retained native fields."""

from dataclasses import dataclass
import numpy as np

from ._validation import frozen_array, admit
from .fields import require_fields, require_continuous, _input_arrays
from ._execution import worker_context
from .operators.sampling import _sample, _sample_output


def _transverse_basis(direction):
    basis = np.eye(3)[np.argmin(np.abs(direction))]
    u = basis-np.dot(basis,direction)*direction
    u /= np.linalg.norm(u)
    return u, np.cross(direction,u)


@dataclass(frozen=True)
class Plane:
    """Own a pixel-center sampling plane.

    Parameters
    ----------
    origin : array-like
        Physical corner (3,); sampling uses pixel centers within the span vectors.
    u, v : array-like
        Independent full-image span vectors (3,); not per-pixel spacing.
    shape : tuple of int
        Positive pixel counts along u and v.
    """
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


def _uniform_geometry(mesh,resolution,bounds):
    resolution=tuple(resolution)
    if len(resolution)!=3 or any(not isinstance(n,(int,np.integer)) or n<1 for n in resolution):
        raise ValueError("resolution needs three positive integer extents")
    lower,upper=(mesh.lower,mesh.upper) if bounds is None else bounds
    lower,upper=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or np.any(upper<=lower):
        raise ValueError("uniform bounds must be finite ordered triplets")
    return tuple(map(int,resolution)),lower,upper


def iter_uniform(fields,resolution,*,components=None,bounds=None,tile_rows=64,workers=1,memory_limit=None):
    """Yield owned (z index, SliceResult) slabs without materializing a volume.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    resolution : sequence of int
        Positive uniform-grid cell counts (nx, ny, nz).
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    bounds : array-like, optional
        Lower and upper physical bounds; defaults to the original domain.
    tile_rows : int
        Maximum rows sampled per tile; collected output still occupies memory.
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
