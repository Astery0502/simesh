"""Owned sampling geometry and compact, identified two-branch trajectories."""

from dataclasses import dataclass
import math
from typing import ClassVar
import numpy as np

from ._validation import frozen_array, identifiers, array_bytes, admit
from .slices import Plane


def _unit_vectors(value, count, label):
    vectors = np.asarray(value,dtype=float)
    if vectors.shape not in ((3,),(count,3)) or not np.isfinite(vectors).all():
        raise ValueError(f"{label} must be a finite vector or one vector per point")
    scale = np.max(np.abs(vectors),axis=-1,keepdims=True)
    if np.any(scale == 0):
        raise ValueError(f"{label} must be nonzero")
    vectors = vectors/scale
    return frozen_array(vectors/np.linalg.norm(vectors,axis=-1,keepdims=True),float)


@dataclass(frozen=True, eq=False)
class PointSet:
    positions: np.ndarray
    ids: np.ndarray | None = None
    shape: tuple | None = None
    normals: np.ndarray | None = None
    plane: Plane | None = None

    def __post_init__(self):
        positions = frozen_array(self.positions, float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("positions must be finite (n,3) coordinates")
        count = len(positions)
        shape = (count,) if self.shape is None else tuple(self.shape)
        if (not shape or any(type(n) is not int or n < 0 for n in shape) or math.prod(shape) != count):
            raise ValueError("shape must describe exactly the supplied points")
        if self.plane is not None and not isinstance(self.plane,Plane):
            raise TypeError("plane must be a Plane describing the parent surface")
        normals = self.normals
        if normals is not None:
            normals = _unit_vectors(normals,count,"normals")
        object.__setattr__(self,"positions",positions)
        object.__setattr__(self,"ids",frozen_array(identifiers(self.ids,count),np.int64))
        object.__setattr__(self,"shape",shape)
        object.__setattr__(self,"normals",normals)

    def __len__(self):
        return len(self.positions)

    @property
    def nbytes(self):
        plane_arrays = () if self.plane is None else (self.plane.origin,self.plane.u,self.plane.v)
        return array_bytes((self.positions,self.ids,self.normals,*plane_arrays))

    @classmethod
    def from_plane(cls, plane, *, ids=None, memory_limit=None):
        if not isinstance(plane,Plane):
            raise TypeError("a Plane is required")
        admit(math.prod(plane.shape)*128+512,memory_limit,"plane points")
        u,v = ((np.arange(n)+.5)/n for n in plane.shape)
        positions = plane.origin+u[:,None,None]*plane.u+v[None,:,None]*plane.v
        return cls(positions.reshape(-1,3),ids,plane.shape,np.cross(plane.u,plane.v),plane)

    @classmethod
    def iter_plane(cls, plane, *, batch_size=4096, memory_limit=None):
        """Generate pixel centers by batch; IDs are flat indices on the parent plane."""
        if not isinstance(plane,Plane) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("a Plane and positive batch_size are required")
        nx,ny = plane.shape
        count = nx*ny
        if count > np.iinfo(np.int64).max:
            raise ValueError("plane indices exceed int64")
        admit(min(batch_size,count)*128+512,memory_limit,"plane point batch")
        normal = np.cross(plane.u,plane.v)
        for start in range(0,count,batch_size):
            ids = np.arange(start,min(start+batch_size,count),dtype=np.int64)
            u,v = (ids//ny+.5)/nx,(ids%ny+.5)/ny
            positions = plane.origin+u[:,None]*plane.u+v[:,None]*plane.v
            yield cls(positions,ids,normals=normal,plane=plane)

    @classmethod
    def boundary(cls, mesh, face, shape, *, ids=None, memory_limit=None):
        faces = {"xmin":(0,-1),"xmax":(0,1),"ymin":(1,-1),"ymax":(1,1),"zmin":(2,-1),"zmax":(2,1)}
        if face not in faces:
            raise ValueError("face must be xmin/xmax/ymin/ymax/zmin/zmax")
        axis,side = faces[face]
        origin = mesh.lower.copy()
        origin[axis] = mesh.lower[axis] if side < 0 else mesh.upper[axis]
        tangent = [a for a in range(3) if a != axis]
        u,v = np.eye(3)[tangent]*(mesh.upper-mesh.lower)[tangent,None]
        result = cls.from_plane(Plane(origin,u,v,shape),ids=ids,memory_limit=memory_limit)
        normal = frozen_array(np.eye(3)[axis]*side,float)
        object.__setattr__(result,"normals",normal)
        return result

    def rows(self, mask):
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape not in (self.shape,(len(self),)):
            raise ValueError("selection must be a boolean mask matching the point layout")
        return np.flatnonzero(mask.ravel())

    def select(self, mask):
        rows = self.rows(mask)
        normals = self.normals
        if normals is not None and normals.ndim == 2:
            normals = normals[rows]
        return PointSet(self.positions[rows],self.ids[rows],normals=normals,plane=self.plane)

    def reshape(self, values):
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(self):
            raise ValueError("values must have one row per point")
        return values.reshape(*self.shape,*values.shape[1:])


@dataclass(frozen=True, eq=False)
class RaySet:
    origins: PointSet
    directions: np.ndarray
    near: object = 0.
    far: object = np.inf

    def __post_init__(self):
        if not isinstance(self.origins,PointSet):
            raise TypeError("origins must be a PointSet")
        directions = _unit_vectors(self.directions,len(self.origins),"ray directions")
        near,far = (frozen_array(np.broadcast_to(value,self.origins.shape).ravel(),float)
                    if np.asarray(value).shape != (len(self.origins),) else frozen_array(value,float)
                    for value in (self.near,self.far))
        if not np.isfinite(near).all() or np.any(near < 0) or np.isnan(far).any() or np.any(far < near):
            raise ValueError("near/far must be ordered nonnegative arclength limits")
        object.__setattr__(self,"directions",directions)
        object.__setattr__(self,"near",near)
        object.__setattr__(self,"far",far)

    @classmethod
    def from_plane(cls, plane, direction, *, near=0., far=np.inf, ids=None, memory_limit=None):
        if not isinstance(plane,Plane):
            raise TypeError("a Plane is required")
        admit(math.prod(plane.shape)*160+1024,memory_limit,"plane rays")
        return cls(PointSet.from_plane(plane,ids=ids),direction,near,far)

    @property
    def nbytes(self):
        return self.origins.nbytes+array_bytes((self.directions,self.near,self.far))

    def select(self, mask):
        rows = self.origins.rows(mask)
        direction = self.directions if self.directions.ndim == 1 else self.directions[rows]
        result = RaySet(self.origins.select(mask),direction,self.near[rows],self.far[rows])
        # Renormalization must not perturb a selected ray at a grazing face.
        object.__setattr__(result,"directions",frozen_array(direction,float))
        return result


@dataclass(frozen=True, eq=False)
class LineSet:
    """Packed branches, each stored from seed to endpoint, with stable seed IDs.

    offsets has 2*n+1 entries. Branches 2*i and 2*i+1 are against and along B.
    Arrays supplied by internal factories are independently owned.
    """
    NOT_REQUESTED: ClassVar[int] = -1
    TANGENT_SEED: ClassVar[int] = -2
    seeds: PointSet
    positions: np.ndarray
    offsets: np.ndarray
    termination: np.ndarray
    source_identity: object

    def __post_init__(self):
        if not isinstance(self.seeds,PointSet):
            raise TypeError("line seeds must be a PointSet")
        count = len(self.seeds)
        if (not all(isinstance(a,np.ndarray) for a in (self.positions,self.offsets,self.termination)) or
                self.positions.ndim != 2 or self.positions.shape[1] != 3 or
                self.positions.dtype != np.float64 or not self.positions.flags.c_contiguous or
                not np.isfinite(self.positions).all() or self.offsets.shape != (2*count+1,) or
                self.offsets.dtype != np.int64 or self.offsets[0] != 0 or
                self.offsets[-1] != len(self.positions) or np.any(np.diff(self.offsets) < 0) or
                self.termination.shape != (count,2) or self.termination.dtype != np.int64):
            raise ValueError("invalid packed trajectory storage")
        for array in (self.positions,self.offsets,self.termination):
            array.flags.writeable = False

    @property
    def nbytes(self):
        return self.seeds.nbytes+array_bytes((self.positions,self.offsets,self.termination))

    def branch(self, seed_id, direction):
        matches = np.flatnonzero(self.seeds.ids == seed_id)
        if len(matches) != 1 or direction not in (-1,1):
            raise ValueError("select an existing seed ID and direction -1 or 1")
        index = 2*int(matches[0])+int(direction == 1)
        return self.positions[self.offsets[index]:self.offsets[index+1]]

    def line(self, seed_id):
        negative,positive = self.branch(seed_id,-1),self.branch(seed_id,1)
        return np.concatenate((negative[::-1],positive[1:] if len(negative) else positive))
