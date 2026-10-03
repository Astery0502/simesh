"""Physical sampling geometry independent of mesh topology and field storage.

Positions, spans, normals and ray directions use a shared Cartesian XYZ frame.
A spherical or cylindrical coordinate tuple must be mapped into that frame before
constructing these objects. LengthUnits supplies an isotropic physical scale;
it does not convert angular coordinates or change a vector basis. Native mesh
sections and cell quadrature are defined separately in simesh.geometry.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import ClassVar
import numpy as np

from ._validation import frozen_array, identifiers, array_bytes, admit


def _box(bounds, dimension=3):
    box = np.asarray(bounds, dtype=float)
    if box.shape != (2, dimension) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
        raise ValueError(f"bounds must be finite strictly ordered (2, {dimension}) vectors")
    return box


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
class LengthUnits:
    """Explicit isotropic length conversion: output lengths = coordinates * scale."""

    scale: float
    unit: str

    def __post_init__(self):
        if (isinstance(self.scale, (bool, np.bool_)) or not isinstance(self.scale, Real)
                or not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("length scale must be finite and positive")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("length unit must be a nonempty label")


@dataclass(frozen=True)
class AxisAlignedSurface:
    """Rectangle at coordinate on axis, with bounds on the other axes in XYZ order.

    normal is +1 or -1. side chooses the positive/negative coordinate-side
    cell at internal interfaces; domain faces always use the interior cell.
    Neither the side nor the selected component changes when normal is reversed.
    """

    axis: int | str
    coordinate: float
    bounds: np.ndarray
    normal: int = 1
    side: str = "positive"

    def __post_init__(self):
        axis = "xyz".index(self.axis) if isinstance(self.axis, str) and self.axis in ("x", "y", "z") else self.axis
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2):
            raise ValueError("axis must be x, y, z or 0, 1, 2")
        if not isinstance(self.coordinate, Real) or not math.isfinite(self.coordinate):
            raise ValueError("surface coordinate must be finite")
        if isinstance(self.normal, (bool, np.bool_)) or self.normal not in (-1, 1):
            raise ValueError("normal must be +1 or -1")
        if self.side not in ("positive", "negative"):
            raise ValueError("side must be positive or negative")
        object.__setattr__(self, "axis", int(axis))
        object.__setattr__(self, "bounds", frozen_array(_box(self.bounds, 2), float))


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
    """Own finite sampling positions, stable IDs and an optional image layout.

    Parameters
    ----------
    positions : array-like
        Finite physical positions (n, 3), copied into read-only storage.
    ids : array-like, optional
        Unique integer IDs; omission allocates consecutive IDs.
    shape : tuple of int, optional
        Layout containing exactly n positions; omission gives (n,).
    normals : array-like, optional
        One (3,) or n nonzero normals, normalized by construction.
    plane : Plane, optional
        Parent sampling surface; selection retains this association.
    """
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
        """Accounted geometry arrays in bytes, including the optional plane."""
        plane_arrays = () if self.plane is None else (self.plane.origin,self.plane.u,self.plane.v)
        return array_bytes((self.positions,self.ids,self.normals,*plane_arrays))

    @classmethod
    def from_plane(cls, plane, *, ids=None, memory_limit=None):
        """Copy all plane pixel centers with optional IDs and image layout.
        """
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
        """Sample a Cartesian box face with outward normals.

        mesh.lower and mesh.upper must describe that physical box in XYZ. This
        convenience constructor does not identify curved or unstructured domain
        boundaries; use explicit positions and normals for those surfaces.
        """
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
        """Return an owned flat PointSet selected by a matching boolean mask; preserve IDs.
        """
        rows = self.rows(mask)
        normals = self.normals
        if normals is not None and normals.ndim == 2:
            normals = normals[rows]
        return PointSet(self.positions[rows],self.ids[rows],normals=normals,plane=self.plane)

    def reshape(self, values):
        """Reshape arrays whose first axis matches the points into the stored layout.
        """
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(self):
            raise ValueError("values must have one row per point")
        return values.reshape(*self.shape,*values.shape[1:])


@dataclass(frozen=True, eq=False)
class RaySet:
    """Associate identified origins with normalized rays and depth clipping.

    Parameters
    ----------
    origins : PointSet
        Identified ray origins and their output layout.
    directions : array-like
        Shared (3,) or per-ray (n, 3) finite nonzero directions; normalized on copy.
    near, far : float or array-like
        Nonnegative coordinate-distance limits broadcast to the origin layout.
        near is finite; far is at least near and may be infinity.
    """
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
        """Construct identified parallel rays from a plane and coordinate-depth limits.
        """
        if not isinstance(plane,Plane):
            raise TypeError("a Plane is required")
        admit(math.prod(plane.shape)*160+1024,memory_limit,"plane rays")
        return cls(PointSet.from_plane(plane,ids=ids),direction,near,far)

    @property
    def nbytes(self):
        """Accounted origin geometry, directions and clipping arrays in bytes."""
        return self.origins.nbytes+array_bytes((self.directions,self.near,self.far))

    def select(self, mask):
        """Return selected ray geometry, preserving IDs and clipping distances.
        """
        rows = self.origins.rows(mask)
        direction = self.directions if self.directions.ndim == 1 else self.directions[rows]
        result = RaySet(self.origins.select(mask),direction,self.near[rows],self.far[rows])
        # Renormalization must not perturb a selected ray at a grazing face.
        object.__setattr__(result,"directions",frozen_array(direction,float))
        return result


@dataclass(frozen=True, eq=False)
class LineSet:
    """Packed identified trajectories with two branch slots per seed.

    Attributes
    ----------
    seeds : PointSet
        Original identified seeds.
    positions : ndarray
        Packed float64 coordinates (total_points, 3).
    offsets : ndarray
        Branch offsets (2*n+1,); negative branch precedes positive branch.
    termination : ndarray
        Termination codes (n, 2), plus NOT_REQUESTED=-1 and TANGENT_SEED=-2.
    source_identity : object
        In-memory traced-field identity; not verified after file loading.

    Notes
    -----
    Direct construction marks arrays read-only but does not remove external
    writable aliases. Keep any such alias unchanged during use and saving.
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
        """Accounted seed geometry and packed path arrays in bytes."""
        return self.seeds.nbytes+array_bytes((self.positions,self.offsets,self.termination))

    def branch(self, seed_id, direction):
        """Return the seed-to-endpoint coordinate view for seed_id and direction -1/+1.
        """
        matches = np.flatnonzero(self.seeds.ids == seed_id)
        if len(matches) != 1 or direction not in (-1,1):
            raise ValueError("select an existing seed ID and direction -1 or 1")
        index = 2*int(matches[0])+int(direction == 1)
        return self.positions[self.offsets[index]:self.offsets[index+1]]

    def line(self, seed_id):
        """Join against/along branches for display, omitting the duplicated seed.
        """
        negative,positive = self.branch(seed_id,-1),self.branch(seed_id,1)
        return np.concatenate((negative[::-1],positive[1:] if len(negative) else positive))

    def select(self, mask):
        """Copy selected seeds and both branches, retaining IDs and termination.

        Parameters
        ----------
        mask : array-like
            Boolean selection matching the seed layout.

        Returns
        -------
        LineSet
            Owned compact geometry in original seed order; source identity is
            retained without verifying the snapshot represented by loaded paths.
        """
        rows = self.seeds.rows(mask)
        seeds = self.seeds.select(mask)
        counts = np.diff(self.offsets).reshape(-1,2)[rows].ravel()
        positions = np.empty((int(counts.sum()),3),dtype=float)
        offsets = np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
        for selected,row in enumerate(rows):
            start,stop = self.offsets[2*row],self.offsets[2*row+2]
            positions[offsets[2*selected]:offsets[2*selected+2]] = self.positions[start:stop]
        return LineSet(seeds,positions,offsets,self.termination[rows],self.source_identity)
