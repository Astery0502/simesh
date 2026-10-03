"""Native Cartesian AMR sections and cell quadrature.

AxisSlice retains the original immutable Mesh. Cell edges and area weights follow
that Mesh's Cartesian block layout. Mesh-independent sampling objects are defined
in simesh.spatial and remain available here as public aliases.
"""

from dataclasses import dataclass, field
from numbers import Real
import numpy as np

from .mesh import Mesh
from .spatial import (Plane as Plane, PointSet, RaySet as RaySet, LineSet as LineSet,
                      LengthUnits as LengthUnits, AxisAlignedSurface as AxisAlignedSurface)


def _axis_candidates(mesh, axis, coordinate, side):
    """Select one side of an interface; physical domain faces take the inward trace."""
    if coordinate == mesh.lower[axis]:
        side = "positive"
    elif coordinate == mesh.upper[axis]:
        side = "negative"
    lo, hi = mesh.bounds[:, 0, axis], mesh.bounds[:, 1, axis]
    active = ((lo <= coordinate) & (coordinate < hi) if side == "positive"
              else (lo < coordinate) & (coordinate <= hi))
    return active, side


def _cell_edges(mesh, leaf, axis, offsets=None):
    """Build original cell edges, preserving the exact upper block face."""
    if offsets is None:
        offsets = np.arange(mesh.block_shape[axis] + 1)
    edges = mesh.bounds[leaf, 0, axis] + offsets*mesh.spacing[leaf, axis]
    edges[-1] = mesh.bounds[leaf, 1, axis]
    return edges


def _cell_index(edges, coordinate, side):
    """Choose an adjacent interior cell using explicit edges, not a rounded quotient."""
    index = int(np.searchsorted(edges, coordinate, side="right" if side == "positive" else "left")) - 1
    return min(max(index, 0), len(edges) - 2)


def _uniform_geometry(mesh,resolution,bounds):
    resolution=tuple(resolution)
    if len(resolution)!=3 or any(not isinstance(n,(int,np.integer)) or n<1 for n in resolution):
        raise ValueError("resolution needs three positive integer extents")
    lower,upper=(mesh.lower,mesh.upper) if bounds is None else bounds
    lower,upper=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or np.any(upper<=lower):
        raise ValueError("uniform bounds must be finite ordered triplets")
    return tuple(map(int,resolution)),lower,upper


@dataclass(frozen=True, eq=False)
class AxisSlice:
    """Describe complete native block sections on an axis-aligned plane.

    Parameters
    ----------
    mesh : Mesh
        Shared immutable original geometry; no field storage is retained.
    axis : int or str
        Normal axis: 0, 1, 2 or x, y, z. Stored as an integer.
    coordinate : float
        Finite plane position in mesh coordinates, including domain faces.
    side : str
        Positive or negative coordinate-side cell at internal interfaces,
        spelled 'positive' or 'negative'. Domain faces always use the interior.

    Attributes
    ----------
    leaf_ids, cell_indices : ndarray
        Read-only int64 arrays (nblocks,): ascending original leaf IDs and
        corresponding normal-axis interior-cell indices, excluding halo.

    Notes
    -----
    Geometry covers the full domain section, independently of supplied fields.
    There is no transverse clipping or new two-dimensional AMR tree.
    """

    mesh: Mesh
    axis: int | str
    coordinate: float
    side: str = "positive"
    leaf_ids: np.ndarray = field(init=False)
    cell_indices: np.ndarray = field(init=False)

    def __post_init__(self):
        if not isinstance(self.mesh, Mesh):
            raise TypeError("axis slice requires a Mesh")
        axis = self.axis
        if isinstance(axis, str) and axis in ("x", "y", "z"):
            axis = "xyz".index(axis)
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2):
            raise ValueError("axis must be x, y, z or 0, 1, 2")
        if (isinstance(self.coordinate, (bool, np.bool_)) or not isinstance(self.coordinate, Real)
                or not np.isfinite(self.coordinate)):
            raise ValueError("slice coordinate must be finite")
        if self.side not in ("positive", "negative"):
            raise ValueError("side must be positive or negative")
        mesh, coordinate = self.mesh, float(self.coordinate)
        if not mesh.lower[axis] <= coordinate <= mesh.upper[axis]:
            raise ValueError("slice coordinate outside original domain")
        active, side = _axis_candidates(mesh, axis, coordinate, self.side)
        ids = np.flatnonzero(active)
        offsets = np.arange(mesh.block_shape[axis] + 1)
        indices = np.empty(len(ids), dtype=np.int64)
        for row, leaf in enumerate(ids):
            indices[row] = _cell_index(_cell_edges(mesh, leaf, axis, offsets), coordinate, side)
        object.__setattr__(self, "axis", int(axis))
        object.__setattr__(self, "coordinate", coordinate)
        ids.flags.writeable = indices.flags.writeable = False
        object.__setattr__(self, "leaf_ids", ids)
        object.__setattr__(self, "cell_indices", indices)

    @property
    def axes(self):
        """Transverse axis indices in XYZ order, without display transposition."""
        return tuple(a for a in range(3) if a != self.axis)

    @property
    def block_shape(self):
        """Interior cell counts (nu, nv), shared by all output blocks."""
        return tuple(self.mesh.block_shape[a] for a in self.axes)

    @property
    def bounds(self):
        """Read-only block bounds (nblocks, lower/upper, u/v), in mesh coordinates."""
        bounds = self.mesh.bounds[np.ix_(self.leaf_ids, (0, 1), self.axes)]
        bounds.flags.writeable = False
        return bounds

    @property
    def spacing(self):
        """Read-only transverse cell spacing (nblocks, 2), in mesh coordinates."""
        spacing = self.mesh.spacing[np.ix_(self.leaf_ids, self.axes)]
        spacing.flags.writeable = False
        return spacing

    @property
    def levels(self):
        """Read-only original refinement levels (nblocks,), with root level one."""
        levels = self.mesh.forest.node_levels[self.mesh.leaf_nodes[self.leaf_ids]]
        levels.flags.writeable = False
        return levels

    def cell_edges(self, row):
        """Return read-only u/v cell-edge vectors for an output row, not a leaf ID."""
        if (isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer))
                or not 0 <= row < len(self.leaf_ids)):
            raise ValueError("row outside slice blocks")
        leaf = self.leaf_ids[row]
        result = []
        for axis in self.axes:
            edges = _cell_edges(self.mesh, leaf, axis)
            edges.flags.writeable = False
            result.append(edges)
        return tuple(result)

    @property
    def nbytes(self):
        """Accounted geometry arrays, including the shared original Mesh."""
        return self.mesh.nbytes + self.leaf_ids.nbytes + self.cell_indices.nbytes


def native_bottom_seeds(mesh):
    """Return native bottom-face cell centers and their quadrature areas.

    Parameters
    ----------
    mesh : Mesh
        Original Cartesian AMR geometry, including the full physical bottom.

    Returns
    -------
    points : PointSet
        One point on zmin per intersecting leaf cell, ordered by leaf, x, y.
    areas : ndarray
        Owned positive cell-face areas (n,) in squared coordinate-length units.
        Refinement changes both seed density and weights, with no halo seeds.
    """
    section = AxisSlice(mesh,'z',float(mesh.lower[2]))
    positions, areas = [], []
    for row in range(len(section.leaf_ids)):
        x,y = section.cell_edges(row)
        xx,yy = np.meshgrid(.5*(x[:-1]+x[1:]),.5*(y[:-1]+y[1:]),indexing='ij')
        positions.append(np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,mesh.lower[2]))))
        areas.append((np.diff(x)[:,None]*np.diff(y)[None,:]).ravel())
    return PointSet(np.concatenate(positions)), np.concatenate(areas)
