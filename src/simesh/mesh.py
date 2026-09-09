"""Shared immutable AMR topology and geometry, without field storage."""

from dataclasses import dataclass
import numpy as np

from ._validation import frozen_array, indices, array_bytes
from ._amr.forest import RefinedForest, refined_forest
from ._amr.morton import level1_morton
from ._amr.refined_geometry import refined_leaf_geometry
from ._amr.balance import validate_refined_all_touch_2to1


@dataclass(frozen=True, eq=False)
class Selection:
    """Ordered complete target leaves and an optional requested physical box.

    Attributes
    ----------
    mesh : Mesh
        Original geometry; region edges are not physical boundaries.
    leaf_ids : ndarray
        Ordered original leaf IDs, not storage rows.
    requested_bounds : ndarray, optional
        Finite (2, 3) box remembered for cell-overlap reductions.
    """

    mesh: "Mesh"
    leaf_ids: np.ndarray
    requested_bounds: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "leaf_ids", frozen_array(
            indices(self.leaf_ids, self.mesh.leaf_count), np.int64))
        if self.requested_bounds is not None:
            box = np.asarray(self.requested_bounds, dtype=float)
            if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
                raise ValueError("requested bounds must be a finite ordered box")
            object.__setattr__(self, "requested_bounds", frozen_array(box, float))


@dataclass(frozen=True, eq=False)
class Mesh:
    """Immutable Cartesian 3D topology and geometry, without field values.

    Construct with mesh_from_forest or obtain source.mesh.

    Attributes
    ----------
    lower, upper : ndarray
        Original physical domain bounds (3,).
    block_shape, root_shape : sequence of int
        Interior cells per leaf and root-block counts.
    bounds, spacing : ndarray
        Per-leaf physical bounds (leaf, 2, 3) and cell spacing (leaf, 3).

    Notes
    -----
    Native ownership uses half-open bounds. Region selection retains the original
    Mesh; its edges are not physical boundaries.
    """

    lower: np.ndarray
    upper: np.ndarray
    block_shape: tuple
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    forest: RefinedForest
    roots: np.ndarray
    node_lower: np.ndarray
    node_upper: np.ndarray
    bounds: np.ndarray
    spacing: np.ndarray

    @property
    def children(self):
        """Child-node directory for the immutable forest."""
        return self.forest.child_node_ids

    @property
    def node_leaves(self):
        """Node to original leaf ID mapping; parent nodes have no leaf ID."""
        return self.forest.node_leaf_ids

    @property
    def leaf_nodes(self):
        """Original leaf ID to tree-node mapping."""
        return self.forest.leaf_node_ids

    @property
    def leaf_count(self):
        """Number of original leaf blocks."""
        return len(self.leaf_nodes)

    @property
    def nbytes(self):
        """Accounted immutable topology and geometry arrays in bytes."""
        return array_bytes((*vars(self).values(), *self.forest))

    def locate(self, points):
        """Return original owner leaf IDs for (n, 3) points; outside owners are -1.
        """
        from ._kernels.native import locate
        points = np.ascontiguousarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        owners = np.empty(len(points), dtype=np.int64)
        locate(self.lower, self.upper, self.roots, self.children, self.node_leaves,
               self.node_lower, self.node_upper, points, owners)
        return owners

    def descendants(self, node):
        """Return a Selection of all leaves descended from an original tree node.
        """
        if not isinstance(node, (int, np.integer)) or not 0 <= node < len(self.children):
            raise ValueError("node outside forest")
        pending, leaves = [int(node)], []
        while pending:
            current = pending.pop()
            leaf = self.node_leaves[current]
            if leaf >= 0:
                leaves.append(leaf)
            else:
                pending.extend(reversed(self.children[current].tolist()))
        return Selection(self, leaves)


def _from_validated(root_shape, coord_to_rank, forest, lower, upper, block_shape):
    """Publish geometry from an already validated integer forest binding."""
    root = frozen_array(root_shape, np.int64)
    rank = frozen_array(coord_to_rank, np.int64)
    lower, upper = frozen_array(lower, float), frozen_array(upper, float)
    block = np.asarray(block_shape)
    if (block.shape != (3,) or block.dtype.kind not in "iu" or np.any(block < 1) or
            lower.shape != (3,) or upper.shape != (3,) or
            not np.isfinite([lower, upper]).all() or np.any(lower >= upper)):
        raise ValueError("finite Cartesian 3D bounds and positive integer block shape required")
    block = frozen_array(block, np.int64)
    f = RefinedForest(*(frozen_array(a, np.int64) if isinstance(a, np.ndarray) else a for a in forest))
    validate_refined_all_touch_2to1(root, rank, f.root_node_ids, f.node_levels,
        f.node_coords, f.child_node_ids, f.node_leaf_ids, f.leaf_node_ids)
    domain = root * block
    bounds, spacing = refined_leaf_geometry(lower, upper, root, domain, block,
        f.node_levels, f.node_coords, f.leaf_node_ids, np.arange(len(f.leaf_node_ids), dtype=np.int64))
    # Preserve the accepted coordinate arithmetic, including exact domain faces.
    scales = np.left_shift(1, f.node_levels - 1)
    dx = np.ldexp((upper - lower) / domain, -(f.node_levels - 1)[:, None])
    lower_indices = f.node_coords * block
    upper_indices = lower_indices + block
    nlo = lower + lower_indices * dx
    nhi = lower + upper_indices * dx
    nhi = np.where(upper_indices == domain * scales[:, None], upper, nhi)
    return Mesh(lower, upper, tuple(map(int, block)), root, rank, f,
                frozen_array(f.root_node_ids[rank], np.int64), frozen_array(nlo, float),
                frozen_array(nhi, float), frozen_array(bounds, float), frozen_array(spacing, float))


def mesh_from_forest(root_shape, is_leaf, *, lower, upper, block_shape):
    """Build a balanced Cartesian 3D mesh from depth-first Morton leaf flags."""
    root = np.asarray(root_shape)
    if root.shape != (3,) or root.dtype.kind not in "iu" or np.any(root < 1):
        raise ValueError("root_shape must contain three positive integers")
    root = np.ascontiguousarray(root, dtype=np.int64)
    flags = np.asarray(is_leaf)
    if flags.ndim != 1 or flags.dtype != np.bool_:
        raise ValueError("is_leaf must be a one-dimensional boolean array")
    rank, coordinates = level1_morton(root)
    f = refined_forest(root, rank, coordinates, np.ascontiguousarray(flags))
    return _from_validated(root, rank, f, lower, upper, block_shape)


def select_region(mesh, bounds):
    """Select whole leaves intersecting a half-open box, retaining global geometry."""
    box = np.asarray(bounds, dtype=float)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
        raise ValueError("bounds must contain finite strictly ordered 3D vectors")
    ids = np.flatnonzero(np.all((mesh.bounds[:, 1] > box[0]) & (mesh.bounds[:, 0] < box[1]), axis=1))
    if not len(ids):
        raise ValueError("bounds do not intersect the source domain")
    return Selection(mesh, ids, box)


def resolve_selection(mesh, region=None, leaf_ids=None):
    if region is not None and leaf_ids is not None:
        raise ValueError("choose region or leaf_ids")
    if isinstance(region, Selection):
        if region.mesh is not mesh:
            raise ValueError("selection belongs to a different mesh")
        return region
    if region is not None:
        return select_region(mesh, region)
    return Selection(mesh, np.arange(mesh.leaf_count, dtype=np.int64) if leaf_ids is None else leaf_ids)
