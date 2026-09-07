"""Plain immutable geometry shared by preparation and consumers."""

from dataclasses import dataclass

import numpy as np


def frozen_array(value, dtype):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def indices(value, size, name="leaf_ids"):
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(array < 0) or np.any(array >= size):
        raise ValueError(f"{name} outside available range")
    result = np.ascontiguousarray(array, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True, eq=False)
class MeshIndex:
    """Provider-validated Cartesian 3D geometry, copied at adapter publication.

    Construct through a validated provider adapter. Arrays must be immutable;
    no payload, source offsets or temporary storage slots belong here.
    """

    lower: np.ndarray
    upper: np.ndarray
    block_shape: tuple
    roots: np.ndarray
    children: np.ndarray
    node_leaves: np.ndarray
    node_lower: np.ndarray
    node_upper: np.ndarray
    leaf_nodes: np.ndarray
    bounds: np.ndarray
    spacing: np.ndarray

    @property
    def leaf_count(self):
        return self.leaf_nodes.size

    @property
    def nbytes(self):
        return sum(value.nbytes for value in vars(self).values()
                   if isinstance(value, np.ndarray))

    def locate(self, points):
        from simesh.utils.lib.analysis.native import locate
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n,3)")
        owners = np.empty(len(points), dtype=np.int64)
        locate(self.lower, self.upper, self.roots, self.children, self.node_leaves,
               self.node_lower, self.node_upper, points, owners)
        return owners

    def descendants(self, node):
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
        return np.asarray(leaves, dtype=np.int64)

    def select_box(self, lower, upper):
        """Return leaves and exact half-open cell-center windows in a box."""
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,) or
                not np.all(np.isfinite([lower, upper])) or np.any(lower > upper)):
            raise ValueError("finite ordered three-dimensional bounds required")
        # Search explicit centers to preserve nextafter ownership at cell faces.
        first = np.empty((self.leaf_count, 3), dtype=np.int64)
        last = np.empty_like(first)
        for axis, extent in enumerate(self.block_shape):
            centers = (self.bounds[:, 0, axis, None] +
                       (np.arange(extent) + .5) * self.spacing[:, axis, None])
            first[:, axis] = np.sum(centers < lower[axis], axis=1)
            last[:, axis] = np.sum(centers < upper[axis], axis=1)
        selected = np.flatnonzero(np.all(first < last, axis=1))
        return selected, first[selected], last[selected]
