"""Direct batched consumption of prepared component-adjacent values."""

import numpy as np


def sample(prepared, points):
    """Return (values, owners, valid); missing/outside/nonfinite samples are NaN.

    Sampling never performs I/O or changes a pool. A caller may prepare missing
    owners and explicitly retry. A derivative needs at least one valid layer.
    """
    from simesh.utils.lib.analysis.native import sample_ready
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n,3)")
    if prepared.halo < 1:
        raise ValueError("trilinear sampling requires one valid layer")
    mesh = prepared.mesh
    values = np.empty((len(points), len(prepared.fields)), dtype=np.float64)
    owners = np.empty(len(points), dtype=np.int64)
    valid = np.empty(len(points), dtype=np.uint8)
    sample_ready(mesh.lower, mesh.upper, mesh.roots, mesh.children, mesh.node_leaves,
                 mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
                 prepared.slot_of_leaf, prepared.values, prepared.halo,
                 points, values, owners, valid)
    return values, owners, valid.view(np.bool_)
