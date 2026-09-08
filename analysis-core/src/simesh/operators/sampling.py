"""Direct sampling of completed native fields with no source access."""

import numpy as np

from .._execution import worker_context, run_ranges
from .._validation import admit
from ..fields import require_fields


def _sample(fields, points, workers, executor):
    from .._kernels.native import sample_ready
    mesh = fields.mesh
    output = np.empty((len(points), len(fields.fields)), dtype=float)
    owners = np.empty(len(points), dtype=np.int64)
    valid = np.empty(len(points), dtype=np.uint8)
    values = fields.values

    def fill(first, last):
        sample_ready(mesh.lower, mesh.upper, mesh.roots, mesh.children, mesh.node_leaves,
            mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
            fields.slot_of_leaf, values, fields.storage_halo, points[first:last],
            output[first:last], owners[first:last], valid[first:last])

    run_ranges(len(points), workers, fill, executor)
    return output, owners, valid.view(np.bool_)


def sample(fields, points, *, workers=1, memory_limit=None):
    """Return (values, original owners, validity); missing samples are NaN."""
    fields = require_fields(fields, halo=1)
    points = np.ascontiguousarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    admit(fields.mesh.nbytes + fields.nbytes + points.nbytes +
          len(points)*(8*len(fields.fields)+9), memory_limit, "sampling")
    with worker_context(workers) as executor:
        return _sample(fields, points, workers, executor)
