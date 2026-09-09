"""Direct sampling of completed native fields with no source access."""

import numpy as np

from .._execution import worker_context, run_ranges
from .._validation import admit
from ..fields import require_continuous, _input_arrays


def _sample(fields, points, workers, executor, field_positions, output=None):
    from .._kernels.native import sample_ready
    mesh = fields.mesh
    if output is None:
        output = (np.empty((len(points), len(field_positions))), np.empty(len(points), dtype=np.int64),
                  np.empty(len(points), dtype=bool))
    output, owners, valid = output
    valid_bytes = valid.view(np.uint8)
    values = fields.values

    def fill(first, last):
        sample_ready(mesh.lower, mesh.upper, mesh.roots, mesh.children, mesh.node_leaves,
            mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
            fields.slot_of_leaf, values, fields.storage_halo, points[first:last],
            output[first:last], owners[first:last], valid_bytes[first:last], field_positions)

    run_ranges(len(points), workers, fill, executor)
    return output, owners, valid.view(np.bool_)


def sample(fields, points, *, components=None, output=None, workers=1, memory_limit=None):
    """Return selected (values, original owners, validity); missing values are NaN.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    points : array-like
        Physical coordinates with shape (n, 3).
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    output : tuple of ndarray, optional
        Writable contiguous (float64 values, int64 owners, bool valid) arrays matching
        the return shapes; no input aliases. Failure may leave partial writes.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    tuple of ndarray
        Values (n, k), owner IDs (n,), and coverage mask (n,). Coverage alone does not
        establish finite values.
    """
    selected = np.asarray(require_continuous(fields, components, operation="sample"),dtype=np.int64)
    points = np.ascontiguousarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    admit(fields.mesh.nbytes + fields.nbytes + points.nbytes +
          len(points)*(8*len(selected)+9), memory_limit, "sampling")
    output = _sample_output(output, len(points), len(selected), (*_input_arrays(fields), points))
    with worker_context(workers) as executor:
        return _sample(fields, points, workers, executor, selected, output)


def _sample_output(output, count, width, inputs=()):
    """Validate caller targets before writes, including noncontiguous row layouts."""
    if output is None:
        return None
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise ValueError("output must be (values, owners, valid)")
    for array, shape, dtype in zip(output, ((count,width),(count,),(count,)),
                                   (np.float64,np.int64,np.bool_)):
        if (not isinstance(array,np.ndarray) or array.shape != shape or array.dtype != dtype or
                not array.flags.writeable or any(s <= 0 for n,s in zip(shape,array.strides) if n>1)):
            raise ValueError("output arrays must have matching shapes/dtypes and writable positive strides")
        # Row and component strides must describe disjoint elements.
        axes = sorted((s,n) for s,n in zip(array.strides,shape) if n>1)
        span = array.itemsize
        for stride, n in axes:
            if stride < span:
                raise ValueError("output elements must not overlap")
            span += (n-1)*stride
        if any(np.shares_memory(array, source) for source in inputs):
            raise ValueError("output must not alias inputs")
    if any(np.shares_memory(output[i],output[j]) for i in range(3) for j in range(i)):
        raise ValueError("output arrays must not overlap")
    return tuple(output)
