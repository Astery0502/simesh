"""Cartesian ray quadrature and native scalar integration.

Leaf intersections and knot quadrature use coordinate-length units. Private
batch drivers consume validated fields, ray geometry and numerical controls;
they bind native storage without reading sources or applying physical models.
"""

from enum import IntEnum
import numpy as np

from .._validation import admit
from .._execution import worker_context, run_ranges


class LOSStatus(IntEnum):
    """Ray states. COMPLETE and EMPTY are valid; missing coverage, sample limits and numerical failures are not.
    """
    RUNNING = 0
    COMPLETE = 1
    EMPTY = 2
    MISSING_COVERAGE = 3
    NONFINITE_SCALAR = 4
    GEOMETRY_FAILURE = 5
    SAMPLE_LIMIT = 6
    UNREPRESENTABLE_INTEGRAL = 7
    UNREPRESENTABLE_SAMPLE = 8


def ray_segments(mesh, origin, direction, near, far):
    """Independent vectorized leaf intersections, ordered in physical arclength."""
    first = np.full(mesh.leaf_count, near, dtype=float)
    last = np.full(mesh.leaf_count, far, dtype=float)
    inside = np.ones(mesh.leaf_count, dtype=bool)
    for a in range(3):
        if direction[a] == 0:
            inside &= (origin[a] >= mesh.bounds[:, 0, a]) & (origin[a] < mesh.bounds[:, 1, a])
        else:
            x = (mesh.bounds[:, 0, a]-origin[a])/direction[a]
            y = (mesh.bounds[:, 1, a]-origin[a])/direction[a]
            first = np.maximum(first, np.minimum(x, y))
            last = np.minimum(last, np.maximum(x, y))
    leaves = np.flatnonzero(inside & (first < last))
    leaves = leaves[np.argsort(first[leaves], kind="stable")]
    return leaves, first[leaves], last[leaves]


def ray_nodes(mesh, leaf, origin, direction, first, last, subdivisions):
    """Split at cell-center interpolation knots, then composite two-point Gauss."""
    knots = [first, last]
    for a, n in enumerate(mesh.block_shape):
        if direction[a] != 0:
            times = (mesh.bounds[leaf, 0, a]+(np.arange(n)+.5)*mesh.spacing[leaf, a]-origin[a])/direction[a]
            knots.extend(times[(times > first) & (times < last)])
    knots = np.unique(knots)
    lo, width = knots[:-1], np.diff(knots)/subdivisions
    starts = (lo[:, None]+width[:, None]*np.arange(subdivisions)).ravel()
    weights = np.repeat(width, subdivisions)/2
    nodes = (starts[:, None]+weights[:, None]*(1+np.array([-1., 1.])/np.sqrt(3))).ravel()
    return nodes, np.repeat(weights, 2)


def _run_ray_batches(rays, ray_batch, workers, consume):
    """Bind contiguous directions per batch and finish all ranges before reuse."""
    count = len(rays.origins)
    directions = np.broadcast_to(rays.directions, (count, 3))
    with worker_context(workers) as executor:
        for start in range(0, count, ray_batch):
            stop = min(start+ray_batch, count)
            direction_batch = np.ascontiguousarray(directions[start:stop])

            def run(first, last):
                span = slice(start+first, start+last)
                consume(span, rays.origins.positions[span], direction_batch[first:last],
                        rays.near[span], rays.far[span])

            run_ranges(stop-start, workers, run, executor)


def _integrate_scalar_rays(fields, rays, component, quadrature, step_fraction,
                           max_samples, workers, ray_batch, memory_limit):
    """Integrate validated scalar rays and return six flat result arrays."""
    from .._kernels.native import integrate_ray_set

    mesh = fields.mesh
    count = len(rays.origins)
    admit(mesh.nbytes+fields.nbytes+rays.nbytes+count*48+min(count, ray_batch)*384,
          memory_limit, "ray-set LOS")
    values, entry, exit = (np.empty(count) for _ in range(3))
    status, samples, misses = (np.empty(count, dtype=np.int64) for _ in range(3))
    backing = fields.values

    def consume(span, origins, directions, near, far):
        integrate_ray_set(mesh.lower, mesh.upper, mesh.roots, mesh.children,
            mesh.node_leaves, mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
            fields.slot_of_leaf, backing, fields.storage_halo, origins, directions, near, far,
            component, step_fraction, int(quadrature == "gauss2"), max_samples,
            values[span], entry[span], exit[span], status[span], samples[span], misses[span])

    _run_ray_batches(rays, ray_batch, workers, consume)
    return values, entry, exit, status, samples, misses
