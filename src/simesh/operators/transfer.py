"""Ordered transfer of prepared emission/opacity, independent of physical models.

The application validates coefficients, ray geometry and controls. This operation
owns support-dependent memory admission, native/reference traversal and transfer
arithmetic; it neither reads sources nor supplies missing field coverage.
"""

import numpy as np

from .._validation import admit
from .rays import LOSStatus, ray_segments, ray_nodes, _run_ray_batches
from .sampling import sample


def _integrate_transfer_rays(coefficients, rays, length_unit_cm, background, subdivisions,
                             max_samples, workers, ray_batch, memory_limit, implementation):
    """Return intensity, depths, status, counts, opacity and absorption arrays."""
    n = len(rays.origins)
    mesh = coefficients.mesh
    scratch = (mesh.leaf_count*128+2*subdivisions*(sum(mesh.block_shape)+1)*384
               if implementation == "reference" else min(n,ray_batch)*128)
    admit(mesh.nbytes+coefficients.nbytes+rays.nbytes+n*160+scratch+workers*65536,
          memory_limit, "radiative LOS")
    values, entry, exit, tau, thin = (np.zeros(n) for _ in range(5))
    status, samples, misses = (np.zeros(n, dtype=np.int64) for _ in range(3))
    if implementation == "reference":
        directions = np.broadcast_to(rays.directions, (n, 3))
        _reference_transfer(coefficients, rays, directions, length_unit_cm, subdivisions,
                            max_samples, values, entry, exit, tau, thin, status, samples)
    else:
        from .._kernels.native import initialize_ray_set
        from .._kernels.thermal_rays import integrate_transfer_ray_set_ready

        backing = coefficients.values

        def consume(span, origins, directions, near, far):
            initialize_ray_set(mesh.lower, mesh.upper, origins, directions, near, far,
                               entry[span], exit[span], status[span])
            integrate_transfer_ray_set_ready(mesh.roots, mesh.children, mesh.node_leaves,
                mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
                coefficients.slot_of_leaf, backing, coefficients.storage_halo,
                origins, directions, entry[span], exit[span], subdivisions, max_samples, length_unit_cm,
                values[span], status[span], samples[span], tau[span], thin[span])

        _run_ray_batches(rays, ray_batch, workers, consume)
    valid = np.isin(status, (LOSStatus.COMPLETE, LOSStatus.EMPTY))
    # Compute before adding background: subtracting a bright background later
    # can erase the emitted signal and corrupt the absorption diagnostic.
    fraction = np.ones_like(thin)
    np.divide(values, thin, out=fraction, where=thin > 0)
    fraction = np.clip(1-fraction, 0., 1.)
    with np.errstate(over="ignore", invalid="ignore"):
        values += float(background)*np.exp(-tau)
    bad = valid & ~(np.isfinite(values) & np.isfinite(tau) & np.isfinite(thin))
    status[bad] = LOSStatus.UNREPRESENTABLE_INTEGRAL
    for data in (values, tau, thin, fraction):
        data[~valid | bad] = np.nan
    return values, entry, exit, status, samples, misses, tau, thin, fraction


def _reference_transfer(fields, rays, directions, length, subdivisions, limit,
                        values, entry, exit, tau, thin, status, samples):
    mesh = fields.mesh
    for row, (origin, direction) in enumerate(zip(rays.origins.positions, directions)):
        leaves, first, last = ray_segments(mesh, origin, direction, rays.near[row], rays.far[row])
        status[row] = LOSStatus.EMPTY
        if not len(leaves):
            continue
        entry[row], exit[row] = first[0], last[-1]
        status[row] = LOSStatus.COMPLETE
        if not np.allclose(first[1:], last[:-1], rtol=2e-13, atol=2e-13):
            status[row] = LOSStatus.GEOMETRY_FAILURE
        for leaf, lo, hi in zip(leaves, first, last):
            if status[row] != LOSStatus.COMPLETE:
                break
            if fields.slot_of_leaf[leaf] < 0:
                status[row] = LOSStatus.MISSING_COVERAGE
                break
            nodes, weights = ray_nodes(mesh, leaf, origin, direction, lo, hi, subdivisions)
            nodes = nodes.reshape(-1, 2).mean(axis=1)
            widths = weights.reshape(-1, 2).sum(axis=1)*length
            if samples[row]+len(nodes) > limit:
                status[row] = LOSStatus.SAMPLE_LIMIT
                break
            points = origin+nodes[:, None]*direction
            points = np.maximum(mesh.bounds[leaf, 0], np.minimum(points,
                                np.nextafter(mesh.bounds[leaf, 1], mesh.bounds[leaf, 0])))
            data, owners, valid = sample(fields, points)
            if not np.all(valid) or not np.all(owners == leaf):
                status[row] = LOSStatus.UNREPRESENTABLE_SAMPLE
                break
            if not np.isfinite(data).all() or np.any(data < 0):
                status[row] = LOSStatus.NONFINITE_SCALAR
                break
            with np.errstate(over="ignore", invalid="ignore"):
                for (j, kappa), ds in zip(data, widths):
                    dtau = kappa*ds
                    emission = j*ds
                    thin[row] += emission
                    slab = -np.expm1(-dtau)/dtau if dtau > 0 else 1.
                    values[row] += np.exp(-tau[row])*emission*slab
                    tau[row] += dtau
                if not np.isfinite([values[row], tau[row], thin[row]]).all():
                    status[row] = LOSStatus.UNREPRESENTABLE_INTEGRAL
            samples[row] += len(nodes)
