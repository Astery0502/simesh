"""Magnetic squashing factors and complete-line twist on native AMR fields."""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from .fields import FieldDefinition, publish, require_fields
from .operators.derivatives import derivative
from .operators.sampling import sample
from .slices import _transverse_basis
from .tracing import Termination, _validate_vector, _resolve_curl
from ._validation import admit, array_bytes, remaining, workers_count
from ._execution import worker_context, run_ranges


class Boundary(IntEnum):
    """Localized endpoint surface identifiers, including local spheres and edges.
    """
    NONE = 0
    ZMIN = 1
    ZMAX = 2
    YMIN = 3
    YMAX = 4
    XMIN = 5
    XMAX = 6
    EDGE = 7
    LOCAL_SPHERE = 9


class ConnectivityTermination(IntEnum):
    """Connectivity stop reasons, extending native tracing with local exit and step underflow.
    """
    MAX_STEPS = Termination.MAX_STEPS
    MAX_LENGTH = Termination.MAX_LENGTH
    DOMAIN_EXIT = Termination.DOMAIN_EXIT
    NULL_FIELD = Termination.NULL_FIELD
    NONFINITE_FIELD = Termination.NONFINITE_FIELD
    OUTSIDE_SEED = Termination.OUTSIDE_SEED
    MISSING_COVERAGE = Termination.MISSING_COVERAGE
    UNREPRESENTABLE_NORM = Termination.UNREPRESENTABLE_NORM
    NONFINITE_DIAGNOSTIC = Termination.NONFINITE_DIAGNOSTIC
    UNREPRESENTABLE_SAMPLE = Termination.UNREPRESENTABLE_SAMPLE
    STEP_UNDERFLOW = 11
    LOCAL_EXIT = 12


@dataclass(frozen=True)
class QSLResult:
    """Owned per-seed diagnostics with against/along endpoint order.

    Attributes
    ----------
    seeds : ndarray
        Seed coordinates (n, 3).
    q, log10_q, q_perp, log10_q_perp, twist : ndarray, optional
        Per-seed diagnostics; unrequested quantities are None. Finite Q-perpendicular
        or twist can describe an incomplete accepted segment; inspect complete and
        both termination codes.
    length, complete, valid, stencil_valid : ndarray
        Coordinate length, whole-line completion, Q validity and stencil flags.
    footpoints, endpoint_fields : ndarray
        Localized positions and endpoint fields (n, 2, 3), against/along order.
    boundary, termination, steps : ndarray
        Per-branch surface IDs, stop reasons and accepted step counts (n, 2).
    normalization, method, local_radius : object
        Numerical definition of the requested map.

    Notes
    -----
    Mapping Q requires two localized transverse target surfaces. With local_radius,
    Q describes the local sphere/box map.
    """

    seeds: np.ndarray
    q: np.ndarray | None
    log10_q: np.ndarray | None
    q_perp: np.ndarray | None
    log10_q_perp: np.ndarray | None
    twist: np.ndarray | None
    length: np.ndarray
    footpoints: np.ndarray
    endpoint_fields: np.ndarray
    boundary: np.ndarray
    termination: np.ndarray
    steps: np.ndarray
    complete: np.ndarray
    valid: np.ndarray
    normalization: str
    local_radius: float | None
    method: str
    stencil_valid: np.ndarray

    @property
    def q_local(self):
        """Local mapping Q when local_radius was requested; otherwise None."""
        return self.q if self.local_radius is not None else None


def _unit_gradient(fields, memory_limit):
    require_fields(fields, halo=2)
    block = fields.mesh.block_shape
    shape = (len(fields.leaf_ids), *(n+4 for n in block), 3)
    output_bytes = 8*int(np.prod(shape))
    scratch = 64*int(np.prod(shape[1:4]))
    admit(fields.mesh.nbytes+fields.nbytes+output_bytes+scratch,
          memory_limit, "unit-vector nodes")
    values = np.empty(shape)
    offset = fields.storage_halo
    box = tuple(slice(offset-2, offset+n+2) for n in block)
    backing = fields.values
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for row, leaf in enumerate(fields.leaf_ids):
            b = backing[(fields.slot_of_leaf[leaf], *box, slice(None))]
            norm = np.hypot(np.hypot(b[...,0], b[...,1]), b[...,2])
            values[row] = b/norm[...,None]
            del b, norm
    unit = publish(fields.mesh, values, fields.selection,
                   tuple(FieldDefinition("unit_"+axis, "1", "prepared-node") for axis in "xyz"),
                   2, 2, fields.scheme+"/unit-nodes", fields.source)
    terms = [[(component, axis, 1.)] for component in range(3) for axis in range(3)]
    definitions = [FieldDefinition(f"dunit_{component}_{axis}", "1 / coordinate-length",
                                   "centered-derivative")
                   for component in range(3) for axis in range(3)]
    return derivative(unit, terms, definitions, memory_limit=remaining(memory_limit, fields.nbytes))


def _require_support(fields, *, compute_q=True, method="finite-difference", twist=True, curl_field=None):
    _validate_vector(fields)
    if compute_q and method == "variational":
        require_fields(fields,halo=2,operation="variational QSL")
    if twist and curl_field is None:
        require_fields(fields,halo=2,operation="twist with automatic curl")
    if curl_field is not None:
        require_fields(curl_field,halo=1,operation="retained curl interpolation")


def _controls(fields, seeds, bounds, step_fraction, step, max_steps, max_length,
              null_threshold, boundary_tolerance, local_radius, normalization, workers, twist,
              method, delta, compute_q=True):
    _validate_vector(fields)
    if method not in ("variational", "finite-difference"):
        raise ValueError("method must be 'variational' or 'finite-difference'")
    if not compute_q and delta is not None:
        raise ValueError("delta only applies when Q is requested")
    if compute_q and method == "variational":
        require_fields(fields, halo=2)
        if delta is not None:
            raise ValueError("delta only applies to finite-difference mapping")
    if delta is not None and (not np.isfinite(delta) or delta <= 0):
        raise ValueError("delta must be a positive finite perturbation distance")
    workers_count(workers)
    seeds = np.ascontiguousarray(seeds, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError("seeds must be finite (n,3) coordinates")
    lo, hi = (fields.mesh.lower, fields.mesh.upper) if bounds is None else bounds
    lo, hi = np.ascontiguousarray(lo, dtype=float), np.ascontiguousarray(hi, dtype=float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo,hi]).all() or
            np.any(lo >= hi) or np.any(lo < fields.mesh.lower) or np.any(hi > fields.mesh.upper)):
        raise ValueError("bounds must be a nonempty box inside the physical mesh")
    if (not np.isfinite(step_fraction) or not 0 < step_fraction <= 1 or
            (step is not None and (not np.isfinite(step) or step <= 0)) or
            type(max_steps) is not int or not 0 <= max_steps <= np.iinfo(np.int64).max or
            np.isnan(max_length) or max_length < 0 or not np.isfinite(null_threshold) or
            null_threshold < 0 or type(twist) is not bool):
        raise ValueError("invalid integration controls")
    scale = max(float(np.max(np.abs([lo,hi]))), float(np.max(hi-lo)))
    minimum_tolerance = 32*np.finfo(float).eps*scale
    tolerance = (max(minimum_tolerance, float(np.min(hi-lo))*1.e-10)
                 if boundary_tolerance is None else boundary_tolerance)
    if (not np.isfinite(tolerance) or tolerance < minimum_tolerance or
            tolerance >= np.min(hi-lo)*.01):
        raise ValueError("boundary_tolerance must be representable and small relative to the box")
    if local_radius is not None and (not np.isfinite(local_radius) or local_radius <= 100*tolerance):
        raise ValueError("local_radius must be positive and resolved by boundary_tolerance")
    if normalization not in ("mapping", "flux"):
        raise ValueError("normalization must be 'mapping' or 'flux'")
    return seeds, lo, hi, tolerance


def _squashing(vectors, scales, magnetic, normals, seed_strength, normalization):
    """Evaluate the two endpoint maps without subtracting large Gram products."""
    count = len(vectors)
    logs = np.full(count, np.nan)
    for index in range(count):
        b = magnetic[index]
        norms = np.hypot(np.hypot(b[:,0], b[:,1]), b[:,2])
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            continue
        direction = b/norms[:,None]
        n = normals[index]
        bn = np.sum(direction*n, axis=1)
        if not np.isfinite(bn).all() or np.any(np.abs(bn) <= 64*np.finfo(float).eps):
            continue
        uv = vectors[index].copy()
        uv -= (np.sum(uv*n[:,None,:], axis=2)/bn[:,None])[...,None]*direction[:,None,:]
        amplitudes = np.max(np.abs(uv), axis=(1,2))
        if not np.isfinite(amplitudes).all() or np.any(amplitudes == 0):
            continue
        uv /= amplitudes[:,None,None]
        u0,v0 = uv[0]
        u1,v1 = uv[1]
        matrix = np.outer(v1,u0)-np.outer(u1,v0)
        numerator = np.sum(matrix*matrix)
        if numerator <= 0 or not np.isfinite(numerator):
            continue
        if normalization == "mapping":
            cross = np.array([np.cross(u0,v0), np.cross(u1,v1)])
            area = np.hypot(np.hypot(cross[:,0], cross[:,1]), cross[:,2])
            if np.any(area == 0) or not np.isfinite(area).all():
                continue
            logq = np.log(numerator)-np.log(area).sum()
        else:
            if not np.isfinite(seed_strength[index]) or seed_strength[index] <= 0:
                continue
            logq = (np.log(numerator)+2*(scales[index].sum()+np.log(amplitudes).sum())+
                    np.log(np.abs(bn)).sum()+np.log(norms).sum()-2*np.log(seed_strength[index]))
        logs[index] = logq/np.log(10.)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.power(10., logs), logs


def _trace_arrays(fields, gradient, companion, seeds, lo, hi, tolerance, *, step_fraction,
                  step, max_steps, max_length, null_threshold, local_radius, workers, executor, centers=None):
    from ._kernels.connectivity import trace_halves
    n = len(seeds)
    positions = np.full((n,2,3), np.nan)
    vectors = np.full((n,2,2,3), np.nan)
    scales = np.zeros((n,2))
    magnetic = np.full((n,2,3), np.nan)
    normals = np.zeros((n,2,3))
    lengths, twists = np.zeros((n,2)), np.zeros((n,2))
    steps, status, faces = (np.zeros((n,2), dtype=np.int64) for _ in range(3))
    def run(first, last):
        s = slice(first,last)
        trace_halves(fields, gradient, companion, seeds[s], seeds[s] if centers is None else centers[s], lo, hi,
                     step_fraction, np.inf if step is None else step, max_steps,
                     max_length, null_threshold, tolerance, local_radius or 0.,
                     positions[s], vectors[s], scales[s], magnetic[s], normals[s],
                     lengths[s], twists[s], steps[s], status[s], faces[s])
    run_ranges(n, workers, run, executor)
    return positions, vectors, scales, magnetic, normals, lengths, twists, steps, status, faces


def _sample_seeds(fields, seeds):
    query = np.minimum(np.maximum(seeds, fields.mesh.lower), np.nextafter(fields.mesh.upper, fields.mesh.lower))
    return sample(fields, query)


def _stencil(fields, seeds, lo, hi, tolerance, delta, radius):
    magnetic, owners, valid = _sample_seeds(fields, seeds)
    offsets = np.zeros((len(seeds),4,3))
    distances = np.ones(len(seeds))
    seed_normal_field = np.full(len(seeds), np.nan)
    for row, seed in enumerate(seeds):
        b = magnetic[row]
        norm = np.hypot(np.hypot(b[0],b[1]),b[2])
        if not valid[row] or not np.isfinite(norm) or norm == 0 or np.any(seed < lo) or np.any(seed > hi):
            valid[row] = False
            continue
        bhat = b/norm
        touching = np.flatnonzero(np.minimum(seed-lo,hi-seed) <= tolerance)
        if len(touching) > 1:
            valid[row] = False
            continue
        if len(touching):
            axes = [axis for axis in range(3) if axis != touching[0]]
            u,v = np.eye(3)[axes]
        else:
            u,v = _transverse_basis(bhat)
        distance = float(np.min(fields.mesh.spacing[owners[row]]))*.001 if delta is None else delta
        for basis in (u,v):
            active = np.abs(basis) > 0
            if active.any():
                distance = min(distance, .25*float(np.min(np.minimum(seed-lo,hi-seed)[active]/np.abs(basis[active]))))
        if radius is not None:
            distance = min(distance,radius*.01)
        if distance <= 100*tolerance:
            valid[row] = False
            continue
        distances[row] = distance
        offsets[row] = distance*np.array([u,-u,v,-v])
        seed_normal_field[row] = abs(np.dot(b,np.cross(u,v)))
    return np.ascontiguousarray((seeds[:,None,:]+offsets).reshape(-1,3)), distances, seed_normal_field, valid


def _batch(fields, gradient, companion, seeds, lo, hi, tolerance, *, normalization, method, delta,
           compute_q=True, **controls):
    positions, vectors, scales, magnetic, normals, lengths, twists, steps, status, faces = _trace_arrays(
        fields, gradient, companion, seeds, lo, hi, tolerance, **controls)
    complete = np.all(np.isin(status, (3,12)), axis=1)
    regular = np.all(np.isin(status, (1,2,3,12)), axis=1)
    total_twist = None if companion is None else twists.sum(axis=1)
    if total_twist is not None:
        total_twist[~regular] = np.nan
    if not compute_q:
        valid = complete & np.isfinite(total_twist)
        return QSLResult(seeds.copy(),None,None,None,None,total_twist,lengths.sum(axis=1),
                         positions,magnetic,faces,status,steps,complete,valid,normalization,
                         controls["local_radius"],"twist-only",np.zeros(len(seeds),dtype=bool))
    stencil_valid = np.ones(len(seeds), dtype=bool)
    if method == "finite-difference":
        launches, distances, seed_strength, stencil_valid = _stencil(
            fields, seeds, lo, hi, tolerance, delta, controls["local_radius"])
        stencil_valid &= regular
        rows = np.flatnonzero(stencil_valid)
        vectors.fill(np.nan)
        scales.fill(0.)
        if len(rows):
            neighbors = _trace_arrays(fields, None, None, launches.reshape(-1,4,3)[rows].reshape(-1,3),
                lo, hi, tolerance, centers=np.repeat(seeds[rows],4,axis=0), **controls)
            ends = neighbors[0].reshape(len(rows),4,2,3)
            vectors[rows,:,0,:] = (ends[:,0]-ends[:,1])/(2*distances[rows,None,None])
            vectors[rows,:,1,:] = (ends[:,2]-ends[:,3])/(2*distances[rows,None,None])
            stencil_valid[rows] &= np.all(neighbors[8].reshape(-1,4,2) == status[rows,None,:], axis=(1,2))
            stencil_valid[rows] &= np.all(neighbors[9].reshape(-1,4,2) == faces[rows,None,:], axis=(1,2))
    elif normalization == "flux":
        bseed, _, _ = _sample_seeds(fields, seeds)
        seed_strength = np.hypot(np.hypot(bseed[:,0], bseed[:,1]), bseed[:,2])
    else:
        seed_strength = None
    q, logq = _squashing(vectors, scales, magnetic, normals, seed_strength, normalization)
    norms = np.hypot(np.hypot(magnetic[...,0], magnetic[...,1]), magnetic[...,2])
    with np.errstate(invalid="ignore", divide="ignore"):
        perpendicular = magnetic/norms[...,None]
    qp, logqp = _squashing(vectors, scales, magnetic, perpendicular, seed_strength, normalization)
    valid = complete & stencil_valid & np.all(np.isin(faces, (1,2,3,4,5,6,9)), axis=1) & ~np.isnan(logq)
    q[~valid], logq[~valid] = np.nan, np.nan
    qp[~regular | ~stencil_valid], logqp[~regular | ~stencil_valid] = np.nan, np.nan
    return QSLResult(seeds.copy(), q, logq, qp, logqp, total_twist, lengths.sum(axis=1),
                     positions, magnetic, faces, status, steps, complete, valid,
                     normalization, controls["local_radius"], method, stencil_valid)


def _iter_diagnostics(fields, seeds, *, bounds=None, step_fraction=.25, step=None,
             max_steps=10000, max_length=np.inf, null_threshold=0.,
             boundary_tolerance=None, local_radius=None, normalization="mapping",
             method="finite-difference", delta=None, twist=True, curl_field=None,
             workers=1, seed_batch=256, memory_limit=None, compute_q=True):
    """Yield owned squashing/twist results from completed Cartesian 3D Fields.

    RK4 steps are capped by step_fraction times local AMR spacing (and step,
    when supplied). Surface events are localized by bracketed reintegration.
    Input coverage is never enlarged. bounds explicitly selects mapping surfaces;
    local_radius replaces distant surfaces by a sphere centered on each seed.

    Finite differences trace four neighboring seeds and differentiate the actual
    endpoint map. The variational method uses centered gradients of unit nodes.
    'mapping' normalizes by transported areas; 'flux' uses the divergence-free
    magnetic-flux identity used by FastQSL. Neither is a separatrix detector.
    """
    _require_support(fields,compute_q=compute_q,method=method,twist=twist,curl_field=curl_field)
    seeds, lo, hi, tolerance = _controls(fields, seeds, bounds, step_fraction, step,
        max_steps, max_length, null_threshold, boundary_tolerance, local_radius,
        normalization, workers, twist, method, delta, compute_q)
    if type(seed_batch) is not int or seed_batch < 1:
        raise ValueError("seed_batch must be a positive integer")
    reserve = seeds.nbytes + min(seed_batch,len(seeds))*(8192 if compute_q else 2048) + workers*8192
    limit = remaining(memory_limit, reserve)
    if not len(seeds):
        return
    companion = _resolve_curl(fields, curl_field, twist, limit)
    gradient = (_unit_gradient(fields, remaining(limit, 0 if companion is None else companion.nbytes))
                if compute_q and method == "variational" else None)
    inputs = [value for value in (fields, gradient, companion) if value is not None]
    required = fields.mesh.nbytes + array_bytes(array for value in inputs
        for array in (value.values, value.leaf_ids, value.slot_of_leaf)) + reserve
    admit(required, memory_limit, "QSL batches")
    with worker_context(workers) as executor:
        for start in range(0,len(seeds),seed_batch):
            fields._check()
            yield _batch(fields, gradient, companion, seeds[start:start+seed_batch], lo, hi, tolerance,
                         step_fraction=step_fraction, step=step, max_steps=max_steps,
                         max_length=max_length, null_threshold=null_threshold, local_radius=local_radius,
                         normalization=normalization, method=method, delta=delta, workers=workers,
                         executor=executor,compute_q=compute_q)


def _quantities(quantities):
    names = (quantities,) if isinstance(quantities,str) else tuple(quantities)
    if not names or len(set(names)) != len(names) or any(name not in ("q","twist") for name in names):
        raise ValueError("quantities must select q, twist, or both without duplicates")
    return names


def iter_line_diagnostics(fields, seeds, *, quantities=("q","twist"), **controls):
    """Compute only requested diagnostics; twist-only skips Q transport/stencils.

    Parameters
    ----------
    fields : Fields
        Exactly three continuous vector components with common units and valid support.
    seeds : array-like
        Seed positions with shape (n, 3).
    quantities : sequence of str
        q, twist, or both; twist-only skips Q stencils/transport.
    **controls : object
        Controls from [qsl][simesh.qsl], except quantities selects diagnostics;
        backend/schedule are not accepted.

    Returns
    -------
    iterator of QSLResult
        Owned requested diagnostics and per-branch endpoints/statuses; QSLResult defines
        completion and validity.
    """
    names = _quantities(quantities)
    if "twist" in controls or "compute_q" in controls:
        raise ValueError("use quantities to select diagnostics")
    return _iter_diagnostics(fields,seeds,twist="twist" in names,compute_q="q" in names,**controls)


def iter_qsl(fields, seeds, *, bounds=None, step_fraction=.25, step=None,
             max_steps=10000, max_length=np.inf, null_threshold=0.,
             boundary_tolerance=None, local_radius=None, normalization="mapping",
             method="finite-difference", delta=None, twist=True, curl_field=None,
             workers=1, seed_batch=256, memory_limit=None):
    """Yield QSL batches, with optional twist. Existing controls are unchanged.

    Parameters
    ----------
    fields : Fields
        Exactly three continuous vector components with common units and valid support.
    seeds : array-like
        Seed positions with shape (n, 3).
    bounds : array-like, optional
        Lower and upper physical bounds; defaults to the original domain.
    step_fraction : float
        Positive local step fraction; its role depends on the selected integration method.
    step : float, optional
        Explicit positive coordinate step, constrained by the local step_fraction.
    max_steps : int
        Maximum accepted integration steps per branch.
    max_length : float
        Maximum integrated coordinate length per branch.
    null_threshold : float
        Vector norm at or below which tracing reports a null field.
    boundary_tolerance : float, optional
        Positive physical distance for endpoint localization; default is mesh/step
        dependent.
    local_radius : float, optional
        Positive radius for local sphere/box mapping.
    normalization : str
        mapping uses mapped area; flux uses the magnetic-flux relation. These definitions
        can differ.
    method : str
        finite-difference uses neighbor-seed footpoints; variational uses unit-vector
        gradients and needs two halo layers.
    delta : float, optional
        Positive neighbor-seed perturbation distance, only for finite-difference Q.
    twist : bool
        Compute twist; automatic curl preparation requires two primary halo layers.
    curl_field : Fields, optional
        Matching raw curl result with valid interpolation support and derivation identity.
    workers : int
        Number of workers over disjoint ranges.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    iterator of QSLResult
        Owned requested diagnostics and per-branch endpoints/statuses; QSLResult defines
        completion and validity.

    Notes
    -----
    The variational method requires two primary halo layers. With automatic curl, twist also requires two; a supplied matching curl needs interpolation support.
    """
    return _iter_diagnostics(fields,seeds,bounds=bounds,step_fraction=step_fraction,step=step,
        max_steps=max_steps,max_length=max_length,null_threshold=null_threshold,
        boundary_tolerance=boundary_tolerance,local_radius=local_radius,normalization=normalization,
        method=method,delta=delta,twist=twist,curl_field=curl_field,workers=workers,
        seed_batch=seed_batch,memory_limit=memory_limit)


def qsl(fields, seeds, *, bounds=None, step_fraction=.25, step=None,
        max_steps=10000, max_length=np.inf, null_threshold=0.,
        boundary_tolerance=None, local_radius=None, normalization="mapping",
        method="finite-difference", delta=None, twist=True, curl_field=None,
        workers=1, seed_batch=256, memory_limit=None):
    """Collect QSL batches; use iter_qsl for outputs too large to retain.

    Parameters
    ----------
    fields : Fields
        Exactly three continuous vector components with common units and valid support.
    seeds : array-like
        Seed positions with shape (n, 3).
    bounds : array-like, optional
        Lower and upper physical bounds; defaults to the original domain.
    step_fraction : float
        Positive local step fraction; its role depends on the selected integration method.
    step : float, optional
        Explicit positive coordinate step, constrained by the local step_fraction.
    max_steps : int
        Maximum accepted integration steps per branch.
    max_length : float
        Maximum integrated coordinate length per branch.
    null_threshold : float
        Vector norm at or below which tracing reports a null field.
    boundary_tolerance : float, optional
        Positive physical distance for endpoint localization; default is mesh/step
        dependent.
    local_radius : float, optional
        Positive radius for local sphere/box mapping.
    normalization : str
        mapping uses mapped area; flux uses the magnetic-flux relation. These definitions
        can differ.
    method : str
        finite-difference uses neighbor-seed footpoints; variational uses unit-vector
        gradients and needs two halo layers.
    delta : float, optional
        Positive neighbor-seed perturbation distance, only for finite-difference Q.
    twist : bool
        Compute twist; automatic curl preparation requires two primary halo layers.
    curl_field : Fields, optional
        Matching raw curl result with valid interpolation support and derivation identity.
    workers : int
        Number of workers over disjoint ranges.
    seed_batch : int
        Maximum seeds in a computation/output batch; input fields remain resident.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    QSLResult
        Owned requested diagnostics and per-branch endpoints/statuses; QSLResult defines
        completion and validity.

    Notes
    -----
    The variational method requires two primary halo layers. With automatic curl, twist also requires two; a supplied matching curl needs interpolation support.
    """
    if type(twist) is not bool:
        raise ValueError("twist must be boolean")
    return line_diagnostics(fields,seeds,quantities=("q","twist") if twist else ("q",),
        bounds=bounds, step_fraction=step_fraction, step=step,
        max_steps=max_steps, max_length=max_length, null_threshold=null_threshold,
        boundary_tolerance=boundary_tolerance, local_radius=local_radius, normalization=normalization,
        method=method, delta=delta, curl_field=curl_field, workers=workers,
        seed_batch=seed_batch, memory_limit=memory_limit)


def line_diagnostics(fields, seeds, *, quantities=("q","twist"), memory_limit=None, **controls):
    """Collect identified diagnostic arrays without retaining integration paths.

    Parameters
    ----------
    fields : Fields
        Exactly three continuous vector components with common units and valid support.
    seeds : array-like
        Seed positions with shape (n, 3).
    quantities : sequence of str
        q, twist, or both; twist-only skips Q stencils/transport.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    **controls : object
        Controls from [qsl][simesh.qsl], except quantities selects diagnostics;
        backend/schedule are not accepted.

    Returns
    -------
    QSLResult
        Owned requested diagnostics and per-branch endpoints/statuses; QSLResult defines
        completion and validity.
    """
    names = _quantities(quantities)
    _require_support(fields,compute_q="q" in names,twist="twist" in names,
                     method=controls.get("method","finite-difference"),curl_field=controls.get("curl_field"))
    seeds = np.ascontiguousarray(seeds,dtype=float)
    batches = iter_line_diagnostics(fields,seeds,quantities=names,
        memory_limit=remaining(memory_limit,len(seeds)*512),**controls)
    first = next(batches, None)
    if first is None:
        empty = np.empty(0)
        return QSLResult(seeds.copy(),*(empty.copy() if "q" in names else None for _ in range(4)),
            empty.copy() if "twist" in names else None, empty.copy(),
            np.empty((0,2,3)), np.empty((0,2,3)), np.empty((0,2),np.int64),
            np.empty((0,2),np.int64), np.empty((0,2),np.int64), np.empty(0,bool), np.empty(0,bool),
            controls.get("normalization","mapping"),controls.get("local_radius"),
            controls.get("method","finite-difference") if "q" in names else "twist-only", np.empty(0,bool))
    try:
        arrays = {name: np.empty((len(seeds), *value.shape[1:]), dtype=value.dtype)
                  for name,value in vars(first).items() if isinstance(value,np.ndarray)}
        offset, result = 0, first
        while result is not None:
            stop = offset+len(result.seeds)
            for name, target in arrays.items():
                target[offset:stop] = getattr(result,name)
            offset = stop
            result = next(batches,None)
    finally:
        batches.close()
    return QSLResult(**{**vars(first), **arrays})
