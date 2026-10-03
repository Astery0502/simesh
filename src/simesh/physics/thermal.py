"""Thermal line-of-sight integration and historical thermal API aliases.

Response models and prepared thermal fields are defined in emission and
thermodynamics. This module composes them with sampling and ray integration.
"""

from dataclasses import dataclass, replace
from contextlib import nullcontext
import numpy as np

from .composition import (CoronalComposition as CoronalComposition,
                          PROTON_MASS_G as PROTON_MASS_G, BOLTZMANN_ERG_K as BOLTZMANN_ERG_K)
from .emission import AIA171, EUV
from .thermodynamics import thermal_fields as thermal_fields, emissivity_fields, _check_thermal
from .._validation import frozen_array, admit, remaining, workers_count
from .._execution import native_dispatch, worker_context, run_ranges
from ..operators.sampling import sample
from ..operators.rays import LOSStatus, ray_segments, ray_nodes, _run_ray_batches
from ..projection import LOSResult, integrate_los
from ..spatial import Plane


@dataclass(frozen=True)
class ThermalLOSResult(LOSResult):
    """Raw thermal LOSResult with explicit model and physical length scale.

    Attributes
    ----------
    model : object
        Pinned response identity.
    temperature_label : str
        Temperature provenance.
    density_unit_g_cm3, length_unit_cm : float
        Physical conversion factors used for this product.
    """
    model: str
    temperature_label: str
    density_unit_g_cm3: float
    length_unit_cm: float

    @property
    def depth_cm(self):
        """Physical clipped depth in centimeters, using the recorded coordinate scale."""
        return self.depth*self.length_unit_cm


def _native_response(model):
    if type(model) not in (AIA171, EUV):
        raise ValueError("native response requires AIA171 or EUV; use reference for customized models")
    return model._table


def integrate_thermal_los(thermodynamics, plane, direction, *, length_unit_cm,
                          model=AIA171(), order="thermodynamics-first", subdivisions=4,
                          near=0., far=np.inf, max_samples=1000000,
                          workers=1, implementation="native", memory_limit=None,
                          backend="threadpool", schedule="dynamic"):
    """Full box LOS of the explicitly selected nonlinear thermal reconstruction.

    Parameters
    ----------
    thermodynamics : Fields
        Matching number-density/kelvin fields from thermal_fields, with valid
        interpolation support.
    plane : Plane
        Pixel-center sampling plane.
    direction : array-like
        Finite nonzero viewing direction (3,).
    length_unit_cm : float
        Centimeters per coordinate-length unit.
    model : AIA171 or EUV
        Response matching the thermal fields.
    order : str
        thermodynamics-first interpolates n,T before n²R(T); emissivity-first applies
        response at nodes before interpolation.
    subdivisions : int
        Composite Gauss2 subdivisions for nonlinear thermal response integration.
    near : float or array-like
        Nonnegative ray entry clipping in coordinate-length units.
    far : float or array-like
        Ray exit clipping, at least near; positive infinity is allowed.
    max_samples : int
        Per-ray sampling limit; reaching the limit is not a complete integral.
    workers : int
        Number of workers over disjoint ranges.
    implementation : str
        native for compiled integration; reference for the single-worker reference
        calculation.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    backend : str
        Execution backend: threadpool or explicitly built openmp.
    schedule : str
        Native scheduling policy: static or dynamic.

    Returns
    -------
    ThermalLOSResult
        Plane brightness and statuses with explicit thermal model and physical depth scale.

    Notes
    -----
    These orders define different reconstructions. Composite Gauss2 for nonlinear response is an approximation; check convergence and result status.
    """
    _check_thermal(thermodynamics, model)
    workers_count(workers)
    if implementation not in ("native", "reference"):
        raise ValueError("implementation must be native or reference")
    if implementation == "reference" and workers != 1:
        raise ValueError("reference implementation requires one worker")
    if implementation == "native":
        _native_response(model)
    if not np.isfinite(length_unit_cm) or length_unit_cm <= 0:
        raise ValueError("length unit must be a positive finite cm multiplier")
    if order not in ("thermodynamics-first", "emissivity-first"):
        raise ValueError("unknown thermal reconstruction order")
    if order == "emissivity-first":
        emissivity = emissivity_fields(thermodynamics, model=model, memory_limit=memory_limit)
        result = integrate_los(emissivity, plane, direction, near=near, far=far, max_samples=max_samples,
                               workers=workers, backend=backend, schedule=schedule,
                               memory_limit=remaining(memory_limit,thermodynamics.nbytes))
        return _physical_result(result, length_unit_cm, "prepared-node-emissivity/gauss2", thermodynamics, model)
    if not isinstance(plane, Plane) or type(subdivisions) is not int or subdivisions < 1:
        raise ValueError("thermal LOS requires a Plane and positive subdivisions")
    if (type(max_samples) is not int or not 1 <= max_samples <= np.iinfo(np.int64).max or
            subdivisions > np.iinfo(np.int64).max//2):
        raise ValueError("positive max_samples required")
    direction = np.array(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("finite nonzero LOS direction required")
    direction /= np.max(np.abs(direction))
    direction /= np.linalg.norm(direction)
    mesh = thermodynamics.mesh
    # One leaf's query arrays at a time, plus leaf intersection vectors and image.
    per_leaf = 2*subdivisions*(sum(mesh.block_shape)+1)
    required = (thermodynamics.nbytes+mesh.nbytes+mesh.leaf_count*128+
                int(np.prod(plane.shape))*128+per_leaf*384+workers*65536)
    admit(required,memory_limit,"thermal LOS")
    near = np.broadcast_to(np.asarray(near, dtype=float), plane.shape)
    far = np.broadcast_to(np.asarray(far, dtype=float), plane.shape)
    if not np.isfinite(near).all() or np.any(near < 0) or np.isnan(far).any() or np.any(far < near):
        raise ValueError("ordered nonnegative near/far required")
    values, entry, exit = (np.zeros(plane.shape) for _ in range(3))
    status = np.full(plane.shape, int(LOSStatus.EMPTY), dtype=np.int64)
    samples, misses = (np.zeros(plane.shape, dtype=np.int64) for _ in range(2))
    if implementation == "native":
        _native_los(thermodynamics,plane,direction,near,far,subdivisions,max_samples,
                    workers,values,entry,exit,status,samples,backend,schedule,model)
    for pixel in (np.ndindex(plane.shape) if implementation == "reference" else ()):
        origin = plane.origin+(pixel[0]+.5)/plane.shape[0]*plane.u+(pixel[1]+.5)/plane.shape[1]*plane.v
        leaves, first, last = ray_segments(mesh, origin, direction, near[pixel], far[pixel])
        if not len(leaves):
            continue
        entry[pixel], exit[pixel] = first[0], last[-1]
        status[pixel] = LOSStatus.COMPLETE
        if not np.allclose(first[1:], last[:-1], rtol=2e-13, atol=2e-13):
            status[pixel] = LOSStatus.GEOMETRY_FAILURE
        for leaf, lo, hi in zip(leaves, first, last):
            if status[pixel] != LOSStatus.COMPLETE:
                break
            if thermodynamics.slot_of_leaf[leaf] < 0:
                status[pixel] = LOSStatus.MISSING_COVERAGE
                break
            nodes, weights = ray_nodes(mesh, leaf, origin, direction, lo, hi, subdivisions)
            if samples[pixel]+len(nodes) > max_samples:
                status[pixel] = LOSStatus.SAMPLE_LIMIT
                break
            points = origin+nodes[:, None]*direction
            # A grazing corner can leave an interval only one ulp wide with no
            # representable interior point. Keep its full quadrature weight and
            # bind rounded coordinates to this interval's half-open owner.
            points = np.maximum(mesh.bounds[leaf,0], np.minimum(points,
                np.nextafter(mesh.bounds[leaf,1],mesh.bounds[leaf,0])))
            data, owners, valid = sample(thermodynamics, points)
            if not np.all(valid) or not np.all(owners == leaf):
                status[pixel] = LOSStatus.UNREPRESENTABLE_SAMPLE
                break
            try:
                epsilon = model.from_number_density(data[:, 0], data[:, 1])
            except ValueError:
                status[pixel] = LOSStatus.NONFINITE_SCALAR
                break
            samples[pixel] += len(nodes)
            values[pixel] += np.dot(epsilon, weights)
        if status[pixel] != LOSStatus.COMPLETE:
            values[pixel] = np.nan
    result = LOSResult(plane, frozen_array(direction, float), values, entry, exit, status,
                       samples, misses, "DN s^-1 pixel^-1 cm^-1 * coordinate-length",
                       f"thermodynamics-first/composite-gauss2/{subdivisions}")
    return _physical_result(result, length_unit_cm, result.quadrature, thermodynamics, model)


def _native_los(fields,plane,direction,near,far,subdivisions,max_samples,
                workers,values,entry,exit,status,samples,backend,schedule,model):
    from .._kernels.native import initialize_rays
    from .._kernels.thermal_rays import integrate_ready, openmp_build_info
    native, dispatch = native_dispatch(backend,schedule,build_info=openmp_build_info)
    mesh = fields.mesh
    grid, ordinates, slopes, mode = _native_response(model)
    nx,ny = plane.shape
    origins = (plane.origin+((np.arange(nx)+.5)/nx)[:,None,None]*plane.u+
               ((np.arange(ny)+.5)/ny)[None,:,None]*plane.v).reshape(-1,3)
    starts,ends = entry.ravel(),exit.ravel()
    flags = status.ravel()
    initialize_rays(mesh.lower,mesh.upper,origins,direction,
        np.ascontiguousarray(near.ravel()),np.ascontiguousarray(far.ravel()),starts,ends,flags)
    output,counts = values.ravel(),samples.ravel()
    output[flags >= LOSStatus.MISSING_COVERAGE] = np.nan
    backing = fields.values
    def run(first,last):
        span = slice(first,last)
        integrate_ready(mesh.roots,mesh.children,mesh.node_leaves,mesh.node_lower,mesh.node_upper,
            mesh.bounds,mesh.spacing,fields.slot_of_leaf,backing,fields.storage_halo,
            origins[span],direction,starts[span],ends[span],subdivisions,max_samples,
            grid,ordinates,slopes,output[span],flags[span],counts[span],
            workers if native else 1,dispatch,mode)
    tasks = workers*4 if dispatch else workers
    context = nullcontext(None) if native else worker_context(workers)
    with context as executor:
        run_ranges(nx*ny,1 if native else tasks,run,executor)


def _scale_thermal_values(values, status, valid, length_unit_cm):
    with np.errstate(over="ignore", invalid="ignore"):
        values = values*length_unit_cm
    status = status.copy()
    bad = valid & ~np.isfinite(values)
    status[bad] = LOSStatus.UNREPRESENTABLE_INTEGRAL
    values[bad] = np.nan
    return values,status


def _physical_result(result, length_unit_cm, quadrature, thermodynamics, model):
    values,status = _scale_thermal_values(result.values,result.status,result.valid,length_unit_cm)
    result = replace(result, values=values, status=status,
                     scalar_units="DN s^-1 pixel^-1", quadrature=quadrature)
    return ThermalLOSResult(**vars(result), model=model.identity,
        temperature_label=thermodynamics.source[2],
        density_unit_g_cm3=thermodynamics.preparation_stats["density_unit_g_cm3"],
        length_unit_cm=float(length_unit_cm))



def _integrate_thermal_rays(fields, rays, model, subdivisions, max_samples,
                            workers, ray_batch, memory_limit):
    """Bind a tabulated response and native fields to validated ray geometry."""
    from .._kernels.native import initialize_ray_set
    from .._kernels.thermal_rays import integrate_ray_set_ready

    grid, ordinates, slopes, mode = _native_response(model)
    mesh = fields.mesh
    count = len(rays.origins)
    admit(mesh.nbytes+fields.nbytes+rays.nbytes+count*96+min(count, ray_batch)*128+workers*65536,
          memory_limit, "ray-set thermal LOS")
    values, entry, exit = (np.zeros(count) for _ in range(3))
    status, samples, misses = (np.zeros(count, dtype=np.int64) for _ in range(3))
    backing = fields.values

    def consume(span, origins, directions, near, far):
        initialize_ray_set(mesh.lower, mesh.upper, origins, directions, near, far,
                           entry[span], exit[span], status[span])
        output = values[span]
        output[status[span] >= LOSStatus.MISSING_COVERAGE] = np.nan
        integrate_ray_set_ready(mesh.roots, mesh.children, mesh.node_leaves,
            mesh.node_lower, mesh.node_upper, mesh.bounds, mesh.spacing,
            fields.slot_of_leaf, backing, fields.storage_halo,
            origins, directions, entry[span], exit[span], subdivisions, max_samples,
            grid, ordinates, slopes, values[span], status[span], samples[span], mode)

    _run_ray_batches(rays, ray_batch, workers, consume)
    return values, entry, exit, status, samples, misses
