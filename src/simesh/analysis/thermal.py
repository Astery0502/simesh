"""Explicit coronal H/He thermodynamics and historical AIA 171 synthesis.

The two reconstruction orders act on the same prepared thermodynamic nodes.
No instrumental PSF, pixel-area integration or observation-date calibration is
implicit. Temperature must be supplied; stored density never determines it.
"""

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ._aia171_table import LOG_T, RESPONSE, UPSTREAM_COMMIT
from .fields import FieldDefinition, PreparedFields
from .mesh import frozen_array
from .sampling import sample
from .los import LOSResult, LOSStatus, integrate_los
from .slices import Plane

PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_ERG_K = 1.380649e-16
_RESPONSE_GRID = frozen_array(LOG_T, float)
_LOG_RESPONSE = frozen_array(np.log10(RESPONSE), float)
_RESPONSE_SLOPES = frozen_array(np.diff(_LOG_RESPONSE)/np.diff(_RESPONSE_GRID), float)


@dataclass(frozen=True)
class ThermalLOSResult(LOSResult):
    model: str
    temperature_label: str
    density_unit_g_cm3: float
    length_unit_cm: float

    @property
    def depth_cm(self):
        return self.depth*self.length_unit_cm


@dataclass(frozen=True)
class CoronalComposition:
    """Fully ionized H/He, ignoring electron mass; abundance is n_He/n_H."""
    helium_abundance: float = .1

    def __post_init__(self):
        if not np.isfinite(self.helium_abundance) or self.helium_abundance < 0:
            raise ValueError("helium abundance must be finite and nonnegative")

    def number_density(self, mass_density_cgs, *, convention="electron"):
        rho = np.asarray(mass_density_cgs, dtype=float)
        if not np.isfinite(rho).all() or np.any(rho < 0):
            raise ValueError("mass density must be finite and nonnegative")
        if convention not in ("electron", "amrvac-hydrogen"):
            raise ValueError("density convention must be electron or amrvac-hydrogen")
        h = self.helium_abundance
        nh = rho / ((1 + 4*h)*PROTON_MASS_G)
        return nh*(1 + 2*h) if convention == "electron" else nh

    def temperature(self, mass_density_cgs, thermal_pressure_cgs):
        """p = (2+3a) n_H k_B T. Pressure is thermal, not total energy."""
        nh = self.number_density(mass_density_cgs, convention="amrvac-hydrogen")
        p = np.asarray(thermal_pressure_cgs, dtype=float)
        if not np.isfinite(p).all() or np.any(p <= 0) or np.any(nh <= 0):
            raise ValueError("EOS temperature requires positive density and thermal pressure")
        return p / ((2+3*self.helium_abundance)*nh*BOLTZMANN_ERG_K)


@dataclass(frozen=True)
class AIA171:
    """Historical response, log10(T)-log10(R) interpolation, zero outside table.

    R has DN cm^5 s^-1 pixel^-1 under the selected emission-measure convention.
    electron is the explicit physical baseline; amrvac-hydrogen reproduces the
    upstream eq_state_units=True H/He density factor. Neither fixes the table's
    undocumented original calibration settings.
    """
    density_convention: str = "electron"
    composition: CoronalComposition = CoronalComposition()

    def __post_init__(self):
        if self.density_convention not in ("electron", "amrvac-hydrogen"):
            raise ValueError("unknown emission-measure density convention")

    @property
    def identity(self):
        return f"amrvac-{UPSTREAM_COMMIT}-171-{self.density_convention}-He{self.composition.helium_abundance:g}"

    def response(self, temperature_k):
        t = np.asarray(temperature_k, dtype=float)
        if not np.isfinite(t).all() or np.any(t <= 0):
            raise ValueError("temperature must be finite and positive in kelvin")
        logt = np.log10(t)
        log_response = np.interp(logt, LOG_T, np.log10(RESPONSE))
        return np.where((logt >= LOG_T[0]) & (logt <= LOG_T[-1]), 10.**log_response, 0.)

    def emissivity(self, mass_density_cgs, temperature_k):
        n = self.composition.number_density(mass_density_cgs, convention=self.density_convention)
        return self.from_number_density(n, temperature_k)

    def from_number_density(self, number_density_cm3, temperature_k):
        n = np.asarray(number_density_cm3, dtype=float)
        if not np.isfinite(n).all() or np.any(n < 0):
            raise ValueError("number density must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            value = n*n*self.response(temperature_k)
        if not np.isfinite(value).all():
            raise ValueError("unrepresentable thermal emissivity")
        return value


def thermal_fields(density, temperature, *, density_unit_g_cm3, model=AIA171(),
                   density_component=0, temperature_component=0,
                   temperature_label, budget_bytes=2*1024**3):
    """Detach (emission-measure number density, T) from prepared primary data.

    density uses an explicit cgs multiplier. External temperature is either a
    positive scalar in kelvin (explicit isothermal model), or a PreparedFields
    scalar with units 'K', the identical mesh and matching requested coverage.
    Use the normal source preparation API for external native temperature arrays;
    this preserves the same two-layer boundary/transfer semantics as density.
    temperature_label must identify measured, isothermal or manufactured input.
    """
    if not isinstance(density, PreparedFields) or density.halo != 2:
        raise ValueError("density requires an owned two-layer primary product")
    if not isinstance(temperature_label, str) or not temperature_label.strip():
        raise ValueError("temperature provenance label is required")
    if not np.isfinite(density_unit_g_cm3) or density_unit_g_cm3 <= 0:
        raise ValueError("density unit must be a positive finite cgs multiplier")
    if type(density_component) is not int or not 0 <= density_component < len(density.fields):
        raise ValueError("invalid density component")
    # Require packed products, including a permuted leaf order.
    density.interior()
    tproduct = isinstance(temperature, PreparedFields)
    if tproduct:
        if (temperature.mesh is not density.mesh or temperature.halo != density.halo or
                type(temperature_component) is not int or
                not 0 <= temperature_component < len(temperature.fields) or
                temperature.fields[temperature_component].units != "K" or
                np.any(temperature.slot_of_leaf[density.leaf_ids] < 0)):
            raise ValueError("temperature must be kelvin on the same mesh and prepared coverage")
    else:
        try:
            scalar_temperature = float(temperature)
        except (TypeError, ValueError):
            raise ValueError("external temperature must be a positive kelvin scalar or prepared field") from None
        if np.ndim(temperature) != 0 or not np.isfinite(scalar_temperature) or scalar_temperature <= 0:
            raise ValueError("external temperature must be a positive kelvin scalar or prepared field")
        temperature = scalar_temperature
    shape = (*density.values.shape[:-1], 2)
    extra = 0 if not tproduct or temperature is density else temperature.nbytes
    block_scratch = 128*int(np.prod(density.values.shape[1:4]))
    total = density.nbytes + extra + density.mesh.nbytes + 8*int(np.prod(shape)) + block_scratch
    if total > budget_bytes:
        raise MemoryError(f"thermal fields need {total} controlled bytes")
    values = np.empty(shape, dtype=float)
    for i, leaf in enumerate(density.leaf_ids):
        rho = density.values[i, ..., density_component]*density_unit_g_cm3
        values[i, ..., 0] = model.composition.number_density(rho, convention=model.density_convention)
        t = temperature.values[temperature.slot_of_leaf[leaf], ..., temperature_component] if tproduct else temperature
        model.response(t)  # Validate physical inputs before publication.
        values[i, ..., 1] = t
    values.flags.writeable = False
    return PreparedFields(density.mesh, values, density.leaf_ids, density.slot_of_leaf,
        (FieldDefinition("emission_measure_density", "cm^-3", "prepared-node"),
         FieldDefinition("temperature", "K", "prepared-node")), 2,
        density.strategy+"/thermal-nodes", (density.source, model.identity, temperature_label),
        {"model": model.identity, "temperature": temperature_label,
         "density_unit_g_cm3": float(density_unit_g_cm3), "controlled_upper_bytes": total})


def emissivity_fields(thermodynamics, *, model=AIA171(), budget_bytes=2*1024**3):
    """Compute n^2 R(T) at every prepared node, then retain a scalar interpolant.

    This is pointwise response AFTER primary halo preparation. It is explicitly
    different from applying AMR prolongation/restriction to interior emissivity.
    """
    _check_thermal(thermodynamics, model)
    thermodynamics.interior()
    count = int(np.prod(thermodynamics.values.shape[:-1]))
    required = thermodynamics.nbytes+thermodynamics.mesh.nbytes+8*count+128*int(np.prod(thermodynamics.values.shape[1:4]))
    if required > budget_bytes:
        raise MemoryError(f"emissivity needs {required} controlled bytes")
    values = np.empty((*thermodynamics.values.shape[:-1], 1))
    for i in range(len(values)):
        values[i, ..., 0] = model.from_number_density(thermodynamics.values[i, ..., 0], thermodynamics.values[i, ..., 1])
    values.flags.writeable = False
    return replace(thermodynamics, values=values,
        fields=(FieldDefinition("aia171_emissivity", "DN s^-1 pixel^-1 cm^-1", "prepared-node"),),
        strategy=thermodynamics.strategy+"/response-before-interpolation")


def _check_thermal(fields, model):
    if (not isinstance(fields, PreparedFields) or len(fields.fields) != 2 or
            type(fields.halo) is not int or fields.halo < 1 or
            tuple(f.units for f in fields.fields) != ("cm^-3", "K") or
            not isinstance(fields.source, tuple) or fields.source[1] != model.identity):
        raise ValueError("thermal fields and response model must have matching physical identity")
    if (not isinstance(fields.values,np.ndarray) or fields.values.dtype != np.float64 or
            fields.values.ndim != 5 or fields.values.shape[-1] != 2 or
            fields.values.shape[1:4] != tuple(n+2*fields.halo for n in fields.mesh.block_shape) or
            fields.slot_of_leaf.shape != (fields.mesh.leaf_count,) or
            np.any(fields.slot_of_leaf < -1) or np.any(fields.slot_of_leaf >= len(fields.values))):
        raise ValueError("thermal backing and slot directory must match the declared mesh and two components")


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
    """Split at thermodynamic interpolation knots, then composite two-point Gauss."""
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


def integrate_thermal_los(thermodynamics, plane, direction, *, length_unit_cm,
                          model=AIA171(), order="thermodynamics-first", subdivisions=4,
                          near=0., far=np.inf, max_samples=1000000,
                          workers=1, implementation="native", budget_bytes=2*1024**3):
    """Full box LOS of the explicitly selected nonlinear thermal reconstruction.

    thermodynamics-first samples n,T then evaluates n^2 R(T). Composite Gauss2
    is a convergent approximation, NEVER an exact nonlinear-response integral.
    emissivity-first computes prepared-node emissivity then uses scalar gauss2.
    native reuses AMR-tree interval ownership and compiled interpolation, with
    independent rays dispatched to 1--4 GIL-free workers. reference retains the
    original Python/all-leaf oracle and requires one worker. Both keep the same
    nonlinear reconstruction and quadrature; per-pixel limits fail closed.
    """
    _check_thermal(thermodynamics, model)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer in 1..4")
    if implementation not in ("native", "reference"):
        raise ValueError("implementation must be native or reference")
    if implementation == "reference" and workers != 1:
        raise ValueError("reference implementation requires one worker")
    if implementation == "native" and type(model) is not AIA171:
        raise ValueError("native response requires AIA171; use reference for customized models")
    if not np.isfinite(length_unit_cm) or length_unit_cm <= 0:
        raise ValueError("length unit must be a positive finite cm multiplier")
    if order not in ("thermodynamics-first", "emissivity-first"):
        raise ValueError("unknown thermal reconstruction order")
    if order == "emissivity-first":
        emissivity = emissivity_fields(thermodynamics, model=model, budget_bytes=budget_bytes)
        result = integrate_los(emissivity, plane, direction, near=near, far=far, max_samples=max_samples,
                               workers=workers, budget_bytes=budget_bytes-thermodynamics.nbytes)
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
    if required > budget_bytes:
        raise MemoryError(f"thermal LOS needs {required} controlled bytes")
    near = np.broadcast_to(np.asarray(near, dtype=float), plane.shape)
    far = np.broadcast_to(np.asarray(far, dtype=float), plane.shape)
    if not np.isfinite(near).all() or np.any(near < 0) or np.isnan(far).any() or np.any(far < near):
        raise ValueError("ordered nonnegative near/far required")
    values, entry, exit = (np.zeros(plane.shape) for _ in range(3))
    status = np.full(plane.shape, int(LOSStatus.EMPTY), dtype=np.int64)
    samples, misses = (np.zeros(plane.shape, dtype=np.int64) for _ in range(2))
    if implementation == "native":
        _native_los(thermodynamics,plane,direction,near,far,subdivisions,max_samples,
                    workers,values,entry,exit,status,samples)
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
                workers,values,entry,exit,status,samples):
    from simesh.utils.lib.analysis.native import initialize_rays
    from simesh.utils.lib.analysis.thermal_rays import integrate_ready
    mesh = fields.mesh
    nx,ny = plane.shape
    origins = (plane.origin+((np.arange(nx)+.5)/nx)[:,None,None]*plane.u+
               ((np.arange(ny)+.5)/ny)[None,:,None]*plane.v).reshape(-1,3)
    starts,ends = entry.ravel(),exit.ravel()
    flags = status.ravel()
    flags.fill(0)
    initialize_rays(mesh.lower,mesh.upper,origins,direction,
        np.ascontiguousarray(near.ravel()),np.ascontiguousarray(far.ravel()),starts,ends,flags)
    output,counts = values.ravel(),samples.ravel()
    output[flags >= LOSStatus.MISSING_COVERAGE] = np.nan
    def run(span):
        integrate_ready(mesh.roots,mesh.children,mesh.node_leaves,mesh.node_lower,mesh.node_upper,
            mesh.bounds,mesh.spacing,fields.slot_of_leaf,fields.values,fields.halo,
            origins[span],direction,starts[span],ends[span],subdivisions,max_samples,
            _RESPONSE_GRID,_LOG_RESPONSE,_RESPONSE_SLOPES,output[span],flags[span],counts[span])
    if workers == 1:
        run(slice(None))
    else:
        # Bounded, coherent row ranges. No numeric cache or provider is mutated.
        edges = np.linspace(0,nx*ny,min(workers*4,nx*ny)+1,dtype=int)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run,slice(a,b)) for a,b in zip(edges[:-1],edges[1:])]
            for future in futures:
                future.result()


def _physical_result(result, length_unit_cm, quadrature, thermodynamics, model):
    with np.errstate(over="ignore", invalid="ignore"):
        values = result.values*length_unit_cm
    status = result.status.copy()
    bad = result.valid & ~np.isfinite(values)
    status[bad] = LOSStatus.UNREPRESENTABLE_INTEGRAL
    values[bad] = np.nan
    result = replace(result, values=values, status=status,
                     scalar_units="DN s^-1 pixel^-1", quadrature=quadrature)
    return ThermalLOSResult(**vars(result), model=model.identity,
        temperature_label=thermodynamics.source[2],
        density_unit_g_cm3=thermodynamics.preparation_stats["density_unit_g_cm3"],
        length_unit_cm=float(length_unit_cm))
