"""Explicit historical AIA171 response and H/He thermodynamics on ready fields."""
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from ._aia171_table import LOG_T, RESPONSE, UPSTREAM_COMMIT
from ..fields import FieldDefinition, Fields, require_fields, publish, require_continuous
from .._validation import frozen_array, admit, remaining
from ..operators.sampling import sample
from ..projection import LOSResult, LOSStatus, integrate_los
from ..slices import Plane
from ..field_ops import _layout

PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_ERG_K = 1.380649e-16
_RESPONSE_GRID = frozen_array(LOG_T, float)
_LOG_RESPONSE = frozen_array(np.log10(RESPONSE), float)
_RESPONSE_SLOPES = frozen_array(np.diff(_LOG_RESPONSE)/np.diff(_RESPONSE_GRID), float)

@dataclass(frozen=True)
class ThermalLOSResult(LOSResult):
    """Raw thermal LOSResult with explicit model and physical length scale.

    Attributes
    ----------
    model : object
        Historical response identity.
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


@dataclass(frozen=True)
class CoronalComposition:
    """Fully ionized H/He, ignoring electron mass; abundance is n_He/n_H."""
    helium_abundance: float = .1

    def __post_init__(self):
        if not np.isfinite(self.helium_abundance) or self.helium_abundance < 0:
            raise ValueError("helium abundance must be finite and nonnegative")

    def number_density(self, mass_density_cgs, *, convention="electron"):
        """Convert mass density in g/cm³ to electron or amrvac-hydrogen number density in cm^-3."""
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
        """Response, density-convention and composition identity used to check compatible thermal fields."""
        return f"amrvac-{UPSTREAM_COMMIT}-171-{self.density_convention}-He{self.composition.helium_abundance:g}"

    def response(self, temperature_k):
        """Interpolate the historical response in log temperature/response; zero outside the table."""
        t = np.asarray(temperature_k, dtype=float)
        if not np.isfinite(t).all() or np.any(t <= 0):
            raise ValueError("temperature must be finite and positive in kelvin")
        logt = np.log10(t)
        log_response = np.interp(logt, LOG_T, np.log10(RESPONSE))
        return np.where((logt >= LOG_T[0]) & (logt <= LOG_T[-1]), 10.**log_response, 0.)

    def emissivity(self, mass_density_cgs, temperature_k):
        """Evaluate emission from mass density in g/cm³ and positive kelvin temperature."""
        n = self.composition.number_density(mass_density_cgs, convention=self.density_convention)
        return self.from_number_density(n, temperature_k)

    def from_number_density(self, number_density_cm3, temperature_k):
        """Evaluate n² R(T) for number density in cm^-3 under this model convention."""
        n = np.asarray(number_density_cm3, dtype=float)
        if not np.isfinite(n).all() or np.any(n < 0):
            raise ValueError("number density must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            value = n*n*self.response(temperature_k)
        if not np.isfinite(value).all():
            raise ValueError("unrepresentable thermal emissivity")
        return value


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
    model : AIA171
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
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
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
                    workers,values,entry,exit,status,samples,backend,schedule)
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
                workers,values,entry,exit,status,samples,backend,schedule):
    from .._kernels.native import initialize_rays
    from .._kernels.thermal_rays import integrate_ready, openmp_build_info
    from .._execution import native_dispatch
    native, dispatch = native_dispatch(backend,schedule,build_info=openmp_build_info)
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
            mesh.bounds,mesh.spacing,fields.slot_of_leaf,fields.values,fields.storage_halo,
            origins[span],direction,starts[span],ends[span],subdivisions,max_samples,
            _RESPONSE_GRID,_LOG_RESPONSE,_RESPONSE_SLOPES,output[span],flags[span],counts[span],
            workers if native else 1,dispatch)
    if workers == 1 or native:
        run(slice(None))
    else:
        # Bounded, coherent row ranges. No numeric cache or provider is mutated.
        tasks = workers*4 if dispatch else workers
        edges = np.linspace(0,nx*ny,min(tasks,nx*ny)+1,dtype=int)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            try:
                for a,b in zip(edges[:-1],edges[1:]):
                    futures.append(executor.submit(run,slice(a,b)))
                for future in futures:
                    future.result()
            finally:
                wait(futures)


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



def thermal_fields(density,temperature,*,density_unit_g_cm3,model=AIA171(),density_component=0,
                   temperature_component=0,temperature_label,memory_limit=None):
    """Detach number-density and kelvin nodes using the common valid input support.

    Parameters
    ----------
    density : Fields
        Mass-density values with explicit physical conversion.
    temperature : float or Fields
        Positive kelvin scalar or kelvin field covering density on the same Mesh.
    density_unit_g_cm3 : float
        Grams per cubic centimeter per stored mass-density value.
    model : AIA171
        Selected historical response, composition and number-density convention.
    density_component : int
        Local mass-density component index.
    temperature_component : int
        Local kelvin temperature component index.
    temperature_label : str
        Explicit provenance/interpretation of the kelvin temperature input.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent number-density (cm^-3) and temperature (K) with common valid support.
    """
    require_fields(density)
    if (not isinstance(temperature_label,str) or not temperature_label.strip() or
            not np.isfinite(density_unit_g_cm3) or density_unit_g_cm3<=0 or
            type(density_component) is not int or not 0<=density_component<len(density.fields)):
        raise ValueError("positive density units, component and temperature provenance are required")
    require_continuous(density,(density_component,),halo=0,operation="thermal density")
    tproduct=isinstance(temperature,Fields)
    if tproduct:
        require_fields(temperature)
        if (temperature.mesh is not density.mesh or type(temperature_component) is not int or
                not 0<=temperature_component<len(temperature.fields) or
                temperature.fields[temperature_component].units!='K' or
                np.any(temperature.slot_of_leaf[density.leaf_ids]<0)):
            raise ValueError("temperature must be kelvin on the same mesh and prepared coverage")
    else:
        try:
            value=float(temperature)
        except (TypeError,ValueError):
            raise ValueError("temperature must be a positive kelvin scalar or prepared field") from None
        if np.ndim(temperature)!=0 or not np.isfinite(value) or value<=0:
            raise ValueError("temperature must be a positive kelvin scalar or prepared field")
        temperature=value
    if tproduct:
        require_continuous(temperature,(temperature_component,),halo=0,operation="thermal temperature")
    h, spatial_shape, boxes = _layout((density,temperature) if tproduct else (density,))
    shape=(len(density.leaf_ids),*spatial_shape,2)
    extra=temperature.nbytes if tproduct and temperature is not density else 0
    scratch=128*int(np.prod(shape[1:4]))
    required=density.nbytes+extra+density.mesh.nbytes+8*int(np.prod(shape))+scratch
    admit(required,memory_limit,"thermal fields")
    values=np.empty(shape)
    # Binding is invariant over this immutable group. Do not revalidate the
    # public window/selector contract for every block in a large snapshot.
    density_values=density.values
    density_slots=density.slot_of_leaf[density.leaf_ids]
    density_box=boxes[0]
    if tproduct:
        temperature_values=temperature.values
        temperature_slots=temperature.slot_of_leaf[density.leaf_ids]
        temperature_box=boxes[1]
    for row,slot in enumerate(density_slots):
        rho=density_values[(slot,*density_box,density_component)]*density_unit_g_cm3
        values[row,...,0]=model.composition.number_density(rho,convention=model.density_convention)
        t=(temperature_values[(temperature_slots[row],*temperature_box,temperature_component)]
           if tproduct else temperature)
        model.response(t)
        values[row,...,1]=t
    definitions=(FieldDefinition("emission_measure_density","cm^-3","prepared-node"),
                 FieldDefinition("temperature","K","prepared-node"))
    return publish(density.mesh,values,density.selection,definitions,h,h,
        density.scheme+'/thermal-nodes',(density.source,model.identity,temperature_label),
        {"model":model.identity,"temperature":temperature_label,
         "density_unit_g_cm3":float(density_unit_g_cm3),"controlled_upper_bytes":required})


def _check_thermal(fields,model,*,halo=1):
    require_continuous(fields,halo=halo,operation="thermal LOS" if halo else "emissivity")
    if (len(fields.fields)!=2 or tuple(f.units for f in fields.fields)!=('cm^-3','K') or
            not isinstance(fields.source,tuple) or len(fields.source)!=3 or fields.source[1]!=model.identity):
        raise ValueError("thermal fields and response model must have matching physical identity")


def emissivity_fields(thermodynamics,*,model=AIA171(),memory_limit=None):
    """Apply response to prepared nodes before interpolation, retaining that order.

    Parameters
    ----------
    thermodynamics : Fields
        Number-density and kelvin fields produced by thermal_fields with the same model.
    model : AIA171
        Response matching the supplied thermal fields.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Node emissivity in DN s^-1 pixel^-1 cm^-1, preserving valid support.

    Notes
    -----
    Response-before-interpolation defines emissivity-first reconstruction; it is not interchangeable with thermodynamics-first.
    """
    _check_thermal(thermodynamics,model,halo=0)
    h=thermodynamics.valid_halo
    block=thermodynamics.mesh.block_shape
    shape=(len(thermodynamics.leaf_ids),*(n+2*h for n in block),1)
    required=(thermodynamics.nbytes+thermodynamics.mesh.nbytes+8*int(np.prod(shape))+
              128*int(np.prod(shape[1:4])))
    admit(required,memory_limit,"emissivity")
    values=np.empty(shape)
    backing=thermodynamics.values
    slots=thermodynamics.slot_of_leaf[thermodynamics.leaf_ids]
    offset=thermodynamics.storage_halo
    box=tuple(slice(offset-h,offset+n+h) for n in block)
    for row,slot in enumerate(slots):
        data=backing[(slot,*box,slice(None))]
        values[row,...,0]=model.from_number_density(data[...,0],data[...,1])
    return publish(thermodynamics.mesh,values,thermodynamics.selection,
        (FieldDefinition('aia171_emissivity','DN s^-1 pixel^-1 cm^-1','prepared-node'),),h,h,
        thermodynamics.scheme+'/response-before-interpolation',thermodynamics.source,
        thermodynamics.preparation_stats)
