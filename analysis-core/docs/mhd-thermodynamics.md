# Explicit MHD thermodynamics

Import this application directly from `simesh.physics.mhd`. It recovers
classical ideal-gas MHD quantities from completed native `Fields`, with
explicit energy definition, gamma, composition and conversion factors.
It does not inspect file metadata to select a physical model.

```python
from simesh.physics.mhd import IdealMHD, MHDUnits, MHDStatus, MHDStateError, mhd_fields
from simesh.diagnostics import MagneticUnits
from simesh.physics.thermal import CoronalComposition

model = IdealMHD(
    gamma=5/3,
    energy_kind="total",
    composition=CoronalComposition(helium_abundance=0.1),
    units=MHDUnits(
        density_kg_m3=1e-12,
        momentum_kg_m2_s=1e-7,
        energy_j_m3=0.01,
        magnetic=MagneticUnits(field_tesla=1.1209982432795857e-4, length_m=1e6),
    ),
)

# These illustrative factors must be replaced by the actual run normalization.
state = mhd_fields(ready, model=model)
velocity = mhd_fields(ready, model=model, outputs="velocity")
```

All factors multiply stored values to obtain SI. The density input is mass per
volume; momentum is momentum **density**, not velocity; energy is energy per
volume, not specific energy or primitive pressure. Magnetic components must be
the complete collocated Cartesian B vector used in that energy definition.
The same uniform scalar permeability is used throughout.

`MHDUnits` requires independent density, momentum-density and energy-density
scales, and reuses `MagneticUnits` for B, permeability and coordinate length.
Labels such as `code`, `SI`, `e`, or `pressure` never establish a scale or energy
definition. The default `MagneticUnits` permeability is the existing conventional
`4*pi*1e-7 H/m`; supply another value explicitly if needed. Local recovery does
not use coordinate length. It is retained for tracing/LOS length conversions.

For a standard nondimensional system with rho scale `rho0`, velocity scale
`v0` and unit magnetic permeability, use momentum scale `rho0*v0`, energy scale
`rho0*v0**2` and B scale `sqrt(mu*rho0*v0**2)`. For CGS input measured in g/cm³,
g/(cm² s), erg/cm³ and gauss, the respective SI multipliers are `1000`, `10`,
`0.1` and `1e-4`. Choosing these factors remains the caller's responsibility.

## Model and quantities

`IdealMHD` requires finite `gamma > 1` and an explicit `CoronalComposition`.
The composition is the existing fully ionized H/He approximation, with
`helium_abundance = n_He/n_H`, neglecting electron mass. Temperature delegates
to `CoronalComposition.temperature(rho_cgs, p_cgs)` after conversion from SI.
No new ionization or EOS machinery is introduced.

After conversion to SI, the implemented relations are

```text
v = m / rho
K = (m dot v) / 2
P_B = |B|^2 / (2 mu)
u = E - K - P_B     for energy_kind="total"
u = E              for energy_kind="internal"
p = (gamma - 1) u
T = existing H/He temperature(rho * 1e-3, p * 10)
```

The total-energy branch assumes precisely internal + kinetic + magnetic energy.
The internal-energy branch subtracts neither kinetic nor magnetic energy.
It still requires momentum and B for a consistent complete MHD state and its
diagnostics. The model is checked on every call, even for velocity-only or
status-only output.

`outputs` accepts one key or a nonempty sequence of unique keys. Output columns
follow that order; `velocity` expands to three Cartesian components.

| Key | Columns / definition | Units |
| --- | --- | --- |
| `density` | Converted mass density | `kg m^-3` |
| `velocity` | `vx`, `vy`, `vz` | `m s^-1` |
| `speed` | Magnitude of v | `m s^-1` |
| `internal_energy` | Recovered u per volume | `J m^-3` |
| `pressure` | Thermal gas pressure p | `Pa` |
| `temperature` | H/He temperature | `K` |
| `beta` | p / P_B | `1` |
| `sound_speed` | sqrt(gamma p / rho) | `m s^-1` |
| `alfven_speed` | magnitude(B) / sqrt(mu rho) | `m s^-1` |
| `sonic_mach` | magnitude(v) / sound_speed | `1` |
| `alfven_mach` | magnitude(v) / alfven_speed | `1` |
| `status` | `mhd_status`, exact integer flags stored as float64 | `1` |

Defaults are density, velocity, pressure, temperature, beta, sound speed,
Alfvén speed, sonic Mach, Alfvén Mach and status. Mach numbers are nonnegative
speed ratios in the input velocity frame. The Alfvén speed uses the full B
magnitude, not a direction-projected characteristic speed. Magnetosonic speeds
and changes of reference frame are outside this interface.

## Field selection, AMR support and ownership

```python
state = mhd_fields(
    conserved,
    magnetic=magnetic,
    model=model,
    density="mass",
    momentum=("mom_x", "mom_y", "mom_z"),
    energy="energy_density",
    magnetic_components=("Bx", "By", "Bz"),
    outputs=("pressure", "temperature", "status"),
)
```

Selectors accept unique names or integer component positions in their respective
groups. With `magnetic=None` (the default), B is selected from `conserved`.
Defaults are `rho`, `(m1,m2,m3)`, `e`, and `(b1,b2,b3)`. Ambiguous names,
duplicate vector components and overlapping physical roles in the same group
are rejected. Vector components must share unit labels; scalar fields may have
different labels because their SI multipliers are independently specified.

Both groups must refer to the same `Mesh` object and exactly the same covered
leaf set. Their selection order and physical storage slots may differ. The
result is packed in conserved selection order, retains its `Selection` and
requested bounds, and preserves the minimum input `valid_halo`. Only that
support is evaluated; allocated invalid padding is ignored. An invalid physical
node does not remove its leaf or change `valid_halo`, which describes spatial
support, not thermodynamic admissibility.

Recovery evaluates the input nodes, including prepared AMR support, **before**
downstream interpolation. A nonlinear conversion of conserved averages is not
an average of primitive variables. Preparing already recovered interiors or
recovering interpolated conservative values defines a different reconstruction.
This implementation does neither implicitly. Total-energy subtraction also
cannot recover internal energy already lost to floating-point cancellation in
the input, notably in very low beta or highly supersonic states.

Outputs own readonly storage and survive source closure and prepared-batch
expiration. They retain the model and source provenance, not input arrays or a
batch lease. `memory_limit` admits input arrays, mesh, output, the new leaf
directory and a fixed conservative per-block scratch allowance. Computation
processes one leaf at a time, with no shared framework or global registry.
No new halos are generated. Interior-only input remains interior-only; tracing
and thermal consumers retain their existing support requirements.

## Invalid states and diagnostic singularities

The default `invalid="raise"` stops at the first invalid state in selection
order, before publishing a result. `MHDStateError` exposes `leaf_id`,
`cell_index` and `status`. Cell indices are relative to the interior: negative
indices and indices at or above the block size identify halo nodes.

`invalid="nan"` publishes NaN in **all** physical output columns at an invalid
state. It never floors density, clips pressure, substitutes a temperature or
drops bad nodes from coverage. Include `status` to locate each condition:

| `MHDStatus` flag | Value | Meaning |
| --- | --- | --- |
| `OK` | 0 | No detected condition |
| `NONFINITE_INPUT` | 1 | A selected input component is NaN or infinite |
| `NONPOSITIVE_DENSITY` | 2 | Converted density is zero or negative |
| `NONPOSITIVE_INTERNAL_ENERGY` | 4 | Recovered internal energy is zero or negative |
| `UNREPRESENTABLE_STATE` | 8 | Otherwise admissible state cannot yield finite positive thermodynamics in float64 |
| `ZERO_MAGNETIC_FIELD` | 16 | Valid state, but beta and Alfvén Mach are undefined |
| `UNREPRESENTABLE_DIAGNOSTIC` | 32 | Valid thermodynamics with a nonfinite diagnostic speed or ratio |

Flags combine by bitwise OR. The first four flags define invalid states; the
last two are nonfatal. Exact zero B is allowed: Alfvén speed is zero, beta and
Alfvén Mach are NaN (including zero flow / zero B), and the other outputs remain
available. Overflowing diagnostic values become NaN with flag 32, without
invalidating finite thermodynamics. These diagnostics are evaluated for status
reporting even when their output columns are not requested.

Statistics are always visible in `preparation_stats["status_counts"]` and
`preparation_stats["invalid_state_counts"]`, separately for `interior` and
`evaluated` (interior plus valid halo). These count nodes, including repeated
support nodes in different leaves; they are not physical-volume statistics.
Status is categorical: inspect `interior()` or `window()` and convert the status
column to integers. Do not interpolate, differentiate or integrate status bits.
For sampled physical values, check consumer status and finiteness separately.
The existing `thermal_fields` rejects invalid density/temperature; NaN recovery
does not silently turn bad plasma into zero emission.

## Complete temperature-to-LOS and velocity workflow

This executable example creates an isothermal 0.8 MK state on a mixed AMR mesh,
recovers temperature, verifies a thermal column and traces velocity in both
directions. Physical normalization is chosen explicitly and is consistent with
the unit-permeability stored MHD energy formula.

```python
import numpy as np
import simesh as sm
from simesh import applications as app
from simesh.physics.mhd import IdealMHD, MHDUnits, mhd_fields

rho0, v0, length_m = 1e-12, 1e5, 1e6
mu = 4*np.pi*1e-7
units = MHDUnits(
    density_kg_m3=rho0, momentum_kg_m2_s=rho0*v0, energy_j_m3=rho0*v0**2,
    magnetic=sm.MagneticUnits(field_tesla=np.sqrt(mu*rho0*v0**2), length_m=length_m),
)
composition = sm.CoronalComposition(helium_abundance=0.1)
model = IdealMHD(gamma=5/3, energy_kind="total", composition=composition, units=units)
mesh = sm.mesh_from_forest(
    (2, 1, 1), np.array([False]+[True]*9),
    lower=(0, 0, 0), upper=(2, 1, 1), block_shape=(8, 8, 8),
)
rho, target_temperature = 2., 8e5
# Invert the existing composition conversion for this manufactured state.
rho_cgs = rho*rho0*1e-3
p_cgs = target_temperature / composition.temperature(rho_cgs, 1.)
p = (p_cgs/10) / units.energy_j_m3
vz, bz = .2, 1.
e = p/(model.gamma-1) + rho*vz**2/2 + bz**2/2
raw = np.empty((mesh.leaf_count, 8, *mesh.block_shape))
raw[:] = np.array([rho, 0., 0., rho*vz, e, 0., 0., bz])[None, :, None, None, None]
with sm.source_from_arrays(mesh, raw, ("rho", "m1", "m2", "m3", "e", "b1", "b2", "b3")) as source:
    ready = sm.prepare(source, scheme="exact-phase")

state = mhd_fields(ready, model=model, outputs=("density", "temperature"))
np.testing.assert_allclose(state.values[..., 1], target_temperature, rtol=1e-13)
response = sm.AIA171(composition=composition)
thermal = sm.thermal_fields(
    state, state,
    density_component=0, temperature_component=1,
    density_unit_g_cm3=1e-3,  # Recovered density is already SI.
    temperature_label="ideal-MHD total-energy recovery, gamma=5/3, He/H=0.1",
    model=response,
)
plane = sm.orthographic_plane(mesh.lower, mesh.upper, [0, 0, 1], (6, 5))
image = sm.integrate_thermal_los(
    thermal, plane, [0, 0, 1], length_unit_cm=length_m*100, model=response,
)
assert image.complete
expected = response.emissivity(rho_cgs, target_temperature)*image.depth*length_m*100
np.testing.assert_allclose(image.values, expected, rtol=1e-12)

velocity = mhd_fields(ready, model=model, outputs="velocity")
seeds = sm.PointSet([[.25, .25, .5], [1.5, .5, .5]])
lines = app.trace(velocity, seeds, step=.025, max_steps=8)
for seed_id in seeds.ids:
    assert len(lines.branch(seed_id, -1)) == len(lines.branch(seed_id, 1)) == 9
```

Tracing follows the instantaneous velocity direction using the existing
normalized-vector, spatial-arclength integrator. Step and path length are in
mesh coordinate units; this is a streamline, not a time-evolved particle orbit.
Use `outputs="velocity"` to supply exactly three columns. `app.trace` defaults
to both directions. Temperature is a kelvin field accepted by `thermal_fields`;
when passing original stored density instead, use the original density factor
`model.units.density_kg_m3*1e-3` and its selected component.

## AMRVAC scope and official references

The official [MHD equations](https://amrvac.org/md_doc_2equations.html) identify
momentum as rho times velocity and use unit magnetic permeability in the stored
normalization. Their classical pressure recovery agrees with the total and
internal branches above; see the official
[MHD implementation](https://amrvac.org/mod__mhd__phys_8t_source.html),
`mhd_get_pthermal_origin` and `mhd_get_pthermal_inte`.

The official [parameter documentation](https://amrvac.org/md_doc_2par.html)
distinguishes `mhd_internal_e` and `mhd_hydrodynamic_e` and warns that
semirelativistic MHD changes the momentum/energy relations. This interface
supports only the explicitly declared classical total/internal definitions.
Hydrodynamic energy (internal + kinetic), background magnetic or equilibrium
splitting, semirelativistic/relativistic MHD, no-energy/isothermal evolution,
partial ionization, tabulated EOS, radiation energy and additional energy
reservoirs are not supported. The interface cannot detect these definitions
from numeric arrays; check the simulation configuration first. It does not
claim compatibility with every AMRVAC physics module. References checked
2026-09-09.
