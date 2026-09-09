# Standard diagnostics and applications

Standard diagnostics return `Fields` for reuse across sampling, maps, uniform
grids and LOS. Preparation schemes remain explicit; physical-boundary derivatives
follow the chosen ghost conditions.

## Mathematical and magnetic fields

```python
import simesh as sm
from simesh import applications as app

strength = sm.magnitude(magnetic)
div_b = sm.divergence(magnetic, components=("b1", "b2", "b3"))
curl_b = sm.curl(magnetic)
grad_density = sm.gradient(density, component="rho")
b_squared = sm.dot(magnetic, magnetic)

units = sm.MagneticUnits(field_tesla=1e-3, length_m=1e6)
current = sm.current_density(magnetic, units=units)
pressure = sm.magnetic_pressure(magnetic, units=units)
energy = sm.magnetic_energy_density(magnetic, units=units)
```

Components may be selected by name or index and must share units within each
vector. Dot inputs need matching component counts, Mesh identity and coverage.
Pointwise results retain common valid support; gradient, divergence and current
consume one valid halo layer, as does curl.

`MagneticUnits` specifies tesla per stored field unit and metres per coordinate
unit. Gauss and centimetre inputs, for example, use `field_tesla=1e-4` and
`length_m=1e-2`. Labels alone do not convert values. Current uses the
magnetostatic/MHD relation `J = curl(B) / mu`, producing A/m². Pressure and energy
use `B² / (2*mu)`, producing Pa and J/m³, respectively.

This model assumes uniform scalar permeability. The default is the conventional
vacuum approximation `4*pi*1e-7 H/m`; supply `permeability_h_m` for another or a
more precise value. Pressure and energy are numerically equal under this model.
Nonfinite or unrepresentable field arithmetic is not automatically repaired.

Use raw curl, not physically scaled current, for connectivity's `curl_field`
argument. `magnitude` retains its input units without applying MagneticUnits.

## Uniform grids and field maps

```python
volume = app.uniform_grid(pressure, (128, 128, 64), memory_limit=2*1024**3)
x, y, z = volume.axes
values = volume.values  # (nx, ny, nz, component)
current_map = app.field_map(sm.magnitude(current), plane, workers=4)
image = current_map.image
```

`field_map` accepts a Plane or PointSet and retains definitions, IDs, owner leaves
and validity. `uniform_grid` returns values, validity, bounds and cell-center
axes. It collects the whole requested volume under a memory check; use existing
`iter_uniform` for streamed large outputs. Both interpolate, not conservatively rebin.
Their `usable` masks combine sampling coverage with finite values in every
component; the original `valid` coverage masks remain available.

## Q, twist, or both on a surface

```python
both = app.surface_diagnostics(magnetic, plane, quantities=("q", "twist"), workers=4)
q_only = app.bottom_diagnostics(magnetic, (128, 128), quantities=("q",), workers=4)
twist_only = app.surface_diagnostics(magnetic, plane, quantities=("twist",), workers=4)
twist_image = twist_only.image("twist")
twist_valid = twist_only.image("twist_valid")
```

Surfaces may be Planes or arbitrary PointSets. The bottom helper constructs
pixel-center seeds on the physical z-min face. Bounds, local radius, step,
limits and memory controls are forwarded to the native consumer.

- Q-only does not construct or integrate curl for twist.
- Twist-only integrates the two central branches with curl. It does not create
  unit-vector gradients, trace Q neighbors or evaluate Q.
- Combined requests reuse the same central integrations.

Twist-only preserves the endpoints and accepted-segment twist of a combined
finite-difference request with matching controls. Uncomputed Q arrays are None;
displaying or thresholding Q raises an error. Its `valid` mask means complete,
finite twist. Prefer `q_valid` and `twist_valid` when combining requests.

General array functions `line_diagnostics` and `iter_line_diagnostics` provide
the same quantity selection. Existing `qsl(..., twist=True/False)` and `iter_qsl`
still compute Q with optional twist. Q_perp accompanies Q. `delta` requires Q;
Q-specific method/normalization settings do not add work to twist-only requests.

## Select points and trace separately

```python
selected = both.threshold(q_min=1e4, abs_twist_min=1., mode="any")
lines = app.trace(magnetic, selected,
                  step=float(magnetic.mesh.spacing.min())*.125,
                  max_steps=4000, workers=4)
```

The thresholds are illustrative. QSL/twist computation never saves paths.
Selection preserves seed IDs; the later trace defaults to both directions.
Its accepted-prefix endpoints remain distinct from localized diagnostic
footpoints. See [geometry and result interfaces](applications.md) for details.

## Scalar and thermal LOS

```python
view = sm.orthographic_plane(density.mesh.lower, density.mesh.upper,
                             (.3, .2, 1.), (256, 256))
rays = sm.RaySet.from_plane(view, (.3, .2, 1.))
column = app.los(density, rays, workers=4)
emission = app.thermal_los(thermal, rays, length_unit_cm=1e8, workers=4)
```

Scalar LOS retains scalar units times coordinate length. Thermal LOS requires
explicit thermodynamics, density/length normalization and the native AIA171
model; it does not infer snapshot temperature.

## Reproducible complete example

From `analysis-core/`:

```bash
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-demo
# Optional plotting dependency:
.venv/bin/python -m pip install '.[plot]'
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-demo --render-only
```

The example prepares an analytic mixed-level AMR arcade and density field,
computes current/pressure products, creates a bottom Q/twist map, selects points
and traces both branches, and computes scalar/thermal LOS images. It saves
`products.npz` and `summary.json`, including units, geometry and status.
`--plot`, or a later `--render-only` run, writes magnetic and LOS PNG figures.
Rendering saved products needs only NumPy and Matplotlib, without rerunning science.

The physical scales and isothermal temperature are illustrative. This is a
composition example, not real-snapshot calibration or high-Q stress validation.
