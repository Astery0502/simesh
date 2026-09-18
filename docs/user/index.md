# User guide

Use this guide after the [installation and first run](../../README.md#install-and-run)
in the README. It explains how to combine interfaces for analysis, regional
extraction and result delivery. The [API reference](api.md) supplies exact
signatures, parameter definitions and result contracts from source docstrings.

Choose a runnable example below, or use the
[application guide (Chinese)](../application-overview.md) to select a workflow by
scientific question. The recipes use these imports:

```python
import simesh as sm
from simesh import applications as app
```

## Examples

| Example | Input and output |
| --- | --- |
| [Quickstart](../../examples/user_quickstart.py) | Generated teaching snapshot and reloadable magnetic map |
| [Magnetic applications](../../examples/standard_applications.py) | Synthetic AMR arcade, Q/twist, paths and LOS; custom NumPy archive |
| [Uniform export](../../examples/uniform_export.py) | AMRVAC fields to NumPy memory maps or uniform VTK using explicit reconstruction |
| [MHD analysis](../../examples/recovered_state_analysis.py) | Analytic recovered state, reductions, profiles and result files |
| [Radiation bands](../../examples/radiation_bands.py) | Isothermal snapshot, EUV/radio images, optical depth and convergence checks |
| [Root-subtree crop](../../examples/root_crop.py) | Regional analysis, independent AMR export, reloadable slice and integral |
| [Along-line calculus](../../examples/line_calculus.py) | RK scalar integrals, weighted means, bounded continuation and sampled-curve derivatives |
| [Current-based proxy](../../examples/current_proxy.py) | Native or supplied seed quadrature, traced-line contributions and checkpointed display volumes |

## Current-based morphology proxy

This composed magnetic application connects native seed geometry, existing
field-line tracing and `sample_line_profiles` to an area-weighted display
product. Its specialized steps are closed-line selection, the arc-length mean
of squared interpolated curl, and one contribution per visited display cell. Default
seeds are the original AMR bottom-face cell centers, weighted by their cell-face
areas. Pass a `PointSet` and explicit `seed_areas` to override the seed geometry.

```python
with sm.open_amrvac("snapshot.dat") as source:
    magnetic = sm.prepare(source, ("b1", "b2", "b3"), scheme="exact-phase")
proxy = sm.current_proxy(magnetic, (128, 128, 128),
                         step=float(magnetic.mesh.spacing.min()) / 4,
                         max_steps=20000, workers=4)
dz = (proxy.upper[2] - proxy.lower[2]) / proxy.emissivity.shape[2]
image = proxy.emissivity.sum(axis=2) * dz
```

This is an uncalibrated magnetic morphology proxy following
[Cheung & DeRosa (2012)](https://doi.org/10.1088/0004-637X/757/2/147), not an EUV
response model. Inspect `closed`, `accepted` and original termination diagnostics separately; vary seed
quadrature, integration step and display resolution before quantitative use.
`iter_current_proxy` yields sparse batch increments for checkpointed applications.
The resident result keeps per-seed diagnostics without retaining those consumed
increments.

To reuse an existing `LineSet` named `lines`, supply its original tracing step
and area weights in `lines.seeds` order:

```python
contribution = sm.current_proxy_from_lines(
    magnetic, lines, (128, 128, 128),
    seed_areas=seed_areas, step=trace_step,
)
```

This route does not retrace the field. The same paths can also feed ordinary
line profiles, or a new proxy grid and new weights. Use `lines.select(mask)`
with the same selection on area weights to reuse a subset. Reusing loaded paths
requires matching snapshot and coordinate conventions. Squaring curl before
interpolation would define a different proxy; native-volume reductions and
thermal radiation are separate consumers with different measures and models.

The [command-line example](../../examples/current_proxy.py) supports native seeds
or an NPZ containing `positions`, `areas` and optional `ids`.

## EUV and radio synthesis

Prepare density explicitly, select a response, then integrate identified rays:

```python
units = sm.MHDUnits.solar()  # Select scales appropriate to the simulation.
model = sm.EUV(wavelength=193)
with sm.open_amrvac("snapshot.dat", fields=["rho"]) as source:
    density = sm.prepare(source, scheme="exact-phase")
thermal = sm.thermal_fields(density, 1e6, model=model,
    density_unit_g_cm3=units.density_g_cm3, temperature_label="prescribed 1 MK")
plane = sm.orthographic_plane(density.mesh.lower, density.mesh.upper, [0, 0, 1], (64, 64))
rays = sm.RaySet.from_plane(plane, [0, 0, 1])
thin = app.thermal_los(thermal, rays, model=model, length_unit_cm=units.length_cm)
coefficients = sm.radiation_fields(thermal, model=model, absorption=sm.HHeAbsorption())
thick = sm.radiative_los(coefficients, rays, length_unit_cm=units.length_cm)
if not thick.complete:
    raise RuntimeError("Incomplete radiation coverage or integration")
sm.save_result("euv193.result.npz", thick.intensity)
sm.save_result("euv193-tau.result.npz", thick.optical_depth)
```

For radio, select `sm.RadioFreeFree(frequency_hz=17e9)` when creating thermal
fields and call `radiation_fields(thermal, model=model)` without a separate
absorber. `radiative_los` returns brightness temperature and its unabsorbed
counterpart. Model choices, table domains, reconstruction differences, observer
direction and physical limits are defined in the [API reference](api.md).
`AIA171` retains its historical default; `EUV` defaults to the current upstream
emission-measure convention. Use the supplied example to synthesize every channel
and compare native and reference integration.

## Native AMR sections

Extract a section while retaining each leaf block's physical cell spacing:

```python
with sm.open_amrvac("snapshot.dat") as source:
    fields = sm.read_fields(source, ("rho", "b3"))
section = sm.slice_axis(fields, "z", 0.5)
geometry = section.geometry
u_edges, v_edges = geometry.cell_edges(0)
block_values = section.values[0]
sm.save_result("section.result.npz", section)
```

The [API reference](api.md) describes block layout, interface-side selection,
coverage and ownership. Geometry and values are independent of plotting tools.

## Root-block regional analysis and independent output

Combine adjacent level-1 root blocks into a rectangular region, retaining each
root's complete refinement subtree. Root indices are zero-based with exclusive
upper bounds; this example selects 3 by 2 by 2 roots in a sufficiently large mesh:

```python
box = ((1, 1, 0), (4, 3, 2))
sm.crop_amrvac("snapshot.dat", "crop.dat", root_bounds=box, fields=("rho", "b3"))
```

For repeated analysis or derived-field export, retain the original Mesh in memory:

```python
with sm.open_amrvac("snapshot.dat") as source:
    selection = sm.select_roots(source.mesh, box)
    fields = sm.read_fields(source, ("rho", "b3"), region=selection)
    metadata = source.metadata
sm.write_amrvac("crop.dat", fields, metadata=metadata, root_bounds=box)
```

For interpolated regional maps or derivatives, prepare the selection while the
original Source is open. Explicit output bounds keep the map inside that region:

```python
with sm.open_amrvac("snapshot.dat") as source:
    selection = sm.select_roots(source.mesh, box)
    ready = sm.prepare(source, ("rho", "b3"), region=selection, scheme="exact-phase")
grid = app.uniform_grid(ready, (48, 32, 32), bounds=selection.requested_bounds)
```

Regional preparation reads neighboring support from the original domain. By
contrast, reopening `crop.dat` makes its edges the new domain boundaries. Include
enough surrounding roots for derivatives and integration paths; an independently
cropped field need not reproduce original-domain results near those edges.

The [API reference](api.md) specifies indexing, coverage, bounded payload storage
and output boundary semantics. A crop retains original refinement and values;
it does not create a level-1 uniform grid. Field units and crop provenance belong
in a separate description, as illustrated by the executable example.

## Field-line integration steps

Magnetic tracing and QSL/Twist diagnostics default to `step_fraction=0.25`:
steps are capped by one quarter of the smallest local cell edge. RK stages
entering finer cells cause the step to shrink and restart. An optional `step`
adds an absolute coordinate-length cap. Use `step_fraction=None, step=...`
to reproduce fixed-step tracing. This controls spatial sampling, not an
estimate of local integration error.

The same policy applies to raw and application tracing, bounded pools, batched
paths and current-proxy tracing. The proxy still requires a `step` cap no larger
than a display cell. Raw traces retain accepted prefixes; QSL diagnostics retain
their separate boundary-localization behavior. Smaller local steps may require
larger `max_steps` to reach the boundary. Previously saved figures and results
retain their original step controls and are not recomputed automatically.

## Compose along-line calculations

Choose the entry point according to the requested result:

| Requested result | Interface to inspect |
| --- | --- |
| Reusable field-line geometry | [`applications.trace`](api.md#simesh.applications.trace), returning `LineSet` |
| QSL or twist values without stored paths | [`applications.connectivity`](api.md#simesh.applications.connectivity) |
| Scalar integrals calculated during tracing | [`trace`](api.md#simesh.trace), returning `TraceResult` with optional trajectory storage |
| Quantities calculated from an existing path | [`sample_line_profiles`](api.md#simesh.sample_line_profiles), followed by sampled-curve operations |

Path storage and scientific diagnostics are separate choices. Request the paths
when they are part of the deliverable; a scalar diagnostic alone need not retain
them. The result types and exact controls are documented at the linked interfaces.

Pass prepared scalar `Fields` as `integrands` to `sm.trace` or `sm.iter_traces`
to integrate all components alongside the trajectory without retaining a path:

```python
traced = sm.trace(vector, seeds, integrands=rates, max_length=0.2, workers=4)
values = traced.integrals
```

The rates must share the vector's Mesh and have one valid halo. Their components
are interpolated at the same RK stage positions; a nonlinear transform prepared
on grid nodes therefore means "transform, then interpolate". Each integral uses
positive branch arc length and retains only accepted steps. Inspect termination
before treating a result as a complete physical line. Weighted averages use two
rates, weight times quantity and weight; divide their integrals only when the
weight integral is nonzero. No automatic unit conversion is performed.

`trace_bounded` accepts the same optional rates as resident Fields covering the
entire Mesh. Its primary/curl pool still manages vector coverage; the rates are
accounted in the memory budget and are never silently loaded by that pool.

For an existing `LineProfile`, apply NumPy transformations and use the independent
sampled-curve operations:

```python
integral = sm.line_integral(profile.values, profile.arclength)
derivative = sm.line_derivative(profile.values, profile.arclength, edge_order=2)
```

These use the supplied sample sequence and distances, not the original RK
stages. Check `profile.usable` first. Joined profiles retain both seed samples;
remove repeated distances before differentiation. Summation and extrema can use
NumPy directly, with a deliberately chosen sampling grid and validity mask.
The [runnable example](../../examples/line_calculus.py) verifies RK integrals,
weighted averages, bounded continuation and sampled derivatives on an AMR field.

## Magnetic connectivity methods

QSL calculations default to single-line variational transport. Prepare two valid
halo layers, then use `sm.qsl(ready, seeds)` or
`app.connectivity(ready, points, quantities="q")`. Select
`method="finite-difference"` to use neighboring-seed endpoint differences;
`delta` applies only to that method. Both methods are available through batched
and application interfaces, and saved results retain the selected method.

Variational transport avoids four additional neighboring trajectories per seed,
but retains a nine-component unit-vector gradient field. Its total memory and
runtime depend on field coverage and seed count. Changing the default does not
change previously saved maps; reproduce those with their recorded method.

## Compose and save analyses

Select continuous quantities by name, so changing the output order does not
change their meaning. After recovering density and temperature in `state`:

```python
thermal = sm.thermal_fields(state, state, density_component="density",
                            temperature_component="temperature", density_unit_g_cm3=1.,
                            temperature_label="recovered ideal-MHD temperature")
column = app.los(state, rays, component="density")
```

The scalar column retains coordinate-length units; physical column density
requires the corresponding explicit length conversion. Use `quantities="q"`,
`"twist"` or `("q", "twist")` consistently with all application connectivity
entry points. A compatibility `twist=False` control selects Q alone only when
`quantities` is omitted; specifying both is rejected.

The same result protocol supports native AMR sections, integral/mean/flux results,
extrema and histograms, as well as sampled maps, lines, profiles and projections:

```python
integral = sm.volume_integral(fields, "rho", units=sm.LengthUnits(1e8, "cm"))
sm.save_result("integral.result.npz", integral,
               metadata={"description": "density integral with explicit length scale"})
restored = sm.load_result("integral.result.npz").result
print(restored.value, restored.units, restored.coverage.fraction)
```

Choose the length scale to match the actual simulation; labels on stored fields
do not convert values. Reductions retain coverage, weights, histogram tails and
surface/location information. Native sections include the original mesh topology
so their block IDs and cell edges can be reconstructed. Whole-result saving and
loading require resident memory; caller-supplied source descriptions remain unverified.

## Uniform output

Export stored fields directly from a snapshot:

```python
grid = sm.export_uniform("snapshot.dat", (256, 256, 256),
                         fields=("rho",), interpolation="zero", batch_size=32)
```

For already loaded or derived interiors, use
`app.uniform_grid(fields, resolution, interpolation="zero")`. Select
`interpolation="native"` for exact placement on the matching source cell lattice.
See the [API reference](api.md) for coverage, output ownership and memory limits.

```bash
.venv/bin/python examples/uniform_export.py snapshot.dat \
  --resolution 256 256 256 --fields rho --output example-output/uniform
```

For direct trilinear output with bounded preparation, choose the scheme explicitly:

```python
grid = sm.export_uniform("snapshot.dat", (256, 256, 256),
                         fields=("rho",), interpolation="linear",
                         scheme="exact-phase", batch_size=32, tile_rows=16)
```

The same workflow is available in the example with
`--interpolation linear --scheme exact-phase`.

## Uniform VTK output

Save an existing uniform result or resample a snapshot directly to a VTK file:

```python
sm.write_uniform_vtk("rho.vtk", grid)
sm.export_uniform_vtk("snapshot.dat", "magnetic.vtk", (256, 256, 256),
                      fields=("b1", "b2", "b3"), interpolation="zero")
```

The [binary legacy VTK format](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html)
uses a uniform cell grid spanning the result bounds. Each component is a separate
scalar cell array; `simesh_valid` records coverage independently of finite values.
Units and source metadata are not stored. Export supports uniform volumes only,
including volumes resampled from AMR data; it does not write AMR hierarchy or lines.
Direct export retains a complete volume and accepts the same sampling controls as
`export_uniform`, including explicit `scheme="exact-phase"` for linear interpolation.

```bash
.venv/bin/python examples/uniform_export.py snapshot.dat --format vtk \
  --resolution 256 256 256 --fields rho --output rho.vtk
```

## Classic AIA colors

Install `.[plot]` to use the bundled SunPy-derived AIA palettes offline:

```python
from simesh.colormaps import aia_colormap

image = ax.imshow(intensity, origin="lower", cmap=aia_colormap(171), norm=norm)
fig.colorbar(image, ax=ax)
```

Use the full wavelength in Angstroms, such as 94, 131, 171 or 193. Choose
intensity normalization separately; a display palette does not change a
band's temperature response. Source and license information is in
the repository’s `ASSETS.md`.

For IRIS 1354 Angstrom intensity maps, use `iris_colormap("FUV")` from the
same module. `eis_intensity_colormap()` follows EISPAC's `Blues_r` intensity
convention for all EIS lines; it is not a separate color assignment per wavelength.
Both functions return independent palettes and leave normalization to the caller.
