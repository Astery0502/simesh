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
