# User interfaces

```python
import simesh as sm
from simesh import applications as app
```

[API reference](api.md): choose a function and read its signature, parameters,
return type and required support. Native analysis supports balanced nonperiodic
Cartesian 3D AMRVAC v5 ordinary fields; units and physical models are explicit.

## Install and run

From the repository root, with Python 3.11 or newer:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python examples/user_quickstart.py --output example-output/user-quickstart
```

Use a fresh output directory. The example creates its input and verifies
`Bz = 0.001 T`, `mass = 1e6 kg`, 64 usable map samples and result save/load.
The scales are teaching values, not simulation calibration.

## Examples

| Example | Input and output |
| --- | --- |
| [Quickstart](../../examples/user_quickstart.py) | Generated teaching snapshot and reloadable magnetic map |
| [Magnetic applications](../../examples/standard_applications.py) | Synthetic AMR arcade, Q/twist, paths and LOS; custom NumPy archive |
| [Uniform export](../../examples/uniform_export.py) | AMRVAC fields to NumPy memory maps or uniform VTK using explicit reconstruction |
| [MHD analysis](../../examples/recovered_state_analysis.py) | Analytic recovered state, reductions, profiles and result files |
| [Root-subtree crop](../../examples/root_crop.py) | Mixed-level AMR snapshot, root-aligned regional export and integral comparison |

Exact API text is generated when
building the documentation; GitHub Markdown displays the object directives.

## Native AMR sections

Extract a section while retaining each leaf block's physical cell spacing:

```python
with sm.open_amrvac("snapshot.dat") as source:
    fields = sm.read_fields(source, ("rho", "b3"))
section = sm.slice_axis(fields, "z", 0.5)
geometry = section.geometry
u_edges, v_edges = geometry.cell_edges(0)
block_values = section.values[0]
```

The [API reference](api.md) describes block layout, interface-side selection,
coverage and ownership. Geometry and values are independent of plotting tools.

## Root-aligned AMR output

Keep complete refined subtrees using integer root-block bounds:

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

The [API reference](api.md) specifies indexing, coverage, bounded payload storage
and output boundary semantics. A crop retains original refinement and values;
it does not create a level-1 uniform grid. Field units and crop provenance belong
in a separate description, as illustrated by the executable example.

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
