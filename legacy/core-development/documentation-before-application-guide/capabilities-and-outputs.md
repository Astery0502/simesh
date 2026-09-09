# Scientific capabilities and output guide

The independent `simesh` distribution provides Python workflows from AMRVAC or
array input through physical analysis to saved numerical products. All native
and compatibility implementations listed here are bundled in the root package;
the parent package and historical worktrees are not runtime dependencies.

This inventory describes the current implementation. It does not establish
numerical equivalence for every old script or validation on every physical model.
See [migration](../../../MIGRATION.md) for interface changes and array layouts.

## Physical capability inventory

Native functions below are exported from `simesh`, unless a namespace is shown.
`app` denotes `simesh.applications`.

| Scientific task | Entry points | Result and scope |
| --- | --- | --- |
| Potential-field extrapolation | `simesh.tools.potential_field_green` | Cell-centered Cartesian vector field from a bottom magnetic array; retained from the parent package |
| Analytic magnetic configurations | `simesh.tools.configurations` | Retained bipolar, dipole, monopole, RBSL, TDm and fan array helpers |
| Custom quantities and field composition | `derive`, `derive_many`, `select_fields`, `merge_fields` | Independent fields with named components, units and common valid support |
| Mathematical diagnostics | `magnitude`, `dot`, `gradient`, `divergence`, `derivative`, `curl` | Pointwise or centered-derivative fields; derivatives consume one valid halo layer |
| Magnetic current, pressure and energy density | `current_density`, `magnetic_pressure`, `magnetic_energy_density` | Explicit `MagneticUnits` normalization and scalar permeability |
| Ideal-MHD recovery | `mhd_fields`, `IdealMHD`, `MHDUnits` | Density, velocity, speed, internal energy, pressure, temperature, beta, sound/Alfvén speeds, Mach numbers and status; explicit total/internal energy model |
| Native integrals and statistics | `volume_integral`, `weighted_mean`, `extrema`, `histogram`, `surface_flux` | Cell-interior reductions with coverage and units; boxes and axis-aligned rectangular surfaces |
| Samples, slices and uniform volumes | `sample`, `sample_plane`, `iter_uniform`; `app.sample`, `app.field_map`, `app.uniform_grid` | Values with geometry and validity; uniform products are sampled views, not conservative rebinning |
| Magnetic lines and velocity streamlines | `trace`, `iter_traces`, `retrace`; `app.trace`, `app.iter_lines` | Accepted trajectory prefixes and termination status; optional magnetic twist; velocity lines are instantaneous streamlines, not time-dependent particle paths |
| Magnetic connectivity | `qsl`, `iter_qsl`, `line_diagnostics`; `app.surface_diagnostics`, `app.bottom_diagnostics` | Q, Q-perpendicular, localized footpoints, length and twist; Q-only and twist-only requests available through diagnostic interfaces |
| Quantities along stored curves | `sample_line_profiles`, `iter_line_profiles` | `LineProfiles` with seed IDs, two-branch geometry, arclength, component units and sampling masks |
| Scalar line-of-sight integration | `integrate_los`, `integrate_los_views`; `app.los` | Integrated scalar with coverage/termination status and explicit viewing geometry |
| Thermal line-of-sight synthesis | `thermal_fields`, `emissivity_fields`, `integrate_thermal_los`; `app.thermal_los` | Historical AIA171 response with explicit temperature, composition and density/length scales |

The parent package's main physical helpers, ordinary file outputs and earlier
analysis workflows have corresponding implementations here. Connectivity,
standard MHD recovery, reductions, identified profiles and application result
files extend that earlier surface. Capability continuity does not imply that
old `simesh.analysis` imports or implicit numerical choices are preserved.

## Supported physical scope

- Native file analysis accepts balanced, nonperiodic Cartesian 3D AMRVAC v5
  ordinary fields. Native array sources use equivalent 3D meshes. Preparation
  currently uses continuous physical boundaries. Units and thermodynamic models
  are explicit inputs; recorded header parameters do not select a model.
- Bundled `simesh.amrvac` retains mutable Dataset workflows, named derived
  fields, Cartesian 2D singleton-z arrays, ordinary reads/writes, and the
  nonperiodic `cont`, `symm`, `asymm`, `noinflow` ghost modes. These are separate
  interfaces from native Fields analysis.
- QSL localizes target-surface endpoints. Ordinary tracing preserves accepted
  prefixes; its endpoints need not coincide with QSL footpoints. Diagnostic
  Q/twist does not store trajectories: select seeds and trace them separately.
- CT face analysis/restart writing, periodic ghost exchange, spherical native
  analysis, GPU analysis, arbitrary curved-surface flux and general multi-band
  radiation are outside the delivered scope. They are not all missing features
  of the earlier canonical package.
- QSL and thermal LOS require resident input Fields even when results are
  batched. Real 10–20 GB input acceptance and high-Q stress validation are not
  established by the synthetic examples.

For numerical details, use the [standard diagnostics](standard-applications.md),
[MHD](mhd-thermodynamics.md), [reductions](reductions.md),
[connectivity](connectivity.md) and [profile](line-profiles.md) guides.

## Choose a result format

The filename extension `.npz` alone does not identify a simesh result file.
Use the producer and reader pair in this table.

| Product | Write or produce | Read or consume | Preserved meaning and limits |
| --- | --- | --- | --- |
| Identified maps, rays, lines, profiles and uniform grids | `save_result(path, result, metadata=..., source=...)` | `load_result(path).result` | Versioned NPZ schema restores the supported application type, nested geometry, IDs, units and statuses |
| Raw QSL diagnostic result | `save_result(path, qsl_result)` | `load_result(path).result` | Keeps seed positions and row order; use `ConnectivityMap` when IDs/image layout are required |
| Line/profile shards | `save_result_shards(directory, batches, seed_ids=...)` | `open_result_shards(directory).load(index).result` | Manifest, checksums and independently loadable NPZ shards; delivery completeness is separate from scientific validity |
| Scalar reductions, extrema and histograms | Explicit JSON/NumPy serialization, or JSON-compatible `metadata` on a supported result | JSON/NumPy, or `load_result(path).metadata` | Reduction dataclasses are not registered result types; explicitly include units, coverage, region/surface and histogram tails |
| Full native AMR fields | `write_amrvac(path, fields, metadata=...)` | `open_amrvac` or `simesh.amrvac` readers | Requires complete original-mesh coverage and matching metadata; ordinary interiors only, without halo, CT faces or field-unit/provenance encoding |
| Existing Dataset or uniform-array file workflows | `simesh.amrvac.write_datfile`, `write_datfile_from_uniform`, Dataset `write_datfile` | `simesh.amrvac` readers | Retained ordinary AMRVAC products, including the 2D convention |
| Level-1 uniform VTK | `simesh.amrvac.datfile_to_vtk` | A VTK reader | Retained legacy structured-points format and endpoint-coordinate convention; not an AMR hierarchy or polyline exporter |
| Example-specific array bundle | `numpy.savez_compressed` | `numpy.load(..., allow_pickle=False)` | Named arrays defined by that example; no automatic application-object restoration |
| Presentation figures | Example plotting with Matplotlib | PNG image viewer | Rendered views; numerical values, masks and provenance belong in accompanying data products |

`save_result` accepts `PointSet`, `RaySet`, `SampledPoints`, `ConnectivityMap`,
`QSLResult`, `RayResult`, `LineSet`, `LineProfiles` and `UniformResult`.
Raw `Fields`, `TraceResult`, `SliceResult`, `LOSResult`, `ThermalLOSResult` and
reduction dataclasses are not accepted by this schema. Use application wrappers
for the corresponding identified products, or explicitly serialize raw arrays.
`Plane` is supported as nested geometry only.

For a complete native field export, retain `source.metadata` before closing the
Source. `write_amrvac` accepts that detached `SnapshotMetadata` or a matching
header dictionary. A region-only field cannot be exported as the original full
forest. See [snapshot metadata](snapshot-metadata.md) and
[migration](../../../MIGRATION.md) for a complete example.

## Information to deliver with a result

Use `metadata` for calculation context and `source` for a caller-provided input
description. The following are application conventions, not required schema
keys or automatically collected settings:

| Context | Record explicitly when absent from the result |
| --- | --- |
| Input | Snapshot description, time and iteration, selected fields; optionally `source.metadata.to_dict()` and an independently computed fingerprint |
| Physical interpretation | Coordinate scale, field units, MHD energy definition/gamma/composition, magnetic normalization, thermal response and temperature assumption |
| Numerical calculation | Preparation scheme, region/target surfaces, trace step and limits, Q method/delta, LOS reconstruction and quadrature, selection thresholds |
| Output quality | Quantity-specific validity/termination, coverage, nonfinite handling, histogram underflow/overflow; counts alone do not replace masks |
| Software | `simesh.__version__` and an application/revision identifier if needed |

Only settings already stored on a result are saved automatically. JSON metadata
requires finite native Python values: convert arrays/scalars explicitly and
represent an unlimited control with a descriptive string, not `np.inf`.
Loading never verifies that the original snapshot matches the saved numbers or
restores an in-memory Fields identity. See [result files](result-files.md).

Sampling `valid` means coverage, while `usable` also checks finite values.
For combined connectivity maps use `q_valid` and `twist_valid`; a finite
accepted-segment twist can accompany an incomplete line. Inspect LOS status and
line termination before presenting a complete physical result. A complete shard
manifest means all requested seed products were delivered, not all integrations
succeeded. Shards do not implement resumable RK/LOS integration.

## Representative examples and their exact files

Run from the repository root after installing this distribution. Use distinct output
directories; both examples overwrite their own named output files on rerun.

```bash
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-standard
.venv/bin/python examples/recovered_state_analysis.py --output /tmp/simesh-quantitative
```

| Example | File | Contents and reader |
| --- | --- | --- |
| Standard magnetic and LOS applications | `products.npz` | Custom NumPy bundle: Q/twist maps and masks, current/divergence maps, sampled pressure volume, selected line IDs/positions/offsets/termination, scalar/thermal images and ray statuses; use `numpy.load` |
| Standard magnetic and LOS applications | `summary.json` | Illustrative physical scales, geometry and result counts; use JSON |
| Standard magnetic and LOS applications, with plotting | `magnetic-applications.png`, `los-applications.png` | Magnetic maps/selected lines and LOS figures |
| Recovered MHD state | `thermal.result.npz` | Restorable `RayResult`; caller metadata also records MHD model, mass, mean temperature, bottom flux and histogram edges/weights |
| Recovered MHD state | `streamlines.result.npz` | Restorable velocity `LineSet` with selected trace metadata |

The MHD example also computes and checks temperature/density/current profiles
in memory and prints quantitative checks. It does not write a profile file or
a standalone summary file. To persist the computed `profiles` object, use:

```python
sm.save_result(output / "profiles.result.npz", profiles,
               metadata={"description": "Quantities on stored streamlines"})
restored_profiles = sm.load_result(output / "profiles.result.npz").result
```

The standard example's bundle supports lightweight plotting and is not a
versioned result file. The summary is an example summary, not a full record of
every numerical control. For restorable geometry and richer metadata, save the
application objects using the producer/reader pairs above.

```python
import json
from pathlib import Path
import numpy as np
import simesh as sm

standard = Path("/tmp/simesh-standard")
with np.load(standard / "products.npz", allow_pickle=False) as data:
    q = np.where(data["q_valid"], data["q"], np.nan)
summary = json.loads((standard / "summary.json").read_text())

saved = sm.load_result("/tmp/simesh-quantitative/thermal.result.npz")
thermal_image = saved.result.image
mass = saved.metadata["mass"]
lines = sm.load_result("/tmp/simesh-quantitative/streamlines.result.npz").result
```

For optional PNG rendering without rerunning the physics:

```bash
.venv/bin/python -m pip install '.[plot]'
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-standard --render-only
```

The analytic MHD checks are mass `2.4e-12 kg`, mass-weighted temperature
`840000 K` and outward bottom flux `-2e-4 T*m^2`. Synthetic checks establish the
example workflow; they do not calibrate a real snapshot or validate extreme QSL.
For full controls and interpretation, see the
[standard example](standard-applications.md) and
[quantitative example](quantitative-workflow.md).

## Remaining output extensions

Direct typed persistence for reduction results and raw native Fields, automatic
capture of all calculation settings, magnetic-line VTK/PolyData export,
HDF5/Zarr products and multi-snapshot orchestration are not supplied by the
current result-file API. These are extensions to the present delivery surface;
they are not prerequisites for using the retained ordinary outputs or the
versioned application products.
