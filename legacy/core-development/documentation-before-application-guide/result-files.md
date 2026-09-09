# Saving application geometry and results

For the complete writer/reader matrix, example filenames and companion metadata
conventions, start with the [output guide](capabilities-and-outputs.md).
An arbitrary NumPy `.npz` bundle is not a versioned simesh result file.

Import the file API explicitly from `simesh.results_io`:

`save_result`, `load_result`, `ResultFile` and `ResultFileError` are also
available from top-level `simesh`.

```python
from simesh.results_io import save_result, load_result, ResultFileError

save_result("diagnostic.npz", diagnostic, metadata={"calculation": controls},
            source={"path": "snapshot.dat", "description": "caller-provided source"})
saved = load_result("diagnostic.npz")
restored = saved.result
image = restored.image("log10_q")
selected = restored.threshold(q_min=1e4)
```

`save_result(path, result, *, metadata=None, source=None, overwrite=False)`
returns the destination `Path`. `load_result(path)` returns a `ResultFile` with
`result`, `metadata`, `source`, `schema_version`, and `source_verification`.
The result has its original application type and owned, read-only arrays. No
Source, Fields, input snapshot, or numerical calculation is needed to load it.
Missing files and filesystem permission failures remain ordinary `OSError`
subclasses; malformed or unsupported files raise `ResultFileError`, a
`ValueError` subclass.

Published line and LOS results cannot contain RUNNING states. Successful LOS
values must be finite, empty LOS values must be zero, and valid Q must be positive
and not NaN (positive infinity remains allowed for overflow). These consistency
checks do not authenticate scientific data or recompute the original calculation.

## Supported objects

One file contains one of the following objects, including its nested geometry:

| Object | Preserved content and restored use |
| --- | --- |
| `PointSet` | Positions, int64 IDs, layout, optional normals and parent plane; `reshape` and `select` |
| `RaySet` | Identified origins, exact normalized directions, near/far intervals; `select` |
| `SampledPoints` | Points, component values, original leaf owners, coverage validity, ordered field definitions and units; `image`, `usable`, `select` |
| `ConnectivityMap` | Identified points and all `QSLResult` data; quantity images and threshold selection |
| `QSLResult` | Seeds, optional Q/Q-perpendicular/log-Q/twist, length, endpoints, endpoint fields, boundary/termination codes, steps, validity, completeness, stencil validity, method, normalization and local radius |
| `RayResult` | Rays, values, units, quadrature, entry/exit distances, status, sample/miss counts, and existing physical model metadata; `image`, `valid`, `select` |
| `LineSet` | Identified seeds and packed two-branch positions, offsets and termination; `branch(seed_id, direction)` and `line(seed_id)` |
| `LineProfiles` | Nested LineSet, arclength and length units, selected values/definitions/indices, owners, coverage/finiteness, boundary adjustments and scheme |
| `UniformResult` | Component values, coverage validity, lower/upper bounds and field definitions; `spacing`, cell-center `axes`, and `usable` |

A raw `QSLResult` has only row order and seed positions. The file does not invent
IDs or an image layout for it; save a `ConnectivityMap` when those associations
are needed. `Plane` is preserved as nested geometry, not a standalone product.
Other result types and subclasses are outside schema version 1.

In particular, raw `Fields`, `TraceResult`, `SliceResult`, `LOSResult`,
`ThermalLOSResult` and reduction dataclasses are not registered. Application
wrappers provide `LineSet`, `SampledPoints`, `RayResult` and `UniformResult` for
identified products. For reductions, explicitly serialize numerical arrays or
attach JSON-compatible summaries, units and coverage as caller metadata on a
supported product. This does not restore the reduction dataclass on load.

A sparse `PointSet` keeps its current `(selected_count,)` layout and its original
parent plane, even if that plane has a different point count. IDs are labels,
not necessarily pixel indices: retain the original full point set to map custom
IDs back to image pixels. `PointSet.iter_plane` uses flat parent-pixel IDs and
therefore already supplies that association.

Line offsets have `2*n+1` entries. Branch `2*i` is against B and `2*i+1` is along
B; each nonempty branch starts at its seed. Empty, unrequested, tangent and
outside-seed branches retain their explicit status. Saving does not join,
resample or expand paths. QSL/twist products contain no paths; tracing selected
points is still a separate operation, defaulting to both directions.

## Numerical meaning and provenance

Saving preserves float64 values, NaN/infinity, masks, integer labels and status
codes. Unit labels and field interpretations are retained without conversions.
Sample and uniform `valid` means coverage validity; `usable` also requires finite
components. Incomplete QSL/twist diagnostics and LOS failures remain incomplete.
Finite accepted-segment twist and Q-perpendicular values can coexist with an
incomplete central line; consult their status before scientific interpretation.
Q-only has `twist=None`; twist-only has all four Q arrays set to `None`. The
reader preserves these distinctions, including quantity-specific validity.

In-memory `source_identity` is deliberately omitted and restored as `None`.
It may include an object identity that cannot establish provenance in a later
process. Equality of restored `None` values establishes no relationship between
files. Never use it to certify that a restored map and a new Fields object have
the same source. The caller decides which explicitly prepared field to use for
later tracing.

`source` is an optional JSON object supplied by the caller. A path, dataset
identifier, or independently computed content digest can be recorded there.
`source_verification` is always `"unverified"` when a source is supplied, or
`"unspecified"` when it is absent. The reader does not open that path, calculate
hashes, or validate a claimed physical model. It rejects a file that claims
verified provenance in the schema's verification field.

`metadata` is a separate caller-supplied JSON object. It does not overwrite a
`RayResult`'s existing `metadata`, so thermal response identity, temperature
label, and density/length normalization remain available on that result.
Metadata may contain strings, booleans, Python integers, finite Python floats,
null, lists, tuples, and dictionaries with string keys. Tuples become lists.
Array metadata, NumPy scalar metadata, callbacks and arbitrary objects must be
converted explicitly. JSON nesting is limited to 64 levels.

Only parameters already present in the result are saved automatically. In
particular, explicitly record calculation controls (step size, max steps/length,
delta, bounds, reconstruction scheme), field/component definitions absent from
the result, coordinate units, unit scales and physical model assumptions. For
unlimited controls such as `max_length=np.inf`, record a finite JSON description
such as `{"max_length": "unlimited"}`. Do not pass native infinite defaults into
JSON metadata. This is a result file, not an executable recipe or solver restart.

## Complete diagnose, reload, select, trace example

Run with the independent package installed, using a fresh output directory.
The constant field and threshold below are an illustrative validation fixture.

```python
from pathlib import Path
import numpy as np
import simesh as sm
from simesh import applications as app
from simesh.results_io import save_result, load_result

out = Path("result-demo")
out.mkdir(exist_ok=True)
mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]),
                          lower=(0., 0., 0.), upper=(1., 1., 1.),
                          block_shape=(8, 8, 8))
values = np.zeros((1, 3, 8, 8, 8))
values[:, 2] = 1.
with sm.source_from_arrays(mesh, values, ("b1", "b2", "b3")) as source:
    magnetic = sm.prepare(source, scheme="exact-phase")
plane = sm.Plane([0., 0., .5], [1., 0., 0.], [0., 1., 0.], (4, 3))
points = sm.PointSet.from_plane(plane, ids=np.arange(100, 112, dtype=np.int64))
controls = {"quantities": ["q", "twist"], "step_fraction": .25, "max_steps": 1000}
source_description = {"kind": "analytic", "field": "B=(0,0,1)",
                      "field_units": "code", "coordinate_units": "code"}
diagnostic = app.connectivity(magnetic, points, **controls)
save_result(out / "map.npz", diagnostic, source=source_description,
            metadata={"calculation": controls, "preparation": "exact-phase"})

# This part can run in a new process without the original field arrays.
saved = load_result(out / "map.npz")
np.testing.assert_allclose(saved.result.image("q"), 2., atol=1e-10)
selected = saved.result.threshold(q_min=1.9)
save_result(out / "selected.npz", selected, source=saved.source,
            metadata={"selection": {"q_min": 1.9}})

# The caller explicitly chooses the prepared field for the new integration.
selected = load_result(out / "selected.npz").result
lines = app.trace(magnetic, selected, step=.025, max_steps=100)
save_result(out / "lines.npz", lines, source=source_description,
            metadata={"calculation": {"direction": "both", "step": .025,
                                      "max_steps": 100}})
restored_lines = load_result(out / "lines.npz").result
np.testing.assert_array_equal(restored_lines.seeds.ids, selected.ids)
polyline = restored_lines.line(100)
```

To save a loaded object again while retaining its caller metadata, pass
`loaded.result`, `metadata=loaded.metadata`, and `source=loaded.source` to
`save_result` explicitly. The envelope itself is not a supported result type.

## Version 1 representation and publication

The single file is a compressed NPZ archive containing ordinary NPY arrays:

- `__manifest__.npy` is a one-dimensional uint8 UTF-8 JSON document, at most 1 MiB.
- Its exact top-level members are `format` (`"simesh-result"`), `schema_version`
  (`1`), `result`, `metadata`, and `provenance`.
- `provenance` has `source` and `verification`. A result node has `type` plus
  exactly the stored attributes listed in the module's closed schemas. Array
  attributes are references to archive keys; nested geometry is another node.
  Optional absent attributes are JSON null, not object arrays.
- Arrays are little-endian float64, signed int64 or bool. Field definitions are
  ordered JSON records with `name`, `units` and `interpretation`; shapes are
  JSON integer lists. Array keys are `a` followed by at least four decimal
  digits; the writer uses `a0000`, `a0001`, etc.
- Each array is referenced once. Unknown or missing members, duplicate archive
  or JSON keys, wrong types/dtypes/shapes, inconsistent optional diagnostics,
  invalid IDs, packed offsets, branch seed associations, or status/mask
  inconsistencies are rejected. Schema upgrades require an explicit reader
  change; unknown versions are rejected rather than guessed.

Reads always use `allow_pickle=False`. No object deserialization, imports from
file content, or source reconstruction is permitted. Structural validation does
not recompute the science or authenticate caller provenance. Arrays containing
nonfinite diagnostic values remain allowed where the result contract uses them.

A temporary sibling file is fully written and closed before publication, using
the same link-or-replace protocol as native `write_amrvac`. The default refuses
an existing destination, including symlinks or a file created concurrently.
`overwrite=True` replaces the destination only after serialization succeeds.
Serialization/publication failures clean up the temporary file and leave any
previous destination intact. Paths are used exactly; no `.npz` suffix is added.
The parent directory must already exist. This protocol requires filesystem
hard-link/atomic-replace support; it does not promise crash-durable storage.

The writer snapshots arrays and validates the serialized representation; the
reader loads owned arrays and closes the archive before returning. This format
may need multiple resident copies of a result. It has no memory-map, streaming,
append, or bounded-memory guarantee. HDF5/Zarr, ParaView, time-series management,
source field storage and automatic recomputation are outside this API.

For a standalone `LineProfiles` save/load example, see
[profile persistence](line-profiles.md#save-restore-or-deliver-by-shard).
For incremental seed-based line/profile delivery and checksummed
incomplete/complete manifests, see [selected delivery](selected-delivery.md).
