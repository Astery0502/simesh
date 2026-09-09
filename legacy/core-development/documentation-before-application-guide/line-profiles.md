# Physical quantities along stored curves

`simesh.line_profiles.sample_line_profiles` samples completed native `Fields`
on an existing `LineSet`. It never traces, interpolates new curve positions, or
reconstructs AMR support. It uses the native field sampler, with the same spatial
interpolation and coverage rules. Import this feature from its focused module
or from the top-level `simesh` namespace.
`LineProfiles` can be saved with `save_result` and restored with `load_result`,
including its nested curve geometry. See [result files](result-files.md) and the
[output guide](capabilities-and-outputs.md) for supported products and readers.

## Diagnose, select, trace, then sample

```python
import numpy as np
import simesh as sm
from simesh import applications as app
from simesh.line_profiles import sample_line_profiles

limit = 2 * 1024**3
with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    magnetic = sm.prepare(source, ("b1", "b2", "b3"),
                          scheme="coordinate-phase", memory_limit=limit)
    # If these quantities are stored in the snapshot, select their actual names.
    quantities = sm.prepare(source, ("temperature", "rho"),
                            scheme="coordinate-phase", memory_limit=limit)

seeds = sm.PointSet.boundary(magnetic.mesh, "zmin", (32, 32),
                            ids=1000 + 3*np.arange(32*32))
diagnostics = app.connectivity(magnetic, seeds, quantities=("q", "twist"))
# Illustrative thresholds; choose criteria for the actual physical problem.
selected = diagnostics.threshold(q_min=1e4, abs_twist_min=1., mode="any")
# Diagnostic Q/twist results do not save paths. This separate call defaults to
# both directions and retains the accepted prefix and its termination status.
lines = app.trace(magnetic, selected,
                  step=float(magnetic.mesh.spacing.min())*.125,
                  max_steps=4000, workers=4, memory_limit=limit)
profiles = sample_line_profiles(
    quantities, lines, ("temperature", "rho"),
    point_batch=4096, workers=4, memory_limit=limit,
    # Explicit example only: one coordinate length represents 1e6 metres.
    length_units=sm.LengthUnits(1e6, "m"),
)
if len(profiles.seed_ids):
    line = profiles.line(profiles.seed_ids[0], memory_limit=limit)
    distance_m = line.arclength
    temperature = line.values[:, 0]
    density = line.values[:, 1]
    temperature_usable = line.usable[:, 0]
    density_usable = line.usable[:, 1]
    trace_status = line.termination  # against/along, independent of sampling
```

This does not infer thermodynamics from an energy column or convert field unit
labels. If temperature is not stored, explicitly recover it using the
[MHD interface](mhd-thermodynamics.md), or provide a separately defined physical
model. For example, `sm.mhd_fields(prepared_state, model=model,
outputs=("temperature", "density"))` produces a group suitable for this consumer
when the input has valid interpolation support. The two columns may have different
units. Current density, scalar derivatives and other prepared/derived quantities
can be sampled in the same way. Do not interpolate categorical status columns.

The function accepts one `Fields` group with the required columns. Select all
columns with `components=None`, one with a name or integer, or multiple ordered
columns with a sequence of names/indices. Mixed names and indices are allowed;
missing or ambiguous names, invalid indices, duplicates and empty selections
are rejected. A single selection still has shape `(point_count, 1)`.

## Packed geometry and point validity

`LineProfiles` retains its read-only `lines` geometry and independent read-only
arrays for the sampled product. No point is removed:

| Attribute | Meaning |
| --- | --- |
| `seed_ids` | Original unique int64 seed labels, not row indices |
| `offsets` | Original `2*n+1` offsets into every packed point array |
| `termination` | Original `(n,2)` trace statuses, including unrequested/tangent branches |
| `arclength` | `(p,)` cumulative geometric distance, restarting at zero for each branch |
| `values` | `(p,k)` sampled values in requested component order |
| `definitions` | Selected `FieldDefinition` objects, including individual units and interpretation |
| `component_indices` | Selected column positions in the supplied `Fields` group |
| `owners` | Original mesh leaf per sample, or `-1` outside the native sampling domain |
| `valid` | `(p,)` native coordinate/coverage validity |
| `finite` | `(p,k)` finiteness of each returned component |
| `usable` | `(p,k)` computed mask combining `valid` and `finite` |
| `boundary_adjusted` | `(p,)` records exact upper-face sampling adjustments |

For seed row `i`, branch `2*i` is against the orientation and branch `2*i+1`
is along it. Each starts at its seed and proceeds to its stored endpoint. Empty
branches retain repeated offsets and their termination status. Empty selections
produce correctly shaped empty arrays. `MAX_STEPS`, `MAX_LENGTH`, missing coverage,
null fields and other trace outcomes remain visible through `termination`.
Sampling success never implies a complete magnetic line. This interface does
not compute a new trace-completion classification.

Within an available owner leaf, a NaN/Inf component can have `valid=True` and
`finite=False`, while another component at the same point remains usable. Missing
prepared coverage has `valid=False` and NaN values, but retains its original owner
leaf; genuinely outside points have owner `-1`. Neither a finite value nor
`usable=True` asserts physical admissibility, positivity or trace completeness.

```python
branch = profiles.branch(seed_id, -1)  # seed-to-endpoint view
joined = profiles.line(seed_id)       # owned display oriented along the curve
packed_rows = joined.point_indices
assert np.array_equal(joined.positions, profiles.lines.positions[packed_rows])
```

Branch arrays generally share the parent result; `point_indices` and `directions`
are allocated. The joined display reverses the negative branch and then appends
the positive branch. Its `directions` is `-1` or `+1` per point and `point_indices`
maps every display row back to the original arrays. Both accessors preserve the
two termination statuses for that seed and accept an optional `memory_limit`.
They also retain field definitions, length conversion and both provenance tokens.

## Distance, orientation and duplicate seeds

Distances are cumulative Euclidean lengths of the **stored polyline**, including
zero-length repeated segments. They are not an exact ODE arclength, the tracer's
length accumulator, a local-QSL endpoint distance, or an elapsed time. Resolution
and accepted endpoints are properties of the original trace. No interpolation
across omitted/missing curve segments is inferred.

- In packed storage and branch views, each nonempty branch starts at distance
  zero and increases away from its seed.
- In a joined display, the reversed negative branch uses `-arclength`; the
  positive branch uses `+arclength`. Values thus proceed from negative to positive
  distance in the along direction. A negative-only line ends at zero; a
  positive-only line begins at zero.
- **Both stored seed copies remain in the joined profile**, each at distance
  zero and with its own direction and point index. This intentionally differs
  from `LineSet.line`, which omits one seed. No samples or masks disappear.
  Any later plotting-only deduplication must be an explicit caller choice.
- Separate lines at coincident seed coordinates keep their distinct seed IDs.

Without `length_units`, distances use `LengthUnits(1., "coordinate-length")`.
An explicit `LengthUnits(scale, unit)` supplies an isotropic conversion:
`distance = polyline_coordinate_length * scale`. Positions and seed coordinates
always remain in the original mesh coordinates. Different component unit labels
are preserved unchanged. Unrepresentable distance sums or conversions raise
`ValueError` rather than returning misleading finite lengths.

## Physical boundary convention

The native sampler owns a half-open physical box `[lower, upper)`. The default
`boundary="interior"` allows exact upper-face coordinates retained by application
tracing to be sampled: only if the entire point is inside the **closed** physical
box, coordinates exactly equal to an upper bound are replaced in a temporary
sampling copy by `np.nextafter(upper, lower)`. The original positions, distance
calculation and offsets remain untouched; `boundary_adjusted` records the point.
Lower-face coordinates are sampled directly. Corners may adjust multiple axes.

No tolerance, arbitrary clipping, or regional-coverage adjustment is applied.
Even a point one representable value outside the physical domain stays outside.
An exact face in a missing prepared leaf remains a missing-coverage sample.
The sampled boundary value is the native interpolation at that nearest interior
point, including the chosen preparation's ghost/boundary behavior; it is not a
separate analytic boundary extrapolation.

Use `boundary="native"` to sample unchanged coordinates with the native half-open
rule. Then exact upper-face points are invalid and no adjustment flags are set.

## Other supplied curves and provenance

Use the existing `LineSet` representation for caller-supplied finite curves:

```python
points = np.array([[.5, .5, .5], [.6, .5, .5], [.7, .6, .5]])
curves = sm.LineSet(
    seeds=sm.PointSet(points[:1], ids=np.array([72])),
    positions=points.copy(),
    offsets=np.array([0, 0, len(points)], dtype=np.int64),
    # Caller convention: no negative branch; supplied positive curve is a prefix.
    termination=np.array([[sm.LineSet.NOT_REQUESTED, sm.Termination.MAX_LENGTH]],
                         dtype=np.int64),
    source_identity=None,
)
profile = sample_line_profiles(quantities, curves, "temperature")
```

Each nonempty branch must start exactly at its corresponding seed; malformed
input is rejected. Offsets and statuses follow `LineSet`'s existing contract.
For nonmagnetic curves, the negative/positive signs describe the supplied
orientation, not a claim about B; assign and interpret statuses explicitly.

`source_identity` is the sampled `Fields.value_identity`; `line_source_identity`
is the original `LineSet.source_identity`. They are independent: a density or
temperature field should not have the magnetic field's value identity. `scheme`
records the sampled preparation/derivation scheme. These are in-process logical
tokens, not payload hashes or file provenance.

`LineSet` does not carry a mesh or time descriptor, so it cannot establish mesh
identity, spatial-unit agreement, common time, or cross-source physical
compatibility. Passing another field/source is an explicit caller choice. The
consumer validates completed `Fields`, active borrowed lifetime and at least one
`valid_halo` layer; allocated `storage_halo` padding alone cannot satisfy it.
It neither reads nor retains a Source or Fields. Results survive source closure
and expiration of a prepared batch; the borrowed field must remain active until
the synchronous call returns, and must not be advanced/closed concurrently.

## Memory and execution

`point_batch` bounds coordinate, segment and native sampling scratch. One
thread-pool context is reused for all batches; `workers` defaults to one.
Batch size and worker count do not change points, component order, or distance
arithmetic. The native sampler evaluates only selected components and writes
batches directly into final result arrays.

`memory_limit` admits mesh and input field arrays, shared line geometry, all
retained output arrays, and a conservative batch-scratch estimate before
allocating outputs. Full output size grows with stored point count. Smaller
batches cannot make an oversized retained result fit. No geometry, component,
coverage, or precision is silently reduced to meet the budget.

This is a controlled-array estimate, not a process-RSS cap. Other retained
products, Python objects and external allocations remain the caller's
responsibility. The computed `usable` property allocates a new mask; explicit
branch/joined displays have their own optional admission check. Plotting and
field composition are separate consumers; the tracing algorithm is unchanged.

## Save, restore, or deliver by shard

```python
sm.save_result("profiles.result.npz", profiles,
               metadata={"description": "Physical quantities on selected curves"})
restored = sm.load_result("profiles.result.npz").result
np.testing.assert_array_equal(restored.values, profiles.values)
np.testing.assert_array_equal(restored.seed_ids, profiles.seed_ids)
```

The file retains geometry, component definitions, arclength units, masks,
boundary adjustments and scheme. Both in-memory identity tokens are unset on
load; the caller establishes compatibility with any later input snapshot.
Individual `LineProfile` branch/joined views are not standalone result-file
types; save their parent `LineProfiles`.

`iter_line_profiles` consumes LineSet shards independently. Combine it with
`save_result_shards` and load one shard at a time with `open_result_shards`.
This bounds retained output by shard when the caller releases each batch;
input Fields still need to be supplied. See
[selected delivery](selected-delivery.md) for the complete workflow.
