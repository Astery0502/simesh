# Experimental Native Analysis Usage

Use [current](current.md) for the exact delivered stage and unresolved gates.
Canonical `simesh.amrvac` APIs remain available. The new numerical kernels build
with the standard setuptools discovery under `src/simesh/utils/lib/analysis`:

```sh
.venv/bin/python -m pip install -e .
# Subsequent native consumer kernel iteration:
.venv/bin/python build.py --inplace --group analysis
```

The core has no implicit file/cache provider. Full builds explicitly bundle the
retained `simesh_rewrite` provider; installed workflows need no manual PYTHONPATH.
The old development adapter now forwards to `simesh.analysis.providers`.
The new file-source profile reads ordinary fields from supported 3D v5 records,
including validated staggered tails. The original DAT-003 factory still rejects
staggered files. Source arrays/file descriptors remain immutable/alive during
preparation; CT/staggered computation is not enabled.

```python
from simesh.analysis import open_source, open_prepared, PreparedPool, trace

with open_source("snapshot.dat", field_names=["b1", "b2", "b3"]) as source:
    pool = PreparedPool(source, [0, 1, 2], capacity=256)
    try:
        lines = trace(pool, seeds, step=0.001, max_steps=1000, workers=4)
    finally:
        pool.close()

resident = open_prepared("snapshot.dat", field_names=["b1", "b2", "b3"])
```

The source context closes its owned file; detached products remain usable.
`field_indices` alternatively selects original file positions. Selected fields
become source columns 0..K-1; inspect `original_field_ids` for the file mapping.
`field_units` can declare a unit string or a name-to-unit mapping; it does not
convert values or infer EOS. The default label is code units. File analysis
currently requires nonperiodic Cartesian 3D v5 and even blocks of at least four.
Canonical APIs remain the route for their supported 2D and other file workflows.

## Prepared Fields And Repeated Sampling

Given an assembled `FieldSource` named `source`:

```python
import numpy as np
from simesh.analysis import prepare, PreparedPool, sample

selected = np.array([7, 2, 11], dtype=np.int64)
magnetic = prepare(source, selected, [0, 1, 2], budget_bytes=2 * 1024**3)
values, owners, valid = sample(magnetic, points)
```

Field IDs select source columns; their physical names, units and cell-average
interpretation are declared in `FieldDefinition`. `prepare` preserves caller
leaf and field order and returns detached readonly component-adjacent values,
geometry, two valid halo layers and an explicit `slot_of_leaf` directory.
Outside/missing/nonfinite coordinates return invalid/NaN samples. Sampling
never performs implicit preparation. Source closure does not revoke a detached
product. A field group does not concatenate/rebuild another group's backing.
Pass `halo=0` for an interior-only native product; this performs no ghost work.

```python
pool = PreparedPool(source, [0, 1, 2], capacity=256)
try:
    with pool.borrow(selected) as fields:
        values, owners, valid = sample(fields, points)
        block = fields.window(7, [0, 0, 0], source.mesh.block_shape)
        # Consume block synchronously here; it expires at context exit.
finally:
    pool.close()
```

One coordinator lease freezes the pool. Workers share its readonly views;
nested borrows, clear and close are rejected until return. Pool arrays are not
retained products. `interior()` returns a direct view for packed owned products;
nonpacked borrowed slots require per-leaf windows, avoiding an implicit gather.
The same scope applies to `iter_prepared` batches, which expire on advance and
can be synchronously written to an array/memmap sink.

## Magnetic Lines

```python
from simesh.analysis import trace, iter_traces

pool = PreparedPool(source, [0, 1, 2], capacity=256)
result = trace(pool, seeds, step=0.001, max_steps=1000, workers=4)
for chunk in iter_traces(pool, seeds, step=0.001, max_steps=1000,
                         workers=4, seed_batch=128):
    write_summary(chunk)  # Caller-supplied sink; no mandatory concatenation.
pool.close()
```

Seeds are finite contiguous float64 `(n,3)` with optional stable unique int64 IDs.
Choose step/error/length limits for the scientific request; seed spacing is a
separate quantity. The tangent is signed normalized B and the accumulator is
accepted arclength. Results include last-accepted positions, length, steps,
termination reasons and miss/sample counts. `localized_endpoint=False` means
an exiting trial was rejected, not an exactly localized footpoint. Half-open
ownership accepts lower-domain faces; upper-face inward launches require a later
profile. Null fields, nonfinite fields, unrepresentable norms and missing supplied
coverage are distinct. Private RK stages survive bounded-pool misses.

Use one worker for tiny jobs. The measured 2048x600 prepared workload benefited
from four workers; the 8x8 WENO case did not. Cache capacity depends on actual
visited owners: the recorded long repeated WENO request fits 256 slots and
thrashes at 64. Those are measured case decisions, not universal defaults.

## Global Derivatives And Retained Slices

```python
from simesh.analysis import global_curl, Plane, sample_plane

current_like = global_curl(source, field_ids=[0, 1, 2], batch_size=256)
plane = Plane(origin, full_width_vector, full_height_vector, (256, 256))
first = sample_plane(current_like, plane)
second = sample_plane(current_like, moved_plane)
```

This calculates curl(B) on every physical leaf plus one remaining valid halo
before any slice. It does not infer current normalization. The global output is
independent of temporary primary slots; subsequent slices use that retained
result without recomputing curl. Pass a writable contiguous float64 array or
memmap as `output` to choose backing explicitly. Keep supplied backing unchanged
while its product is in use. A failed global call returns no certified product;
an explicit sink can contain earlier completed batches.

Plane pixels sample `origin + u*t + v*s` at uniform pixel-center fractions.
The spans need not be axis-aligned. This is a sampled plane, not an area average
or native intersection. Invalid/outside pixels carry false validity and NaN.

When the fields fit, `simesh.amrvac.analysis.prepare_resident` adapts canonical
bulk ghost preparation directly to the new field-group boundary. It takes the
validated mesh index/root shape/forest flags plus canonical field-major interiors
and definitions. Its named coordinate-phase transfer differs slightly from the
bounded exact-phase provider; all retained C backing, including coarse workspace,
is kept alive and counted. Compute `curl(resident)` for an independent derived
group. Source interiors can be released after resident preparation.

All admission figures describe controlled arrays, including relevant source,
metadata, scratch and output backing. They are not hard process RSS/page-cache
limits. Large-file, installed-adapter and additional consumer acceptance must be
read from the current checkpoint, not inferred from these examples.

## Twist And Explicit Retracing

```python
from simesh.analysis import CurlPool, retrace

base_pool = PreparedPool(source, [0, 1, 2], capacity=256)
magnetic_and_curl = CurlPool(base_pool)
try:
    diagnostics = trace(magnetic_and_curl, seeds, step=0.001, max_steps=1000,
                        twist=True, workers=4)
    chosen = diagnostics.seed_ids[np.argsort(np.abs(diagnostics.twist))[-3:]]
    selected_lines = retrace(magnetic_and_curl, diagnostics, chosen,
                             step=0.001, max_steps=1000, twist=True, workers=4)
finally:
    magnetic_and_curl.close()
    base_pool.close()
```

The companion owns separate one-halo curl storage and borrows the primary pool.
Closing it releases that companion. `with_curl(primary)` is the corresponding
retained resident composition. Twist uses stage quadrature over accepted segments;
it does not imply exact boundary endpoints or Q. Set `trajectories=True` on an
ordinary `trace`/`iter_traces` call when points are requested from the start.
Use `point_counts` to read valid trajectory prefixes. Full output and retained
prior results count against admission; streaming avoids mandatory concatenation.

## LOS Of A Supplied Scalar

```python
from simesh.analysis import integrate_los, orthographic_plane

view = orthographic_plane(scalar.mesh.lower, scalar.mesh.upper, [0.3, 0.2, 1.0],
                          (256, 256))
image = integrate_los(scalar, view, [0.3, 0.2, 1.0], workers=4)
```

Supply an explicitly defined scalar/emissivity group or PreparedPool. The default
Gauss2 method integrates its trilinear reconstruction between cell-center planes.
`near`/`far` may be scalars or image-shaped arclength limits. `image.depth`,
`image.valid` and `image.complete` distinguish real physical intervals and
failed pixels; empty rays are zero. Midpoint quadrature is available explicitly
with `quadrature="midpoint", step_fraction=...`. Plane pixels are sampled rays,
not area averages. The returned units are scalar units times coordinate length.

This interface does not infer epsilon(rho,T), EOS, temperature or instrument
response. The recorded WENO rho-column image is a scalar diagnostic; the thermal
response gate remains open in the current checkpoint.

## Bounded Uniform Output

```python
from simesh.analysis import iter_uniform

for z_index, slab in iter_uniform(resident, (1000, 1000, 1000)):
    write_slab(z_index, slab.values)  # (nx, ny, component); caller-supplied sink.
```

Uniform cell centers are sampled one z slab at a time. The iterator also accepts
a PreparedPool for bounded source access. It never constructs a full coordinate
cube or concatenates the full output. Each returned slab is owned; arbitrary
additional retention is the caller's explicit memory choice. The active budget
reserves a current and one normally retained previous slab.

## Historical AIA 171 Thermal Synthesis

The first explicit optically thin model is `AIA171`. Its source table is pinned
MPI-AMRVAC data; its original calibration/abundance generation settings are
undocumented. It is a reproducible historical model, not a current observing-date
calibration. `CoronalComposition` uses fully ionized H/He with n_He/n_H=0.1.
The default emissivity is n_e^2 R(T); `AIA171("amrvac-hydrogen")` explicitly uses
the upstream default hydrogen-density normalization. See the thermal evidence
for the resulting 1.44 brightness factor and physical validation limits.

```python
from simesh.analysis import (
    open_prepared, thermal_fields, integrate_thermal_los, orthographic_plane,
)

rho = open_prepared("snapshot.dat", field_names="rho")
# Explicit demonstration units and external isothermal input. These values
# must be replaced by the simulation's documented normalization for real use.
state = thermal_fields(
    rho, 1.0e6,
    density_unit_g_cm3=2.341670693166e-15,
    temperature_label="isothermal demonstration: 1 MK; not measured snapshot T",
)
plane = orthographic_plane(state.mesh.lower, state.mesh.upper, [0, 0, 1], (64, 64))
image = integrate_thermal_los(state, plane, [0, 0, 1], length_unit_cm=1.0e8)
assert image.complete
print(image.scalar_units, image.temperature_label)  # DN s^-1 pixel^-1, explicit input label
```

`density_unit_g_cm3` multiplies stored rho; `length_unit_cm` multiplies every LOS
path element, not pixel width or area. `image.depth` remains in source coordinates;
`image.depth_cm` exposes physical depth. `ThermalLOSResult` retains model identity,
input label and both normalizations. DN are detector counts, not erg; no extra
4*pi, observer-distance factor, exposure time, PSF or pixel-area averaging is
applied. Samples report native AIA-pixel-normalized brightness at the requested
ray centers, irrespective of the numerical raster's spacing.

For an external nonuniform temperature, prepare a `PreparedFields` scalar with
units `K`, the identical `MeshIndex` and coverage of all requested density leaves.
Pass that product as `temperature`; independently permuted packed leaf order is
supported. Native external arrays can use `prepare_resident(mesh, mesh.roots.shape,
mesh.node_leaves >= 0, temperature_interiors, (FieldDefinition("T", "K"),))`, where
`temperature_interiors` is float64 `(nleaf, 1, bx, by, bz)` in the density mesh's
source leaf order. This performs the same two-halo preparation, and does not
infer physical registration from array shape. A combined rho/T source can also
prepare separate groups on the same mesh. For thermal pressure input,
`CoronalComposition.temperature(rho_cgs, p_thermal_cgs)` implements the explicit
ideal gas EOS; total energy requires the caller's actual kinetic/magnetic/
background-field model before thermal pressure can be supplied. Missing T raises
an error; density alone never selects temperature.

The default order is `thermodynamics-first`: interpolate number density and T,
then evaluate the nonlinear response. `subdivisions=4` applies composite Gauss2
after splitting at thermodynamic interpolation knots; it is an approximation.
Check refinement for a scientific request, especially at response-table knots.
`order="emissivity-first"` evaluates the response at every prepared primary node,
then integrates that scalar interpolant. This differs from preparing halos of
interior emissivity, since nonlinear response and AMR transfer do not commute.

For repeated views of the node-emissivity model, call `emissivity_fields(state)`
once, retain that product and use `integrate_los` repeatedly, multiplying its
values by the explicit cm-per-coordinate multiplier. The scalar consumer returns
its usual scalar-times-coordinate units; retain the thermal input provenance
alongside that lower-level result. Node-emissivity scalar Gauss2 exactness does
not establish accuracy with respect to nonlinear thermodynamic reconstruction.
Both options are currently resident. The default nonlinear implementation now
uses compiled AMR-tree traversal; `workers=1`, `2` or `4` selects independent
GIL-free ray workers. `implementation="reference"` keeps the earlier Python
all-leaf implementation and requires one worker. These have the same response,
reconstruction and subdivision count, with tolerance-level agreement and exact
native serial/parallel equality. The main session's numerical cache and broader
execution backends remain separate. A bounded response provider is still open.

```python
image = integrate_thermal_los(
    state, plane, [0.3, 0.2, 1.0], length_unit_cm=1.0e8,
    subdivisions=4, workers=4,
)
reference = integrate_thermal_los(
    state, plane, [0.3, 0.2, 1.0], length_unit_cm=1.0e8,
    subdivisions=4, implementation="reference",
)
```

The Python thread-pool startup is included in each call. Small images may not
benefit from more workers; choose using the actual request's measured cost.
The native kernel advances complete ray intervals and counts samples actually
computed. A limit failure always invalidates the pixel; partial counter values
can differ from the reference, which admits a whole leaf's quadrature first.
Native response supports the built-in AIA171 model; custom response subclasses
must use the explicit reference path until given a compiled implementation.
See [thermal ray evidence](evidence/thermal-rays.md) for source comparisons,
acceptance, actual 500x500 images and 1/2/4-worker timings. Desktop memory pressure
made four workers slower than two in the axis case, so this is an explicit caller
choice, not an automatic maximum-worker setting. Rebuild both analysis extensions
after updating the shared native.pxd interpolation definition.

## Optional Retained Geometric Fill Plan

```python
import numpy as np
from simesh.analysis import open_source, build_fill_plan

with open_source("snapshot.dat", field_names=["b1", "b2", "b3"], support_capacity=256) as source:
    selected = np.arange(min(256, source.mesh.leaf_count), dtype=np.int64)
    plan = build_fill_plan(source, selected, capacity=256)
    b = plan.prepare(source, [0, 1, 2])
    b1 = plan.prepare(source, [0])  # geometry reused; fresh reads and values
```

The plan retains geometry only and makes no runtime cache-retention decision.
Its chunk-local support ordinals bind to private execution arrays, never external
cache slots. Values, minmod slopes and physical numerical results are recomputed.
Successful products are detached; failure returns no product and cannot poison
another execution. A different mesh object, source-cell order, geometry, halo
width, boundary/transfer strategy, selected leaf set or partition needs a new
plan. Field count/order or a new immutable value lifecycle on the identical mesh
can reuse geometry. Plan construction and retained bytes must amortize over the
actual repeated request; one-shot callers retain the ordinary `prepare` path.
The initial plan supports only the source provider's existing two-layer,
continuous, exact-phase strategy. It is not a generic sparse linear matrix:
limited prolongation is nonlinear. The main session owns scheduling and numerical
slot caches; they may hold a plan reference and bind private run storage later.
