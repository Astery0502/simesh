# Next-generation simesh

This is an independent, experimental native AMR analysis package. Its first
usable profiles provide immutable file/array sources, exact-phase regional and
coordinate-phase full-domain preparation, sampling, derivatives/curl, slices,
RK4 magnetic tracing with twist, scalar/thermal LOS, reusable geometry plans and
explicit bounded or streamed workflows.
It also bundles the stateful AMRVAC and array-tool compatibility workflows;
see [MIGRATION.md](MIGRATION.md) for their separate data and execution boundaries.
The N4 core design is fixed. New work can focus on applications consuming these
interfaces; use the [application guide](../docs/analysis-core/application-development.md).

For identified point/ray sets, diagnostic-map selection and separately traced
compact magnetic lines, use the optional `simesh.applications` namespace.
See [application interfaces](docs/applications.md). QSL/twist calculations do
not save trajectories; selected points are explicitly traced afterward.

Standard helpers provide magnitude, gradient, divergence, dot products and
explicitly normalized magnetic current/pressure/energy. Surface diagnostics can
compute Q, twist or both; twist-only requests skip Q work. See
[standard applications and the complete example](docs/standard-applications.md).

Quantitative workflows now include [native AMR integrals/statistics](docs/reductions.md),
[explicit ideal-MHD recovery](docs/mhd-thermodynamics.md), and
[versioned application result files](docs/result-files.md). These APIs are
available through `simesh` as well as their focused modules. The
[combined workflow](docs/quantitative-workflow.md) recovers a state, reduces its
interiors, generates thermal LOS and streamlines, and reloads saved products.

[Field composition](docs/field-composition.md) adds owned component selection,
leaf-aligned merging and multi-output pointwise recipes. Use
[line profiles](docs/line-profiles.md) to sample these quantities along stored
curves with seed IDs, branch distances and per-component validity. Both APIs
are available through `simesh`; the combined workflow also demonstrates them.

The package imports as `simesh`. Install it in its own environment: the existing
package in the parent repository uses the same import name. Runtime code and
build inputs live entirely in this directory; no historical `simesh_rewrite`
package, parent source path, or donor worktree is needed after installation.

## Install and develop

Python 3.11 or newer, NumPy and Cython are required. From this directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
make build
make test
```

Ordinary builds need no OpenMP runtime. `SIMESH_OPENMP=1 make build` enables the
optional native build configuration. Full-domain preparation accepts explicit
OpenMP; tracing and scalar/thermal LOS also accept `backend="openmp"`.
Portable thread pools remain the default, with one worker. Numerical primitives retain their original
compiler semantics separately from the optimized analysis kernels.
The retained compatibility AMR extensions stay serial, including in OpenMP
builds; native scientific kernels keep their explicit parallel options.

## Prepare once and reuse

```python
import numpy as np
import simesh as sm

limit = 2 * 1024**3
with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    lower, upper = source.mesh.lower, source.mesh.upper
    width = upper - lower
    magnetic = sm.prepare(
        source,
        fields=("b1", "b2", "b3"),
        region=(lower + .4 * width, lower + .6 * width),
        scheme="exact-phase",
        memory_limit=limit,
    )

# The file is closed. These operations only use the completed field group.
curl_b = sm.curl(magnetic, workers=4, memory_limit=limit)
plane = sm.Plane(lower + width * [.44, .44, .5],
                 width * [.12, 0, 0], width * [0, .12, 0], (48, 40))
section = sm.sample_plane(curl_b, plane, workers=4, memory_limit=limit)
seeds = np.ascontiguousarray(((lower + upper) / 2)[None, :])
lines = sm.trace(magnetic, seeds, step=float(.125 * magnetic.mesh.spacing.min()),
                 max_steps=64, trajectories=True, workers=4, memory_limit=limit)
```

The region selects all intersecting complete leaves and reads their necessary
support from the original mesh. Actual coverage is the union of
`magnetic.mesh.bounds[magnetic.leaf_ids]`; the requested box is retained in
`magnetic.selection.requested_bounds`. A region edge is not a physical boundary.
Missing supplied coverage yields invalid samples or `MISSING_COVERAGE` tracing
status. An exiting RK proposal is rejected; endpoints are last accepted points,
not exactly localized boundary footpoints.

Preparation explicitly selects its numerical scheme. `exact-phase` uses
`rewrite-ratio2-minmod-exactphase-cont-v1`; full-domain `coordinate-phase` uses
`canonical-coordinatephase-cont-v2`. Region or worker changes never silently
select another scheme. The schemes are not promised bitwise equivalent.

## Whole-domain preparation

```python
with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    whole = sm.prepare(source, ("b1", "b2", "b3"), scheme="coordinate-phase",
                       workers=4, memory_limit=limit)
```

This path uses contiguous record reads, then loads an independently owned final
NumPy array. Raw interiors are released before allocating coarse exchange
scratch. The published result retains neither the exchange bindings, their
temporary geometry nor coarse buffers, and publication performs no full-array
copy. Local ghost stages share one thread pool; cross-target transfers keep
their validated serial order. `backend="openmp"` is explicit and requires an
OpenMP build.

Coordinate preparation requires complete source coverage. Requests whose selected
leaves do not cover the source are rejected instead of being silently expanded
or switched to exact-phase. If a
caller supplies a permutation of all leaves, Selection retains that order while
physical slots remain in SFC order; use per-leaf windows rather than assuming
`values` follows the requested leaf ordering. `interior()` requires packed order.
The `support_capacity` tuning option belongs to exact-phase preparation.

The v2 coordinate profile corrects an inherited support-table overrun when a
coarse block spans both physical sides of a transverse axis. It preserves the
existing coordinate arithmetic and fills both sides correctly. The old v1
constant-field error is not retained as a compatibility mode.

## Arrays and interior fields

```python
mesh = sm.mesh_from_forest(
    (1, 1, 1), np.array([True]),
    lower=(0., 0., 0.), upper=(1., 1., 1.), block_shape=(8, 8, 8),
)
values = np.zeros((1, 3, 8, 8, 8), dtype=np.float64)
values[:, 0] = 1.
with sm.source_from_arrays(mesh, values, ("b1", "b2", "b3")) as source:
    raw = sm.read_fields(source, ("b3", "b1"))
    ready = sm.prepare(source, scheme="exact-phase")
```

Array sources accept contiguous native float64
`(leaf, component, x, y, z)` values. The default copies the input; with
`copy=False`, callers must keep it unchanged throughout the source lifetime.
File sources expose stable original field positions; field names or integer
positions are resolved when calling `read_fields` or `prepare`, preserving the
requested order. Reading interior fields does not construct halo workspaces.

`Fields.values` uses readonly component-adjacent
`(slot, x, y, z, component)` storage. `storage_halo` locates interiors, while
`valid_halo` independently describes completed support. Primary preparation
completes two layers; a centered derivative consumes one. Allocated invalid
padding cannot satisfy a stencil. `interior()` and `window(...)` expose native
views; derived groups own independent storage and do not rebuild their inputs.

Names, unit labels and cell-average/derived interpretation describe fields.
They do not automatically convert units, recover thermodynamics, or turn
`curl(B)` into physically normalized current.

## Custom pointwise fields and named derivatives

`derive` evaluates an explicit pointwise recipe and returns an independently
owned `Fields`, ready for the usual scientific consumers:

```python
b_squared = sm.derive(
    magnetic, "b_squared",
    lambda ctx: sum(ctx.field(name)**2 for name in ("b1", "b2", "b3")),
    units="code^2",
)
gradient_x = sm.derivative(
    b_squared,
    terms=[[("b_squared", "x", 1.)]],
    definitions=[sm.FieldDefinition("gradient_x", "code^2 / coordinate-length",
                                    "centered-derivative")],
)
combined = sm.derive(
    {"magnetic": magnetic, "curl": curl_b}, "b_dot_curl_b",
    lambda ctx: sum(
        ctx.field(b, group="magnetic") * ctx.field(c, group="curl")
        for b, c in zip(("b1", "b2", "b3"), ("curl_x", "curl_y", "curl_z"))
    ),
    units="code^2 / coordinate-length",
)
```

Derivative terms accept either component indices or unique names, and axes
`0`, `1`, `2` or `"x"`, `"y"`, `"z"`. Each output is a sum of weighted
directional derivatives; supply multiple term lists and definitions for multiple
outputs. A derivative consumes one valid halo layer.

Recipes run once per leaf on read-only arrays containing interiors and the
common valid halo. Return a scalar constant or a same-shaped array. Formulas
must be pointwise: spatial shifts, `np.gradient`, spatial reductions and other
stencils belong outside this callback contract. Use `derivative` for spatial
operations. Nonlinear formulas are evaluated on prepared values, including
their support; applying a nonlinear formula before preparation is a different
reconstruction. No automatic thermodynamic or unit conversion is performed.

Multiple inputs must share the same Mesh object and leaf coverage, although
their slot and leaf ordering may differ. The recipe explicitly chooses its
input groups; callers ensure compatible physical meanings, times and schemes.
The result uses the first group's selection order and the minimum valid halo
of all supplied groups. Zero-halo results support interior access, but need
explicit preparation through a suitable source before stencil consumption.
Only pass groups needed by the recipe, since every supplied group limits support.

Results own their arrays even when an input is a scoped prepared batch. Recipes
and input arrays are not retained, and there is no registry or automatic
recomputation. `memory_limit` covers the input/output arrays and two result blocks;
arbitrary temporary allocations in user callbacks are outside that estimate.
Nonfinite formula values propagate; choose explicit handling for invalid physics.

## Explicit bounded preparation

```python
with sm.open_amrvac("snapshot.dat") as source:
    for batch in sm.iter_prepared(
        source, ("b1", "b2", "b3"), scheme="exact-phase", batch_size=32,
        memory_limit=2 * 1024**3,
    ):
        interior = batch.interior()
        # Consume or write the values before advancing the iterator.
```

Batches and every view into their backing expire when the iterator advances
or closes. Descriptor access after expiration raises; an escaped NumPy view
does not become a retained product and must not be used after that scope.
This path reuses a bounded transfer workspace and output buffer without a
cache, eviction policy or implicit loading in scientific consumers.
`iter_traces` instead yields independently owned result batches.

`memory_limit` is optional and bounds estimated controlled live arrays,
including relevant inputs, scratch and outputs. It is not a process-RSS or
filesystem-cache cap. Callers account for additional retained products; memory
pressure never changes requested coverage, precision or numerical scheme.

## Twist, retracing and line-of-sight images

```python
twisted = sm.trace(magnetic, seeds, step=float(.125 * magnetic.mesh.spacing.min()),
                   max_steps=1000, twist=True, curl_field=curl_b,
                   workers=4, memory_limit=limit)
selected = sm.retrace(magnetic, twisted, twisted.seed_ids[:1],
                      step=float(.125 * magnetic.mesh.spacing.min()),
                      max_steps=1000, twist=True, curl_field=curl_b,
                      memory_limit=limit)

with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    density = sm.prepare(source, ("rho",), scheme="coordinate-phase",
                         memory_limit=limit)
view = sm.orthographic_plane(density.mesh.lower, density.mesh.upper,
                             (.3, .2, 1.), (500, 500))
column = sm.integrate_los(density, view, (.3, .2, 1.), workers=4,
                          memory_limit=limit)
```

Twist accumulates only accepted RK segments. A retained curl must match mesh,
source values, selected component definitions and preparation scheme; shape alone
does not establish compatibility. Without `curl_field`, a twist request computes
one temporary curl. Derived fields retain logical provenance without retaining
the primary values. `retrace` accepts selected original seed IDs and returns owned
trajectories from a new integration with the supplied controls.

Scalar LOS integrates the requested component through the physical box, clipped
by scalar or per-pixel `near`/`far` distances. `gauss2` splits at interpolation
knots; `midpoint` is also available. `integrate_los_views` preserves supplied view
and pixel order. Inspect `complete` and `status`: incomplete coverage or sample
limits are explicit outcomes, not valid full-column images.

```python
# Explicit illustrative normalization and isothermal model, not an inference
# about snapshot units or temperature.
thermal = sm.thermal_fields(density, 1.e6, density_unit_g_cm3=1.e-15,
                            temperature_label="isothermal 1 MK example",
                            memory_limit=limit)
image = sm.integrate_thermal_los(thermal, view, (.3, .2, 1.),
                                length_unit_cm=1.e8, workers=4,
                                memory_limit=limit)
```

Thermal LOS retains the historical `AIA171` response and explicit
`CoronalComposition`. Temperature is a positive kelvin scalar or an independently
prepared kelvin field on the same mesh. Density and length conversions are
mandatory. The default samples number density and temperature before evaluating
the nonlinear response; `order="emissivity-first"` applies the response to prepared
nodes first. These are different reconstructions. Composite Gaussian quadrature
for the nonlinear response is approximate. `implementation="reference"` retains
the independent Python ray path and requires one worker.

## Repeated geometry, bounded consumers and large outputs

For QSL maps, localized magnetic footpoints and full-line twist, use
`qsl(magnetic, seeds)` or `iter_qsl`. These consume native AMR Fields and offer
explicit endpoint finite differences or transverse-vector integration, along
with local-sphere maps. See [magnetic connectivity](docs/connectivity.md) for
methods, validity, units and the distinction from accepted-prefix `trace`.

```python
from simesh import bounded

with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    plan = sm.plan_preparation(source.mesh, region=(lower, upper),
                               scheme="exact-phase", support_capacity=128)
    ready = sm.prepare(source, ("b1", "b2", "b3"), scheme="exact-phase",
                        plan=plan, memory_limit=limit)
    with bounded.PreparedPool(source, ("b1", "b2", "b3"),
                               scheme="exact-phase", capacity=64,
                               memory_limit=limit) as pool:
        batches = bounded.iter_traces_bounded(
            pool, seeds, step=float(.125 * source.mesh.spacing.min()),
            max_steps=1000, twist=True, memory_limit=limit)
        for result in batches:
            # Consume each independently owned result batch.
            pass

for index, slab in sm.iter_uniform(ready, (1000, 1000, 1000), memory_limit=limit):
    # Write or consume this independently owned slab before retaining more.
    pass
```

Plans retain mesh geometry and transfer actions, independently of Source, field
values and limiter results. Every preparation binds and computes new values;
the same Mesh can be shared by compatible array sources. `cache_source` is a
separate opt-in raw-interior cache. It borrows its parent Source, must be closed
before releasing that parent, and never changes the numerical scheme.

The bounded namespace also supplies tracing/retracing, plane/uniform sampling
and scalar LOS with missing-block recovery. Pool leases expire on scope exit;
scientific direct APIs continue to accept only Fields. Thermal LOS currently
consumes a completed thermal field group.

`global_curl` prepares all original leaves in bounded batches into independent
storage or a caller-supplied output. `global_curl_file` uses independent file
tasks with `backend="process"` or `"thread"`; each task closes its own Source,
and all workers are joined before returning. Process calls belong inside a
`if __name__ == "__main__":` guard. Caller output may contain partial writes if
a task fails; no completed field is returned on failure.

Uniform iterators hold no full volume. Keeping all yielded slabs or trace
batches alive makes their cumulative storage the caller's responsibility.
The validated output profiles include one million seeds and an actual streamed
1000³ grid; this does not establish large-input acceptance.

## Current scope and provenance

The current profiles accept balanced, nonperiodic Cartesian 3D AMRVAC v5 ordinary fields and
equivalent array sources. Exact-phase preparation requires even block extents
of at least four cells. Saved ghost extents and staggered tails are validated;
CT values are not exposed. Coordinate preparation also checks the canonical
32-bit extent/coordinate limits and the representability of prolongation stencils
before unchecked numerical access. Region preparation is serial; sampling,
derivatives, slices and direct tracing accept explicit thread-pool workers.

Mutable Dataset, ordinary file writing, array-to-file construction, level-1 VTK
export and the independent potential-field helper are available through the
bundled compatibility interfaces. `source_from_dataset` snapshots loaded interior
values; `write_amrvac` exports complete native Fields with explicit metadata.
See [migration and examples](MIGRATION.md) for layouts, memory costs, periodic/CT
file limits and the preserved VTK coordinate convention. New native
2D/periodic/CT/GPU analysis is not claimed.
The separate `qsl` consumer now supplies Q and localized footpoints under its
documented validity rules. Real 10–20 GB input acceptance is not claimed.

[ASSETS.md](ASSETS.md) records implementation provenance. The
[fixed core design](../docs/analysis-core/next-generation-design.md) and
[application guide](../docs/analysis-core/application-development.md) are the active
development references. Earlier specifications, experiments and N1–N4 acceptance
records are preserved in the [archive](../docs/analysis-core/archive/README.md).
