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

For repeated twist requests, construct `with_curl(primary)` or `CurlPool(primary)`
once and reuse that composition. Passing plain B to `trace(..., twist=True)`
creates a temporary companion for that call and can recompute curl on the next
call. The retained resident composition's `.curl` also supplies repeated slices.
Changing seeds or planes does not require rebuilding compatible B/curl groups;
each new trajectory and its stage quadrature still has to be computed.

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

## Runtime Reuse And File Lifetimes

[Runtime execution](runtime-execution.md) distinguishes numerical reuse,
scheduling and backend evidence. A `PreparedPool` retains complete two-halo
blocks, including their ghost values. Size it for the measured working set when
possible. F and single-view LOS preserve the original request-priority heuristic.
Multi-view tile interleaving uses non-touching coverage leases so recently
completed blocks remain useful to the next nearby view; explicit block requests
still update recency. This is not sample-level exact LRU. Actual-hit tracking and
extra padded miss staging were compared and not retained in the selected path.

`open_source(..., value_cache_capacity=2048)` additionally retains up to 2,048
ordinary-interior blocks behind the provider, deduplicating repeated support
reads. It defaults to zero and can be less effective than using those bytes for
complete prepared blocks. Its one ordered field-group binding is cleared when
the selected fields/order change. `read_value_bytes` reports underlying loaded
values; `requested_value_bytes` and the provider's `selected_load_count` include
support requests served from cache. File-record/header bytes are separate.

File sources check their opened file identity and context lifetime even for
prepared-cache hits. Changed values require a new source and new pools; shared
geometry does not validate old values. Detached products remain snapshots after
source closure. Array-backed sources require the owner's immutable-lifetime
promise, or a `validate_values` callback that raises when their epoch changes.
Closing the source context closes its file; its cache storage remains owned by
source/pool references until those references are released.

## Nearby LOS Views

Use one finite pool for views of the same immutable scalar field. This returns
all requested images in input order; tile scheduling changes no per-pixel sum.
The common `near`/`far` bounds can be scalars or image-shaped arrays. All planes
must have the same image shape, and every output image is admitted together.

```python
from simesh.analysis import integrate_los_views, orthographic_plane

directions = [[0.3, 0.2, 1.0], [0.31, 0.2, 1.0], [0.3, 0.21, 1.0]]
planes = [orthographic_plane(source.mesh.lower, source.mesh.upper, d, (32, 32))
          for d in directions]
images = integrate_los_views(scalar_pool, planes, directions,
                             tile_shape=(4, 4), view_order="tile")
```

The measured benefit is for nearby views whose tile working sets overlap. Use
`view_order="view"` for the sequential comparison. Single-image `integrate_los`
keeps its existing behavior. These are supplied-scalar integrals; this interface
does not select AIA response or thermodynamic inputs.

## Independent Preparation For Dense Curl

`global_curl_file` uses bounded file/preparation tasks and returns the complete
native one-halo curl product, usable by the existing slice/sampling APIs. It
admits full output, private providers, derivative buffers, in-flight results and
IPC copies together. The optional process path can parallelize preparation that
remains serial under Python threads. `global_curl(source, ...)` stays available.

Run the process path from an importable script with a main guard:

```python
from simesh.analysis import global_curl_file


def main():
    current = global_curl_file("snapshot.dat", workers=2, task_size=512,
                               batch_size=256, support_capacity=256,
                               budget_bytes=1024**3)
    # current covers every native leaf, independently of later slice geometry.


if __name__ == "__main__":
    main()
```

`backend="thread"` retains the matched independent-thread preparation comparator;
it was not faster on the measured preparation-bound WENO case. No provider/pool
is shared mutably between workers. A file change aborts publication. Failure may
leave completed ranges in an explicitly supplied output sink, as in `global_curl`.
Larger retained outputs require their own admitted budget; this does not establish
10--20 GB input acceptance.

## Optional Consumer Scheduling

Tracing and LOS accept `backend="threadpool"` (default) or `backend="openmp"`,
and `schedule="static"` (default) or `schedule="dynamic"`. Thread-pool dynamic
mode queues smaller row groups to the same bounded worker team; it can be worse
for small tiles. Static OpenMP did not improve the measured tracing case over
threads. Dynamic OpenMP is an explicit option for repeated, already-prepared LOS,
with the complete-cost limits recorded in the runtime evidence.

Build and inspect the **analysis** extension, not just the canonical AMR module:

```sh
.venv/bin/python build.py --inplace --group analysis --openmp
OMP_WAIT_POLICY=PASSIVE OMP_DYNAMIC=FALSE .venv/bin/python your_analysis.py
```

```python
from simesh.utils.lib.analysis.native import openmp_build_info
from simesh.analysis import integrate_los

assert openmp_build_info()["enabled"]
image = integrate_los(resident_scalar, plane, direction, workers=4,
                      backend="openmp", schedule="dynamic", tile_shape=(64, 64))
```

The environment settings must be chosen before the OpenMP runtime starts.
Default builds need no OpenMP runtime. Explicitly requesting the OpenMP backend
from a default build raises a clear error; it does not silently change engines.
To restore the default analysis build:

```sh
SIMESH_OPENMP=0 .venv/bin/python build.py --inplace --group analysis
```
