# Experimental Native Analysis Usage

Use [current](current.md) for the exact delivered stage and unresolved gates.
Canonical `simesh.amrvac` APIs remain available. The new numerical kernels build
with the standard setuptools discovery under `src/simesh/utils/lib/analysis`:

```sh
.venv/bin/python build.py --inplace --group analysis
```

The core has no implicit file/cache provider. Current development assembly uses
`scripts/analysis_core/rewrite_provider.py` with `PYTHONPATH=src:rewrite/src:scripts`.
It takes existing validated rewrite forest and block-reader descriptors; it does
not make `simesh_rewrite` an undeclared installed dependency. The bounded reader
supports the original non-staggered v5 scope; the WENO driver explicitly uses a
resident ordinary-field bootstrap from its staggered records. Source arrays/file
descriptors are borrowed and remain immutable/alive during preparation.

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

result = trace(pool, seeds, step=0.001, max_steps=1000, workers=4)
for chunk in iter_traces(pool, seeds, step=0.001, max_steps=1000,
                         workers=4, seed_batch=128):
    write_summary(chunk)  # Caller-supplied sink; no mandatory concatenation.
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
