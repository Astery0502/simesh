# Identified points, rays and application results

The optional `simesh.applications` namespace organizes geometry and result
associations over the existing native consumers. Keep using top-level `sample`,
`qsl`, `trace` and LOS functions when raw arrays are more convenient. Preparation
still explicitly produces `Fields`; application functions do not own a Source
or choose a ghost scheme.

## Diagnose, select, then trace

For standard mathematical/magnetic fields and ready-to-use surface or uniform
products, see [standard diagnostics and applications](standard-applications.md).

```python
import numpy as np
import simesh as sm
from simesh import applications as app

with sm.open_amrvac("snapshot.dat") as source:
    magnetic = sm.prepare(source, ("b1", "b2", "b3"),
                          scheme="coordinate-phase")

points = sm.PointSet.boundary(magnetic.mesh, "zmin", (128, 128))
diagnostics = app.connectivity(magnetic, points, workers=4)
log_q_image = diagnostics.image("log10_q")

# Illustrative thresholds: choose these for the actual physical problem.
selected = diagnostics.threshold(q_min=1e4, abs_twist_min=1., mode="any")
lines = app.trace(magnetic, selected, direction="inward",
                  step=float(magnetic.mesh.spacing.min())*.125,
                  max_steps=4000, workers=4)

for seed_id in lines.seeds.ids:
    polyline = lines.line(seed_id)
    # Consume the polyline and inspect lines.termination for completeness.
```

**QSL/twist computation does not save paths.** Threshold selection operates on
the completed diagnostics and returns a `PointSet`. Tracing is a separate,
explicit calculation, so changing thresholds can reuse the same diagnostic map.
The general QSL implementation and result contract are unchanged.

`threshold` combines inclusive Q and absolute-twist criteria with `mode="any"`
or `"all"`. Q uses the diagnostic `valid` mask; twist requires complete central
lines and finite twist. Disabled twist raises if requested. For arbitrary
criteria, use `diagnostics.select(boolean_mask)` and construct the validity
conditions explicitly. No thinning, ranking or physical threshold is inferred.

Pass `quantities=("twist",)`, `("q",)` or `("q","twist")` to
`app.connectivity`/`iter_connectivity` to compute only requested diagnostics.
Without `quantities`, existing QSL controls including `twist=False` retain their
meaning. Do not specify both quantity selection and the legacy twist flag.
Use `q_valid` / `twist_valid`, or their image forms, for quantity-specific validity.

## Point sets and result association

`PointSet(positions, ids=None, shape=None, normals=None, plane=None)` accepts
arbitrary finite physical coordinates of shape `(n,3)`, including AMR cell centers
and locations unrelated to the stored grid. IDs are unique int64 labels, not
array indices; they default to `arange(n)`. Construction owns read-only geometry.

- `PointSet.from_plane(plane)` generates pixel centers in the same order as
  `sample_plane`, with the plane's two-dimensional shape.
- `PointSet.boundary(mesh, face, shape)` uses a physical mesh face. Face names
  are `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`; normals point outward.
- `PointSet.iter_plane(plane, batch_size=...)` generates coordinate batches
  without allocating the full coordinate grid. IDs remain flat parent-pixel indices.
- `points.select(mask)` preserves IDs and parent-plane metadata. Its current
  shape becomes `(selected_count,)`; it does not pretend a sparse selection is
  still a rectangular image.
- `points.reshape(values)` restores the current point layout, preserving any
  trailing component dimensions. Masks accept this layout or a flat `(n,)` array.

```python
plane = sm.Plane([0., 0., .5], [1., 0., 0.], [0., 1., 0.], (128, 96))
points = sm.PointSet.from_plane(plane)
sampled = app.sample(magnetic, points, workers=4)
vector_image = sampled.image
selected = sampled.select(points.reshape(sampled.usable))
```

`SampledPoints` retains points, values, owner leaves, validity and field
definitions. `ConnectivityMap` retains points and its raw `QSLResult` as `data`.
Both record the input field's logical `source_identity` token, without retaining
its value arrays. This identifies provenance; geometric selections are reusable
on another field by explicit caller choice.

Sampling `valid` retains the native coordinate/coverage meaning. `usable` also
requires all sampled components to be finite, which is useful when selecting
points from fields containing NaN or infinity. `UniformResult` provides the same
distinction. These masks do not infer whether a finite value is physically valid.

`app.iter_connectivity` preserves original IDs across diagnostic batches and
reuses the general iterator's diagnostics. Each batch owns its result and point
subset. Its image layout is one-dimensional; the parent plane and global IDs
remain available to place values in a larger output.

## Magnetic trajectories

`app.trace` returns a compact `LineSet`. Directions are:

- `"both"`: integrate against and along B.
- `"along"` or `"against"`: integrate one requested branch.
- `"inward"`: choose the sign separately for each point on a physical box face.
  A seed must lie on exactly one face. Tangent seeds have no inward direction.

The numerical controls remain those of the general accepted-prefix tracer:
physical `step`, `max_steps`, `max_length`, `null_threshold`, workers and backend.
This does not adopt QSL's localized endpoint or local-sphere semantics. Upper
boundary seeds start from the nearest representable interior coordinate to
respect the native half-open ownership rule; their displayed initial point
keeps the exact supplied boundary coordinate.

`LineSet.positions` stores only accepted points. `offsets` has `2*n+1` entries:
branches `2*i` and `2*i+1` go from seed i against and along B. The returned line
set keeps the input point IDs and the magnetic field's source token.

- `lines.branch(seed_id, -1 or 1)` returns a read-only branch view.
- `lines.line(seed_id)` returns an owned polyline oriented along B, joining the
  reversed negative branch and positive branch without duplicating the seed.
- `termination` has shape `(n,2)`. Ordinary codes are `Termination`;
  `LineSet.NOT_REQUESTED` means no branch was requested and
  `LineSet.TANGENT_SEED` means an inward direction was undefined.

Packing occurs after bounded calls to the existing tracer, which still uses
temporary `max_steps+1` storage per seed. `seed_batch` limits that temporary
storage. `memory_limit` admits geometry, native calls and packed output assembly;
it is not a promise of constant memory when all output paths are retained.

## Identified LOS rays

`RaySet` holds a `PointSet` of origins, normalized directions, and nonnegative
near/far arclength limits. Directions can be shared `(3,)` or per-ray `(n,3)`.
Near/far can be scalars, flat arrays, or arrays matching the origin layout.
Origins may lie outside the field domain; integration clips to the intersection.

```python
direction = (.3, .2, 1.)
plane = sm.orthographic_plane(density.mesh.lower, density.mesh.upper,
                              direction, (256, 256))
rays = sm.RaySet.from_plane(plane, direction)
column = app.los(density, rays, component=0, workers=4)
image = column.image
selected_rays = column.select(rays.origins.reshape(column.valid))

thermal_image = app.thermal_los(thermal, rays, length_unit_cm=1e8, workers=4)
```

Here `density` and `thermal` are explicitly prepared scalar and thermodynamic
field groups; see the main README for physical normalization. Scalar LOS reuses
the native Gaussian/midpoint integrator. Thermal LOS reuses the native AIA171
response and either thermodynamics-first or emissivity-first reconstruction.
The ray-set adapters support per-ray directions with thread-pool workers and
bounded `ray_batch` scratch. Existing plane-based APIs keep their other backend
and reference options.

`RayResult` preserves ray IDs, image layout, entry/exit distances, sample counts
and status. `valid` distinguishes complete/empty rays from failures; `complete`
states whether every ray is valid. Thermal results record physical metadata.
Selection preserves the exact normalized ray directions and clipping intervals.
LOS rays are straight lines; they are not magnetic field-line trajectories.

All application result arrays belong to their results; input geometry is shared
read-only. Memory limits account for known live arrays, not process RSS or other
results retained by the caller. Use geometry/diagnostic batches when full point
grids or full output products would be too large.
