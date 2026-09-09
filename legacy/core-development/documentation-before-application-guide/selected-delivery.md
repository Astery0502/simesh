# Select values, spatial support and delivery independently

`Fields` describes values and their spatial support separately. `storage_halo`
locates interiors in the allocated array; `valid_halo` counts completed layers.
Allocated, zero-filled or stale padding never establishes valid support.
`FieldDefinition.interpretation`, `source`, `value_identity` and `derivation`
describe numerical meaning and provenance independently of those layer counts.
A derived field can have zero, one or two valid layers; an original interior
field has no valid halo simply because its values came directly from a file.

## Choose the required preparation

```python
import simesh as sm

with sm.open_amrvac("snapshot.dat", fields=("rho", "e", "b1", "b2", "b3")) as source:
    interiors = sm.read_fields(source, ("rho", "e"))
    magnetic = sm.prepare(source, ("b1", "b2", "b3"), scheme="coordinate-phase")

mass = sm.volume_integral(interiors, "rho")
values, owners, valid = sm.sample(magnetic, [[0.4, 0.4, 0.4]], components="b3")
```

Choose `read_fields` for interiors or `prepare` for two completed halo layers.
Use the same explicit scheme as the intended numerical workflow: regional and
bounded preparation uses `exact-phase`; `coordinate-phase` requires complete
coverage. No consumer reads a Source, expands coverage or switches schemes
implicitly. Invalid coverage inside the supplied field group retains the
existing sample/termination status conventions.

| Consumer | Accepts interior fields? | Minimum valid halo and further requirements |
| --- | --- | --- |
| `volume_integral`, `weighted_mean`, `extrema`, `histogram`, `surface_flux` | Yes | 0; native cell values and existing region/face measures |
| `derive`, `derive_many`, magnitude, dot, magnetic pressure/energy | Yes | 0; pointwise evaluation on common valid support |
| `select_fields`, `merge_fields` | Yes | 0; independent copies of selected/common valid support |
| `mhd_fields` | Yes | 0; explicit physical model/units; common input support |
| `thermal_fields`, `emissivity_fields` | Yes | 0; explicit density/temperature/model; common input support |
| `sample`, `sample_plane`, `iter_uniform`, application sample/map/uniform grid | No | 1, in each selected continuous component |
| `sample_line_profiles`, `iter_line_profiles` | No | 1, in selected continuous components; stored geometry is unchanged |
| `derivative`, curl, gradient, divergence, current density | No | 1; output has input `valid_halo - 1` |
| `trace`, `iter_traces`, `retrace`, application trace/`iter_lines` | No | 1 for the vector; three ordered components with common units |
| Trace twist with automatic curl | No | 2 for the vector when twist integration is requested; differentiation leaves 1 for curl interpolation |
| Trace twist with retained curl | No | 1 for vector and 1 for curl; exact derivation certificate and coverage must match |
| QSL finite-difference mapping without twist | No | 1 for vector; includes neighboring-seed endpoint integrations |
| QSL variational mapping | No | 2 for vector; unit-vector derivatives retain 1 for interpolation |
| QSL/line diagnostics with twist | No | Add the same automatic/retained curl requirements as trace; twist-only skips Q work |
| Scalar LOS, including application ray sets | No | 1 for selected continuous scalar; quadrature and missing-coverage rules are unchanged |
| Thermal LOS, both reconstruction orders | No | 1 for both physical thermal components and matching model identity |
| `write_amrvac` | Yes | 0; requires full original-mesh coverage and explicit matching metadata |

Trace requests with zero steps or zero permitted length do not compute an
unused automatic curl. An explicitly supplied companion is still validated.
Bounded consumers use explicitly prepared pool leases; they do not give direct
Fields consumers implicit recovery behavior.

A derivative of a one-layer field is a valid interior field for reductions and
pointwise formulas, but cannot be interpolated. Composition and MHD/thermal
recovery use the minimum valid layer count across inputs. Extra allocated
padding is excluded. Restore support by explicitly preparing the original
inputs and recomputing the derivation, allowing one layer per derivative;
there is no general operation that turns arbitrary padding into valid data.
Two successive derivatives of a two-layer input leave no valid halo.

Categorical values, such as MHD status codes, remain categorical even with two
valid layers. Continuous interpolation, tracing, derivatives and LOS reject
selected categorical components. Select the physical components for those
operations; inspect status through native values and masks.

## Read into final storage

Built-in file and array Sources write selected interiors directly into the
final `(slot, x, y, z, component)` backing. With halos, that backing is allocated
first and published only after filling succeeds. The full-domain file reader
keeps bounded record buffers (normally 16 MiB); sequential records may contain
unselected fields and validated CT tails, but only selected ordinary values are
decoded into output. Regional reads keep the selected-field reader's record,
endian and source-identity checks.

Exact-phase preparation still needs bounded field-major support and arithmetic
workspace. Primary values first enter final storage and are copied into that
workspace for the unchanged fill calculation; only completed halos are copied
back. A reusable geometry plan no longer needs a separate raw-interior buffer.
Coordinate-phase preparation releases read buffers before allocating coarse
exchange workspace, and still skips coarse storage when no coarse/fine
adjacency exists. Publication adds no full-array copy.

Advanced Sources that only provide a field-major BlockReader use a one-leaf
conversion buffer for final-storage reads. `Source.read_into` retains its
field-major contract. `Source.read_native_into(..., storage_halo=0)` is the
explicit caller-storage transfer and does not establish halo validity.
Failure may leave its caller's output partially written.

## Restrict a Source before caching

```python
with sm.open_amrvac("snapshot.dat") as original:
    with sm.select_source(original, ("b3", "b1", "b2")) as selected:
        with sm.cache_source(selected, fields=("b1", "b2", "b3"), capacity=128) as cached:
            ready = sm.prepare(cached, scheme="exact-phase")
```

`select_source` creates a borrowed adapter with an ordered local-to-original
component map. The map copies no numerical values, shares Mesh and source
identity, and preserves original component IDs in logical value certificates.
Nested adapters and caches therefore remain compatible with equivalent direct
preparation. Closing an adapter leaves its parent open; the parent must outlive
the adapter and remain immutable. Parent closure or file mutation invalidates
reads, including cache hits. `open_amrvac(fields=...)` owns its underlying file
and closes it when the returned Source closes.

Cache arrays and preparation budgets use the restricted component count.
`cache_source(source, fields=...)` is shorthand for restricting before cache
allocation. Omitting `fields` retains the Source's entire component directory.
Changing a cache's requested ordered subset invalidates its old keys.
Array Sources still retain their original borrowed/owned backing: an adapter
cannot release unselected portions of an array owned by another object.

## Compute selected columns and write final outputs

The sampler takes the original Fields plus component indices. Location and
interpolation coordinates are shared; the unchanged arithmetic evaluates only
selected columns. This avoids materializing a global field subset for sampling,
planes, profiles or uniform volumes. `select_fields` remains the explicit
compact independent-copy API for independent lifetimes and consumers requiring
a standalone vector group.

`sample(..., components=..., output=(values, owners, valid))` accepts writable
float64/int64/bool arrays with matching shapes and disjoint positive strides.
`sample_plane` accepts the same tuple in contiguous plane-shaped arrays.
Application `sample` and `field_map` expose these options as well.
`derivative(..., output=...)` and `curl(..., output=...)` accept the matching
contiguous native float64 output layout.

```python
import numpy as np
from simesh import applications as app

shape = (256, 256, 256)
values = np.memmap("b3.bin", mode="w+", dtype="float64", shape=(*shape, 1))
valid = np.empty(shape, dtype=bool)
volume = app.uniform_grid(magnetic, shape, components="b3", output=(values, valid))
```

Uniform-grid sampling writes directly to the final volume, including strided
z-plane destinations; it does not build an intermediate SliceResult per plane.
Plane samples and line profiles also write batches into their final arrays.
`iter_uniform` still delivers independent slabs without allocating a volume.

Caller output is borrowed and must not alias field or geometry inputs, including
uniform-grid bounds. Retain it unchanged
while using a returned field/result. A successful Fields publication is
read-only through its view; it does not revoke the caller's writable alias.
A failed call can leave partial output. No complete product is returned on
failure; never save or label such an output as complete. Memory admission counts
logical output storage even for a memmap and is not a process RSS limit.

## Deliver variable-length lines by seed shard

```python
lines = app.iter_lines(magnetic, seeds, seed_batch=128, step=0.01, max_steps=1000)
profiles = sm.iter_line_profiles(physical_fields, lines, components=("temperature", "density"))
product = sm.save_result_shards("profiles", profiles, seed_ids=seeds.ids,
                                source={"snapshot": "snapshot.dat"})
one_shard = sm.open_result_shards("profiles").load(0).result
```

`seeds` is a `PointSet` with stable global IDs. Each `iter_lines` result is a
compact `LineSet` containing at most `seed_batch` seeds, both requested branches,
original statuses and packed offsets. `iter_line_profiles` accepts any iterable
of LineSet shards, including individually loaded files. Neither API concatenates
all trajectories/profiles. Retaining all yielded results remains the caller's
memory responsibility. `sample_line_profiles` and application `trace` retain
their complete-result interfaces.

`save_result`/`load_result` also support a standalone `LineProfiles`, preserving
geometry, selected component indices/definitions, arclength units, boundary
adjustments and finite/coverage masks. Loaded value and tracing identities are
unset, exactly as with other result files; caller-supplied source metadata is
unverified and never restores an in-memory identity certificate.

`save_result_shards` requires a new destination directory and the complete
ordered int64 `seed_ids`. It snapshots these IDs before consuming the producer,
validates each next range, saves an independent NPZ,
and atomically updates `manifest.json`. Each entry records its half-open seed
row range, result kind, filename and SHA-256 checksum. The manifest stores the
full global IDs and marks delivery complete only after all requested seeds have
been saved. Completeness concerns delivery, not scientific success: termination
and physical validity remain separate per-seed results.

A failed producer/save leaves the manifest explicitly incomplete. Completed
shards remain individually loadable. `open_result_shards` reads only the manifest;
`.load(index)` verifies and loads just that shard. It rejects checksum, ID and
range mismatches. This is incremental delivery, not resumable RK/LOS integration
or a crash-durability guarantee. NPZ encoding still snapshots one shard at a
time; include that shard's serialization memory in an application budget.
