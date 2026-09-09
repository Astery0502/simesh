# Using the independent distribution

Install this directory in a separate environment. Both generations use the
distribution and import name `simesh`; installing one over the other replaces
that environment's package. The parent repository and the N3 saved wheel remain
available for rollback. No runtime imports need either source tree.

## Choose the data model

`PointSet`, `RaySet` and `simesh.applications` add optional geometry/result
associations over native Fields. General array-based calls retain their existing
interfaces. See [application interfaces](docs/applications.md) for selection,
separate trajectory computation and LOS ray IDs.

| Work | Interface in this distribution | Ownership and scope |
| --- | --- | --- |
| Prepare once for sampling, derivatives, tracing or LOS | Top-level `simesh` functions | Immutable Source and independent Fields; nonperiodic Cartesian 3D v5 ordinary values |
| Edit a dataset, register/materialize named fields, refresh ghosts | `simesh.amrvac.open_dataset` | Retained mutable Dataset and its compatibility AMRMesh; independent of the native analysis core |
| Existing native-block or uniform reads | `simesh.amrvac.read_blocks`, `read_uniform`, `load_uniform_data` | Existing field-index selectors, layouts and resampling behavior |
| Construct from uniform arrays or write datasets | `simesh.amrvac.load_from_uniform`, `write_datfile_from_uniform`, `write_datfile` | Retained ordinary-value file workflows; 2D arrays keep singleton z |
| Export level-1 data to legacy VTK | `simesh.amrvac.datfile_to_vtk` | Retained structured-points values and endpoint-based point coordinates |
| Convert a loaded Dataset to an immutable native source | `simesh.source_from_dataset` | Explicit copied snapshot; includes selected materialized fields and excludes ghosts |
| Export complete native Fields to a dat product | `simesh.write_amrvac` | Interior copy in original SFC order; explicit matching metadata; atomic publication |
| Potential-field extrapolation from a bottom array | `simesh.tools.potential_field_green` | Retained independent NumPy helper; optional FFT backend |
| Existing analytic array configurations | `simesh.utils.configurations` | Retained NumPy helpers; no native AMR preparation dependency |

The compatibility implementation is bundled under `simesh.amrvac` and its
`simesh.utils.lib.amr` dependencies. These modules are loaded when requested;
native Source, preparation and scientific consumers do not depend on them.
The old mutable AMRMesh ownership is intentionally confined to this layer.
It is not a replacement for the independent Mesh/Fields design.

Earlier experimental analysis calls map to the new interfaces as follows:

| Previous `simesh.analysis` pattern | Independent equivalent |
| --- | --- |
| `open_source(path, field_names=...)` | `open_amrvac(path)` and select fields at `read_fields`, `prepare` or pool construction |
| `open_prepared(path)` | `open_amrvac` followed by `prepare(..., scheme="coordinate-phase")` for complete coverage |
| `prepare_region` or regional `open_prepared` | `prepare(source, fields, region=..., scheme="exact-phase")` |
| `with_curl(ready)` then twist | `curl_b = curl(ready)` then `trace(ready, ..., twist=True, curl_field=curl_b)` |
| `build_fill_plan(source, ...)` | `plan_preparation(source.mesh, ..., scheme="exact-phase")` |
| Pool objects passed to ordinary consumers | Explicit `simesh.bounded` consumers and pools |
| `budget_bytes` | `memory_limit`, with the accounting documented for each new operation |

Numerical scheme, original versus local field indices, and coverage must be
chosen explicitly during migration. Changing only an import name is insufficient
when the previous interface made those choices implicitly.

## Existing Dataset workflows

For native pointwise formulas, use top-level `derive(fields, name, func,
units=...)`. The callback selects arrays with `ctx.field(name)`; a mapping of
input groups supports formulas combining original and derivative products.
Unlike Dataset registration, this immediately returns an independent field
group and does not mutate inputs or store a recipe. Native `derivative` also
accepts field names and `"x"`/`"y"`/`"z"` axes. See the README for common-halo
and pointwise-reconstruction semantics.

The eight public AMRVAC entrypoints retain their signatures:
`open_dataset`, `read_blocks`, `read_uniform`, `load_from_uniform`,
`write_datfile`, `write_datfile_from_uniform`, `load_uniform_data` and
`datfile_to_vtk`. Dataset registration, materialization, derivatives, field
selection and removal remain available.

```python
from simesh.amrvac import open_dataset

dataset = open_dataset("snapshot.dat", ghost_width=2)
dataset.load_data(field_indices=[0, 1])
dataset.register_derived(
    "difference", lambda ctx: ctx.field("e") - ctx.field("rho"),
    dependencies=["e", "rho"],
)
dataset.materialize_fields(["difference"])
values = dataset.blocks(field_names=["difference"])
```

Use actual source field indices and names in the example. A recipe is evaluated
when materialized; changing its dependencies does not make every stored derived
column automatically fresh. Original file indices and current loaded-field names
remain different selectors. Ghost-dependent recipes retain their existing
restrictions on original loaded dependencies and valid derivative layers.

Compatibility layouts remain `(leaf, component, x, y, z)` for blocks,
`(x, y, z, component)` for file-level uniform reads and `(component, x, y, z)`
for Dataset `uniform_grid`/`uniform_full`. Native Fields use
`(slot, x, y, z, component)` with explicit leaf-to-slot mapping.

The default `read_uniform(..., interpolation="zero")`, exact level-1
`uniform_full()`, and ghost-dependent `interpolation="linear"` retain their
existing meanings. They are not aliases for native `sample_plane` or
`iter_uniform`, whose pixel/cell-center coordinates are described in the README.
Likewise, the compatibility VTK writer preserves its tested endpoint-coordinate
convention; it does not reinterpret existing output as native cell-center geometry.

## Cross the boundary explicitly

```python
import simesh as sm
from simesh.amrvac import open_dataset

dataset = open_dataset("snapshot.dat")
dataset.load_data(field_indices=[4, 5, 6])  # Example original B columns.
source = sm.source_from_dataset(dataset, ("b1", "b2", "b3"),
                                memory_limit=2 * 1024**3)
# Later Dataset edits do not change source.
magnetic = sm.prepare(source, scheme="exact-phase", memory_limit=2 * 1024**3)
source.close()
curl_b = sm.curl(magnetic, memory_limit=2 * 1024**3)
sm.write_amrvac("curl.dat", curl_b, metadata=dataset.metadata,
                memory_limit=2 * 1024**3)
```

`source_from_dataset` requires an already loaded, nonperiodic Cartesian 3D
Dataset. It copies selected interior columns once and creates independent
integer geometry. No Dataset, old AMRMesh, callback recipe or old ghost backing
is retained. Field names address loaded columns, including materialized derived
columns; optional `units` labels do not perform conversions. A snapshot must be
prepared explicitly before ghost-dependent native analysis.

`write_amrvac` requires all original leaves, supports arbitrary physical slot
order, and excludes halo values. Its required `metadata` must be a v5 header
matching the full mesh geometry. It preserves supplied time/model metadata while
rebuilding field names, tree counts and offsets. Ordinary output explicitly has
`staggered=False`. A subset of the domain cannot be serialized as the original
forest. Field units, preparation scheme and derivative provenance are not encoded
by the dat format; save those separately if needed. A curl or other analysis
product is not automatically a valid simulation restart.

Native export holds one additional field-major interior array plus per-block
serialization scratch. `memory_limit` admits these alongside Fields and Mesh;
it does not account for unrelated Dataset or other products still retained by
the caller. The mutable compatibility APIs retain their existing allocation
behavior and do not implement the native memory budget.

The export is written to a temporary sibling file before publication. The
default refuses an existing destination, including one created concurrently.
With `overwrite=True`, replacement happens only after serialization succeeds.
This prevents partially replacing an existing destination; it is not a promise
of crash-durable storage. Existing compatibility writers retain their own
overwrite and partial-write behavior.

## Supported continuity and explicit limits

- Cartesian 2D read/write, uniform arrays and VTK retain singleton-z behavior.
  Native scientific analysis remains 3D; a 2D Dataset is not accepted by the
  snapshot adapter.
- Periodicity metadata survives ordinary-value file reads and writes. The old
  Dataset did not implement periodic ghost neighbors. This distribution rejects
  `ghost_width>0` for periodic files instead of applying physical boundary fills.
  Native preparation continues to reject periodic sources.
- Nonperiodic compatibility ghost modes `cont`, `symm`, `asymm` and `noinflow`
  retain their existing selectors and normal-velocity requirements.
- The inherited coarse-block support-table overrun is corrected here, as in
  native coordinate-phase v2. Constant fields spanning both physical sides no
  longer receive invalid zero ghost values. The parent generation is unchanged.
- Ordinary compatibility writers reject `staggered=True`: they have no CT-face
  payload to serialize. Use the explicit native Fields export for an ordinary
  data product, or the original simulation tooling for a full CT restart.
- Compatibility AMR extensions are built serially, even with
  `SIMESH_OPENMP=1`. Their inherited uniform-grid OpenMP boundary writes are not
  sufficiently validated. Native preparation/tracing/LOS retain their explicit
  OpenMP option. `simesh.utils.openmp_build_info()` reports compatibility status;
  native kernel build status is separate.
- The old `simesh.analysis` names are reorganized into the top-level native
  API and `simesh.bounded`, rather than retained as a second dispatcher. The
  historical `simesh.legacy` and `simesh_rewrite` packages are not runtime
  dependencies of this distribution.

For array-only potential fields, the default `backend="auto"` uses SciPy FFT
convolution when available and otherwise uses the retained direct NumPy path.
Install `pip install '.[fft]'` from this directory to enable the optional FFT
dependency; the base package requires only NumPy at runtime.

The N1–N4 baseline did not deliver Q or localized boundary footpoints. The
subsequent native `qsl`/`iter_qsl` consumer adds these and complete-line twist;
see [magnetic connectivity](docs/connectivity.md) for methods and validity.
Existing `trace` still returns accepted prefixes. GPU/CT/periodic native analysis,
bounded nonlinear thermal LOS, and acceptance of real 10–20 GB input files
remain outside the delivered scope.
