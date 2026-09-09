# Read-Only Spherical AMRVAC Support Design

## Purpose

Add honest first-stage spherical geometry support to `simesh` without changing
the existing Cartesian behavior. This milestone lets users load spherical
AMRVAC snapshots, inspect native spherical coordinates, compute geometry-aware
metrics, and see clear capability/progress information. Operations that need
full AMRVAC-style spherical AMR behavior remain blocked and are recorded as
follow-up work.

## Scope

This design targets read-only spherical support in the canonical Python AMRVAC
path under `src/simesh/amrvac/`. It does not implement pole-aware ghost-cell
exchange, spherical AMR prolongation/restriction, or spherical vector
operators.

Allowed for spherical datasets:

- read metadata and block data
- expose native `(r, theta, phi)` coordinate roles
- compute/query spherical cell centers and edges
- compute/query spherical cell volumes
- compute/query spherical face areas
- compute/query physical path lengths
- expose a progress/status record explaining active support and deferred work

Blocked for spherical datasets:

- opening with `ghost_width > 0`
- `exchange_ghost_cells()`
- linear interpolation that depends on ghost-cell-padded storage
- write-back after mutation
- AMR coarsen/prolong behavior
- pole-crossing neighbor exchange
- Cartesian output claims unless an explicit spherical-to-Cartesian conversion
  path is implemented

## Architecture

The first milestone adds a small Python geometry layer before the Cython AMR
operational layer. Existing `.dat` metadata reading remains in
`src/simesh/amrvac/datio.py`. After metadata is read, the dataset constructs a
geometry descriptor from the header fields:

- `geometry`
- `ndim`
- `xmin`
- `xmax`
- `domain_nx`
- `block_nx`
- `periodic`

For Cartesian datasets, behavior stays unchanged. For spherical datasets, the
dataset enters `spherical_read_only` mode. It can load metadata and interior
block data, but it must not route data into Cython ghost-cell exchange or AMR
mutation paths.

The Cython `AMRForest` and `AMRMesh` remain Cartesian-operational during this
milestone. Spherical behavior is handled by Python guards and read-only metric
helpers. This avoids pretending that the existing Cartesian mesh has
spherical-aware topology, ghost exchange, or interpolation.

## Components

### GeometryDescriptor

Add a small Python object in the AMRVAC package, for example
`src/simesh/amrvac/geometry.py`.

Responsibilities:

- normalize geometry labels from the AMRVAC header
- classify geometry as Cartesian or spherical
- expose coordinate roles
- validate supported dimensionality for read-only spherical support
- report whether the dataset is operational or read-only

Expected coordinate roles:

- spherical 3D: axis 0 is `r`, axis 1 is `theta`, axis 2 is `phi`
- spherical 2.5D or 2D variants are detected but not supported in this
  milestone, and requests that require interpreting them raise `ValueError`

### SphericalMetrics

Add pure-Python metric helpers, likely in the same module as
`GeometryDescriptor` unless it grows too large.

Responsibilities:

- compute cell edge coordinates from block extents and local block sizes
- compute cell center coordinates
- compute cell volumes from spherical shell and angular factors
- compute radial, theta, and phi face areas
- compute physical path lengths:
  - `ds_r = dr`
  - `ds_theta = r dtheta`
  - `ds_phi = r abs(sin(theta)) dphi`

Face areas should use positive metric factors near poles. In particular,
theta-dependent factors should use `abs(sin(theta))` where sign would otherwise
make an area negative.

The first implementation should favor correctness and clear tests over Cython
speed. These helpers can return NumPy arrays for a requested block or for all
loaded blocks.

### Dataset Status

Expose a lightweight status/progress record on the dataset, such as
`ds.geometry_status`.

For spherical read-only datasets, it should include:

- `mode`: `spherical_read_only`
- `geometry`: normalized geometry label
- `metrics_available`: true when metric helpers are attached
- `blocked_operations`: operation names that are intentionally unavailable
- `todo`: follow-up work required for operational spherical AMR support
- `notes`: short human-readable notes about native spherical coordinate order

This should not print by default. Public helper calls may return or expose this
status so users can log it in their own workflows.

### Guard Layer

Add clear guards in `AMRVACDataSet` and public entrypoints before operations
that would be wrong for spherical data.

Guard examples:

- `open_dataset(path, ghost_width=1)` on spherical data raises
  `NotImplementedError`
- `read_uniform(..., interpolation="linear")` on spherical data raises
  `NotImplementedError`
- `AMRVACDataSet.exchange_ghost_cells()` on spherical data raises
  `NotImplementedError`
- `AMRVACDataSet.write_datfile()` on spherical data raises
  `NotImplementedError`
- Cartesian VTK export from spherical data raises `NotImplementedError`

Error messages should state that spherical support is currently read-only and
name the deferred capability, such as pole-aware ghost exchange or
spherical-to-Cartesian output conversion.

## Data Flow

1. User opens a `.dat` file through `open_dataset()` or another public AMRVAC
   helper.
2. `datio.get_metadata()` reads the existing header, forest flags, and tree
   information.
3. `AMRVACDataSet.load_metadata()` builds a `GeometryDescriptor`.
4. If geometry is Cartesian, the existing path continues unchanged.
5. If geometry is spherical:
   - the dataset records `spherical_read_only` status
   - unsupported constructor options are rejected early
   - interior block data can be read with existing block readers
   - metric helpers are available for native spherical analysis
   - ghost-cell, mutation, and Cartesian-output paths are blocked

## Error Handling

Errors should be early, explicit, and tied to the missing spherical capability.

Use `NotImplementedError` when the requested operation is conceptually valid
but not implemented for spherical geometry yet. Use `ValueError` when the file
or request is inconsistent with read-only support.

Examples:

- unsupported spherical dimensionality: `ValueError`
- spherical plus `ghost_width > 0`: `NotImplementedError`
- spherical linear interpolation: `NotImplementedError`
- spherical write-back: `NotImplementedError`
- pole assumptions that require periodic phi but the metadata is not periodic:
  status warning for read-only metrics, `ValueError` for any operation that
  would require pole topology

## Testing

Add focused tests under `tests/amrvac/`.

Required coverage:

- spherical geometry label is detected and normalized from metadata
- spherical datasets expose read-only status
- simple spherical cell volumes match analytic expectations on a small grid
- face areas are positive near theta poles
- physical path lengths match `dr`, `r dtheta`, and
  `r abs(sin(theta)) dphi`
- `ghost_width > 0` on spherical data raises a clear `NotImplementedError`
- spherical linear interpolation raises a clear `NotImplementedError`
- spherical write-back raises a clear `NotImplementedError`
- existing Cartesian tests continue to pass unchanged

Synthetic test fixtures can build small headers and block coordinate arrays
directly rather than requiring large binary reference files.

## Documentation

Update user-facing docs to state that spherical support is read-only in this
milestone.

Suggested documentation touch points:

- `README.md`: update current limits from Cartesian-only to Cartesian
  operational plus spherical read-only
- `docs/user-guide.md`: add a short spherical read-only section
- `docs/api-reference.md`: document `geometry_status` or equivalent status
  field
- `docs/amr-forest-mesh.md`: record that pole-aware AMR topology remains
  deferred
- `docs/python-api-map.md`: list any new public helper or dataset property

## Follow-Up Work

Operational spherical AMR support is deferred and should be tracked as phase 2.

Phase-2 To-Do Items:

- pass geometry and periodicity into `AMRForest` and `AMRMesh`
- validate spherical pole topology when phi is periodic
- require even phi block count for pole shifts
- add `neighbor_pole` or equivalent connectivity records
- implement theta pole crossing with half-domain phi shift
- implement pole-aware fine-neighbor child selection
- implement pole ghost exchange buffers
- implement pole copy with index reversal and symmetric/asymmetric signs
- implement metric-weighted coarsening
- implement metric-aware prolongation
- add spherical gradient, divergence, and curl operators
- add spherical-to-Cartesian coordinate and vector output conversion

## Non-Goals

- no Cython spherical topology implementation in this milestone
- no pole ghost-cell exchange in this milestone
- no metric-aware AMR prolongation or restriction in this milestone
- no spherical vector calculus operators in this milestone
- no silent Cartesian interpretation of spherical data

## Success Criteria

The milestone is complete when a spherical `.dat` file can be opened without
being treated as Cartesian, native spherical block data can be inspected,
spherical metrics can be queried, unsupported operations fail clearly, progress
and follow-up status are visible, and all existing Cartesian behavior remains
unchanged.
