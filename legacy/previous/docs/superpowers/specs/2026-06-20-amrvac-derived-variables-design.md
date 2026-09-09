# AMRVAC Derived Variables Design

## Context

`simesh.amrvac` currently reads AMRVAC `.dat` variables from disk into
Morton/SFC block data with layout `(nleafs, nw, bx, by, bz)`. Public workflows
identify original fields primarily through `.dat` header indices, while names
are stored in `AMRVACDataSet.wnames`.

Derived variables do not have stable original file indices. They should
therefore be added as in-memory dataset behavior on `AMRVACDataSet`, not as
low-level `.dat` parsing behavior in `datio.py`.

## Goals

- Let users register derived cell-centered variables by name.
- Support simple arithmetic derived fields from loaded cell-centered fields.
- Support derivative-style fields that need ghost-cell-padded data, such as
  components of `J = curl(B)`, when ghost exchange has been performed.
- Let materialized derived fields flow through block and uniform-grid paths
  like loaded fields where the current dataset mode already supports those
  paths.
- Keep memory costs explicit and user-controlled.
- Preserve existing `field_indices` behavior for original `.dat` fields.

## Non-Goals

- Do not add a symbolic expression language in the first implementation.
- Do not make derived fields appear to be original fields from the input file.
- Do not compute derived arrays automatically at registration time.
- Do not relax existing spherical read-only, uniform sampling, or write-back
  restrictions.
- Do not refactor the AMR mesh or `.dat` parser outside what the feature needs.

## Public API Shape

The feature is centered on `AMRVACDataSet`.

```python
ds.register_derived(
    "p",
    func,
    dependencies=["rho", "e"],
    requires_ghosts=False,
)

ds.register_derived(
    "j1",
    func,
    dependencies=["b2", "b3"],
    requires_ghosts=True,
)

ds.materialize_fields(["p", "j1"])
ds.drop_derived_fields(["p"])
```

`register_derived()` stores a named recipe. It does not compute arrays.

`materialize_fields()` computes the requested derived names and appends their
arrays to the loaded field axis of `ds.data`. Each derived function returns one
scalar block field with shape `(nleafs, bx, by, bz)`. Multi-component derived
quantities are represented as separate names, such as `j1`, `j2`, and `j3`.

`drop_derived_fields()` removes materialized derived columns from `ds.data` and
updates the loaded-name bookkeeping.

Existing index-based APIs remain valid. New name-based selection should be
added where needed:

```python
ds.blocks(field_names=["rho", "p"])
ds.uniform_grid(nx, field_names=["p"])
ds.uniform_full(field_names=["rho", "p"])
ds.write_datfile(path, field_names=["rho", "p"], overwrite=True)
```

`field_indices` continues to mean original `.dat` header indices only.
`field_names` may refer to loaded original fields or materialized derived
fields. If both selectors are supplied to the same method, the method should
raise `ValueError` rather than merge two selection models.

## Dataset State

Keep original file metadata separate from in-memory loaded field state.

```python
ds.wnames                  # original file/header field names
ds.loaded_field_indices    # original file/header indices currently loaded
ds.loaded_field_names      # names corresponding to columns in ds.data
ds.derived_definitions     # registered derived recipes
ds.derived_field_names     # materialized derived names
```

`ds.nw` and `ds.metadata["nw"]` continue to describe original input metadata
unless an output header is explicitly assembled for a write operation.

After `ds.materialize_fields(["p"])`, `ds.data` has one additional field
column and `ds.loaded_field_names` includes `"p"`. Downstream code that works
from loaded field columns can then use `"p"` like any other loaded field.

## Derived Function Context

Derived functions should receive a small context object rather than the raw
dataset object as their primary input.

```python
ctx.field("rho")           # interior block field, shape (nleafs, bx, by, bz)
ctx.padded_field("b1")     # padded field, requires ghost cells
ctx.spacing                # block-local cell spacing helper
ctx.dataset                # read-only metadata access
```

The context resolves names against `loaded_field_names`. `ctx.field(name)`
returns the interior block field for a loaded original or already materialized
derived field. `ctx.padded_field(name)` requires `ghost_width > 0` and returns
the corresponding padded field with ghost cells.

Derivative helpers such as current density should be ordinary registered
functions built over this context. The first implementation may add small
finite-difference helper utilities, but the dataset core should only manage
registration, dependency validation, materialization, and selection.

## Ghost-Cell Requirements

For a recipe with `requires_ghosts=True`, `materialize_fields()` must require:

- `ds.ghost_width > 0`
- `ds.data` loaded
- `ds.mesh` initialized
- ghost-cell storage available for all dependencies

Before computing such a field, `materialize_fields()` should call
`ds.exchange_ghost_cells()` so padded data reflects the current interior data.
If the dataset mode blocks ghost exchange, the existing mode-specific error
should surface.

For spherical datasets, existing restrictions remain authoritative. Native
block inspection with supported ghost exchange may materialize ghost-dependent
block fields if padded data is available. Uniform sampling and write-back stay
blocked where they are currently blocked.

## Materialization And Memory Policy

Memory behavior is explicit:

- Registration stores only recipes.
- Materialization computes and stores arrays in `ds.data`.
- The appended `ds.data` column is the cache; do not keep a second separate
  cache for the same array.
- `drop_derived_fields()` releases materialized columns by compacting `ds.data`.
- `load_data()` clears materialized derived fields because the base field set
  may have changed.
- Re-registering a name clears the old materialized result for that name.

This makes memory cost proportional to the derived fields the user explicitly
materializes. Large derived arrays are not retained unless the user chooses to
retain them.

## Validation And Errors

`register_derived()` should validate:

- name is a non-empty string
- name does not collide with an original field name
- dependencies are names, not original field indices
- `func` is callable

`materialize_fields()` should validate:

- every requested derived name is registered
- every dependency is currently loaded or already materialized
- each result has shape `(nleafs, bx, by, bz)`
- each result can be converted to contiguous `float64`
- ghost-cell prerequisites are satisfied for recipes that require ghosts

Errors should be direct `ValueError` or `KeyError` messages that name the
missing field, derived recipe, or invalid result shape.

## Downstream Behavior

`blocks(field_names=[...])` should return selected loaded columns in SFC layout.
For compatibility, `blocks()` without selectors keeps returning all currently
loaded columns.

`uniform_grid(..., field_names=[...])` and
`uniform_full(field_names=[...])` should work for materialized derived fields
because they are columns in `ds.data`. Existing `field_indices` behavior is
unchanged and still maps only original file/header indices through
`loaded_field_indices`.

Writing derived fields should be opt-in by name. `write_datfile()` without
`field_names` should keep current behavior. With `field_names`, the output
header should use those names and `nw=len(field_names)`, including materialized
derived names. This does not mutate the input metadata.

## Minimal Implementation Boundary

The minimal implementation should touch the canonical AMRVAC dataset layer and
its tests:

- `src/simesh/amrvac/amrvac_dataset.py`
- `tests/amrvac/test_amrvac_dataset.py`
- public docs only if the API is implemented, not during the design-only phase

No Cython changes are required for the first version unless profiling later
shows a specific derivative helper must move into the compiled layer.

## Test Coverage

Add focused tests for:

- registering and materializing an arithmetic derived field from two loaded
  original fields
- rejecting materialization when dependencies are missing
- rejecting a derived result with the wrong shape
- dropping a derived field and verifying the data column and loaded name are
  removed
- clearing materialized derived fields after `load_data()`
- rejecting a ghost-required derived field when `ghost_width == 0`
- computing a ghost-required derived field that reads padded data after ghost
  exchange
- selecting materialized fields by name through `blocks()`
- selecting materialized fields by name through one uniform-grid path
- preserving existing index-based tests unchanged

## Success Criteria

- Users can register named derived variables without computing them.
- Users can explicitly materialize selected derived variables and later drop
  them.
- Materialized derived variables can be selected by name through block and
  supported uniform-grid workflows.
- Existing callers that use `field_indices` behave as before.
- Memory behavior is documented and controlled by explicit materialization.
- Ghost-dependent derived variables fail clearly unless ghost-cell storage is
  available and exchange can run.
