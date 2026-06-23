# AMRVAC Field Column Metadata Design

## Context

The derived-variable implementation appends materialized fields to the field
axis of `AMRVACDataSet.data`. Current internal logic distinguishes original
loaded fields from materialized derived fields by combining several pieces of
state:

```python
len(ds.loaded_field_indices)
ds.loaded_field_names
ds.derived_field_names
ds.derived_field_ghost_valid_layers
```

That works while all original fields are first and all derived fields are
afterward, but it makes column ordering part of the internal contract. The next
cleanup should make each loaded data column describe its own ownership and
ghost-cell validity explicitly.

This spec covers only the Python dataset metadata cleanup. The Cython
derivative-kernel hot-loop optimization is a separate future performance spec.

## Goals

- Replace hidden original-vs-derived column ownership inference with explicit
  per-column metadata.
- Preserve current public behavior for loaded field selectors, materialized
  derived fields, and `.dat` writing.
- Keep the existing documented compatibility attributes available as views.
- Make ghost-cell eligibility and valid ghost-layer counts explicit.
- Add a general ghost-width invariant: ghost storage is either disabled with
  `ghost_width == 0` or enabled with `ghost_width >= 2`.
- Keep the implementation narrow and focused on the canonical AMRVAC dataset
  path.

## Non-Goals

- Do not change the public derived-field API.
- Do not change the user-visible meaning of `field_indices` or `field_names`.
- Do not add support for full ghost exchange of materialized derived fields.
- Do not allow derivative fields to depend on materialized derived fields.
- Do not optimize Cython derivative loops in this patch.
- Do not refactor the `.dat` parser, forest reconstruction, or mesh topology.

## Proposed Approach

Introduce a small internal column metadata record:

```python
@dataclass(frozen=True)
class LoadedFieldColumn:
    name: str
    source_kind: str
    original_index: int | None
    ghost_valid_layers: int
```

`AMRVACDataSet` owns `self._field_columns`, a list whose order exactly matches
the field axis in `self.data` and the final field axis in `mesh.padded_view()`
when padded storage exists.

`source_kind` uses two values:

- `"original"` for columns loaded from the AMRVAC header field set
- `"derived"` for materialized derived or derivative columns

Original records must have an integer `original_index`. Derived records must
have `original_index=None`.

This makes `_field_columns` the implementation source of truth. Existing
attributes remain compatibility views:

```python
ds.loaded_field_names
ds.loaded_field_indices
ds.derived_field_names
ds.derived_field_ghost_valid_layers
```

New internal logic should read and update `_field_columns` directly rather than
inferring ownership from list lengths or slices.

## Compatibility Surface

The compatibility attributes keep their current meanings:

- `loaded_field_names`: all currently loaded column names in data-column order
- `loaded_field_indices`: original AMRVAC header indices for loaded original
  columns, in data-column order
- `derived_field_names`: materialized derived column names in data-column order
- `derived_field_ghost_valid_layers`: mapping from derived name to valid
  ghost-layer count

These attributes should be backed by `_field_columns`. They should behave as
read-only compatibility properties for normal use. If transitional setters are
needed for construction paths or existing tests, the setters should rebuild
`_field_columns` conservatively and should not become the preferred internal
mutation path.

## Ghost-Width Policy

`ghost_width` has two valid modes:

- `0`: ghost-cell storage is disabled
- `>= 2`: ghost-cell storage is enabled

`ghost_width < 0` remains invalid. `ghost_width == 1` should be rejected during
dataset construction with a clear `ValueError`.

This is a general dataset invariant, not only a derivative-field rule.
Derivative materialization still requires ghost storage, but the constructor
invariant guarantees that enabled ghost storage has sufficient width.

## Data Flow

### File-Backed Loading

`load_data(field_indices=...)` replaces `self.data` and resets `_field_columns`
to one `"original"` record per loaded field.

For each loaded field:

- `name` is `self.wnames[original_index]`
- `source_kind` is `"original"`
- `original_index` is the selected AMRVAC header index
- `ghost_valid_layers` is `ghost_width` when `ghost_width >= 2`, otherwise `0`

As today, `load_data(...)` clears materialized derived fields because the base
loaded field set may have changed.

### Uniform Dataset Construction

`load_from_uniform(...)` and `_dataset_from_uniform_components(...)` initialize
`_field_columns` with original records for every provided `w_name`, matching
the existing synthetic dataset behavior.

### Python Derived Materialization

Appending Python derived results appends `"derived"` column records:

- `name` is the derived field name
- `source_kind` is `"derived"`
- `original_index` is `None`
- `ghost_valid_layers` is `0`

Materializing Python derived fields must not trigger ghost exchange.

### Derivative Materialization

Derivative fields require ghost storage, so they require `ghost_width >= 2` by
construction. Their appended `"derived"` column records use:

- `original_index=None`
- `ghost_valid_layers=ghost_width - 1`

This preserves the current contract that derivative padded output has one
fewer valid ghost layer than the original padded input and that the outermost
derived-output ghost layer remains invalid.

Derivative materialization must continue to reject dependencies that are not
original loaded fields.

### Dropping Derived Fields

`drop_derived_fields(...)` should compute keep columns from `_field_columns`,
compact `self.data`, compact `_field_columns`, and refresh padded mesh storage
with the existing name-based copy behavior.

### Writing

`write_datfile()` without `field_names` should write columns whose metadata has
`source_kind == "original"`. Output names come from the original/header names
for those records.

`write_datfile(field_names=[...])` remains opt-in and may include materialized
derived fields by name. This does not mutate the input metadata.

## Selector Behavior

`field_indices` continues to mean original AMRVAC header indices only.

`field_names` continues to mean currently loaded column names, including
materialized derived fields.

When resolving `field_indices`, the dataset should search only metadata records
with `source_kind == "original"`. When resolving `field_names`, the dataset
should search all loaded column metadata records.

Supplying both selector forms to the same method remains invalid.

## Ghost Eligibility

Ghost-required Python derived recipes and derivative terms should resolve their
dependencies through `_field_columns`.

The dependency is eligible only when:

- the dependency name is loaded
- the matching column has `source_kind == "original"`
- the dataset has padded storage available
- the original column has enough `ghost_valid_layers` for the operation:
  at least one valid layer for generic ghost-required Python recipes, and at
  least two valid layers for first-derivative recipes

Materialized derived fields remain selectable for interior block, uniform-grid,
and opt-in write workflows, but they do not gain a full ghost-cell exchange
contract from this metadata cleanup.

## Validation And Errors

Validation should be direct and early:

- Dataset construction rejects `ghost_width < 0` and `ghost_width == 1`.
- `_field_columns` length matches `self.data.shape[1]` whenever `self.data` is
  loaded.
- Loaded column names are unique.
- Original metadata records have integer `original_index` values.
- Derived metadata records have `original_index=None`.
- Selector-by-index paths resolve only original records.
- Ghost-required dependency checks use column metadata rather than column
  position.

Errors should remain ordinary `ValueError` or `KeyError` exceptions with
messages that name the invalid selector, missing field, or invalid ghost-width
mode.

## Implementation Boundary

Expected files:

- `src/simesh/amrvac/amrvac_dataset.py`
- `src/simesh/amrvac/derived_fields.py`
- `src/simesh/amrvac/amrvac_uniform.py`
- `tests/amrvac/test_amrvac_dataset.py`

Documentation updates are optional unless public behavior text changes. The
intended public behavior should not change, so docs may only need a small note
if the ghost-width invariant is user-visible.

No Cython source changes are required for this metadata patch.

## Test Coverage

Focused tests should cover:

- `ghost_width=1` is rejected during dataset construction.
- Existing compatibility properties return the same values as before.
- `field_indices` resolves only original metadata records.
- `field_names` resolves both original and materialized derived records.
- Default `write_datfile()` behavior selects source-kind `"original"` records,
  not the first `len(loaded_field_indices)` columns by assumption.
- Ghost-required Python derived recipes reject materialized derived
  dependencies through metadata source kind.
- Derivative recipes reject materialized derived dependencies through metadata
  source kind.
- Dropping one or more derived fields compacts both `self.data` and
  `_field_columns`.
- Reloading data resets `_field_columns` to the newly loaded original fields.

Run the focused AMRVAC dataset tests after implementation:

```bash
PYTHONPATH=src /Users/astery/science/simesh/.venv/bin/python -m pytest \
  tests/amrvac/test_amrvac_dataset.py
```

Because this patch should not touch Cython, a Cython rebuild is not required
unless the implementation unexpectedly changes compiled sources.
