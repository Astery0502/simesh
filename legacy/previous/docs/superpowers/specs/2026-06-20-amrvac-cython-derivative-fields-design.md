# AMRVAC Cython Derivative Fields Design

## Context

`AMRVACDataSet` now has a generic derived-field workflow for in-memory fields:
users register recipes by name and explicitly materialize selected fields into
the loaded field axis. This is appropriate for cell-local algebraic fields, but
derivative and stencil fields such as current density, gradients, divergence,
and vorticity should not be implemented as NumPy slice expressions over padded
views for the production path.

Derivative fields need ghost-cell-padded input data and repeatedly access
neighboring cells. A Python/NumPy slice implementation is useful for validation,
but it creates large temporaries and walks memory multiple times. The canonical
path should keep the public derived-field workflow while moving derivative
stencils into a general Cython kernel.

## Goals

- Keep `register_derived()` as the public interface for ordinary cell-local
  algebraic derived fields.
- Add a derivative-oriented registration path for first-derivative stencil
  fields without hard-coding current density as a special case.
- Express derivative fields declaratively as linear combinations of directional
  derivatives.
- Materialize derivative fields through `materialize_fields(...)`, so users have
  one materialization workflow.
- Compute only the derivative fields requested by the user.
- Batch requested derivative fields when possible.
- Avoid repeated ghost-cell exchange during a derivative materialization batch.
- Use a Cython kernel for derivative stencil loops over padded mesh data.
- Make derived-field ghost-cell validity explicit.

## Non-Goals

- Do not add a symbolic expression language.
- Do not change `register_derived()` to compute eagerly.
- Do not require current density components to be materialized together.
- Do not fully exchange all ghost layers for materialized derived fields.
- Do not track arbitrary direct user mutation of `ds.data`; the design assumes
  users do not mutate dataset internals directly.
- Do not refactor unrelated AMRVAC loading, `.dat` parsing, or mesh topology.

## Public API Shape

Cell-local derived fields continue to use the existing Python callable path:

```python
ds.register_derived(
    "p",
    lambda ctx: ctx.field("e") - 0.5 * ctx.field("rho"),
    dependencies=["rho", "e"],
)
```

Stencil fields use a new declarative derivative registration path:

```python
ds.register_derivative(
    "j1",
    terms=[
        ("b3", "y", +1.0),
        ("b2", "z", -1.0),
    ],
)

ds.register_derivative(
    "dvx_dx",
    terms=[
        ("v1", "x", +1.0),
    ],
)

ds.materialize_fields(["j1", "dvx_dx"])
```

Each derivative field is one scalar output. Multi-component physical quantities
are represented as separate fields. For current density:

```text
j1 = + d(b3)/dy - d(b2)/dz
j2 = + d(b1)/dz - d(b3)/dx
j3 = + d(b2)/dx - d(b1)/dy
```

Users can request any subset:

```python
ds.materialize_fields(["j1"])
```

That request computes only `j1`; it does not compute `j2` or `j3`.

## Derivative Term Format

The first implementation supports a small term tuple:

```text
(field_name, axis, coefficient)
```

- `field_name` is a loaded field name, not an original field index.
- `axis` is `"x"`, `"y"`, `"z"` or the integer aliases `0`, `1`, `2`.
- `coefficient` is converted to `float64`.

One derivative output is the linear combination:

```text
output = sum(coefficient * partial(field_name) / partial(axis))
```

This format covers simple partial derivatives, gradients, divergence, curl
components, and vorticity components while keeping the dataset and Cython
interfaces compact.

## Dataset State

The dataset should distinguish generic Python derived recipes from derivative
recipes. The existing `derived_field_names` remains the list of materialized
derived fields. The existing `loaded_field_names` remains the in-memory column
namespace shared by original and materialized fields.

Derivative recipe metadata should record:

- output name
- normalized terms
- dependencies derived from term field names
- `requires_ghosts=True`
- derivative backend identifier, initially the Cython central-difference backend

The recipe should not store a Python function that performs NumPy slicing.

## Materialization Flow

`materialize_fields(names)` remains the only public materialization method.
Internally it should:

1. Normalize requested names.
2. Skip names already present in `derived_field_names`.
3. Resolve registered recipe types for the remaining names.
4. Materialize Python derived recipes through the existing callable path.
5. Group requested derivative recipes into compatible derivative batches.
6. Ensure original input fields have valid exchanged ghost cells at most once
   per derivative batch.
7. Pass the derivative batch to Cython.
8. Append only requested derivative outputs to `ds.data`.
9. Update `loaded_field_names` and `derived_field_names`.
10. Refresh or rebuild mesh field-axis storage only as needed for the newly
    materialized columns.

For a mixed request such as:

```python
ds.materialize_fields(["p", "j1", "j2"])
```

the dataset may materialize the cell-local `p` through Python and then
materialize `j1` and `j2` through one derivative batch if their specs are
compatible.

## Cython Backend

The Cython side should implement a general first-derivative stencil primitive,
not a current-density-specific routine.

The kernel receives:

- padded mesh storage
- input field column indices
- output slots or output arrays
- term-to-output mapping
- derivative axes
- coefficients
- block size
- ghost width
- per-block spacing from mesh metadata

The conceptual loop is:

```text
for each requested output field:
    for each block and target cell:
        value = 0
        for each term belonging to the output:
            value += coefficient * central_difference(input_field, axis)
        output = value
```

The first backend uses centered finite differences for first derivatives. The
design leaves room for later backend identifiers if higher-order stencils or
non-Cartesian metrics are added, but those are outside this spec.

## Ghost-Cell Policy

The design assumes users do not directly mutate `ds.data`. Padded data lives
with the mesh/original loaded data. If original data is replaced through dataset
loading paths, the padded storage is rebuilt or discarded by those paths.

Derivative materialization must ensure the original input fields have valid
ghost cells before the Cython stencil reads padded data. It must not exchange
ghost cells once per derivative output. A materialization request containing
multiple derivative fields should perform at most one ghost exchange for the
compatible derivative batch, and should reuse already exchanged padded storage
when available.

Derived output ghost validity is intentionally narrower than input ghost
validity:

- Cell-local fields registered with `register_derived()` are valid in the
  interior. They do not claim filled ghost cells.
- Derivative fields require exchanged ghost cells on their input fields.
- A first-derivative output can validly fill the interior plus
  `nghostcells - 1` ghost layers.
- The outermost derived-output ghost layer remains zero.
- This must be documented so users do not assume materialized derived fields
  have fully exchanged ghost cells.

For example, with `ghost_width = 2`:

```text
input b1/b2/b3:
    two valid ghost layers

derived j1/j2/j3:
    interior valid
    one valid ghost layer
    outermost ghost layer is zero
```

Derivative materialization should require `ghost_width >= 2` for this first
design. A dataset opened with fewer ghost cells should fail clearly before the
Cython kernel runs.

## Validation And Errors

`register_derivative()` should validate:

- output name is a non-empty string
- output name does not collide with an original `.dat` field name
- terms is a non-empty sequence
- each term has exactly `(field_name, axis, coefficient)`
- each field name is a non-empty string
- each axis is valid for the current dataset dimensionality
- each coefficient can be converted to `float64`

`materialize_fields()` should validate:

- requested derivative names are registered
- derivative dependencies are loaded or already materialized
- `ghost_width >= 2` for derivative recipes
- mesh and padded storage are available before Cython execution
- each Cython output has the expected interior shape

Errors should be direct:

- `KeyError` for unknown requested recipe names
- `ValueError` for invalid derivative specs
- `ValueError` for missing dependencies
- `ValueError` for insufficient ghost-cell storage

Error messages should name the derivative field and the problematic dependency,
axis, or ghost-cell requirement.

## Testing

Focused tests should cover:

- registering and materializing a single partial derivative such as `dvx_dx`
- registering `j1`, `j2`, and `j3`, then materializing only `j1`
- verifying that unrequested current components are not computed or appended
- materializing `["j1", "j2"]` in one request and verifying ghost exchange is
  performed at most once for that derivative batch
- comparing derivative output against a simple linear field with known exact
  derivative
- rejecting derivative materialization with `ghost_width < 2`
- rejecting invalid axes and malformed term specs
- rejecting missing dependencies
- verifying derivative padded output exposes only `nghostcells - 1` valid ghost
  layers and leaves the outermost layer zero
- verifying cell-local derived fields do not claim filled ghost cells

## Success Criteria

The design is implemented when:

- users can register derivative fields declaratively
- `materialize_fields(...)` handles both Python derived fields and Cython
  derivative recipes
- derivative materialization batches compatible requested outputs
- ghost-cell exchange is not repeated per derivative output
- requested derivative fields are computed independently, so current-density
  components can be requested one at a time
- Cython computes derivative stencils from padded mesh data without NumPy slice
  temporaries
- derived-output ghost-cell limitations are documented and tested
