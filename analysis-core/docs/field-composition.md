# Compose completed field groups

Use these focused imports; package-level re-exports are not required:

```python
from simesh.field_ops import select_fields, merge_fields
from simesh.operators.derived import derive_many
```

All three operations return independently owned, readonly `Fields`. They accept
completed owned fields and live borrowed batches. No option creates a shared
view or extends a batch lease. Closing a Source, advancing a prepared iterator,
or dropping the input Fields does not invalidate a completed result.

## Select, reorder, and rename components

```python
# ready contains density, magnetic components, and temperature in any order.
magnetic = select_fields(ready, ("b1", "b2", "b3"))
thermodynamics = select_fields(ready, ("temperature", "rho"),
                              names=("temperature_K", "density"))
# Mixed names and zero-based indices are accepted, including nonadjacent ones.
subset = select_fields(ready, ("temperature", 0))
```

`components=None` copies all components. A single name or integer selects one
component. Negative indices, booleans, empty selections and repeated component
indices are rejected. A name must identify exactly one input component. For
ambiguous input names, select by index and supply distinct output names.

`names`, when supplied, is an ordered sequence with one name per output
component. It changes only names; units and interpretations are preserved.
Output names must be unique and nonempty. Selecting fields with different unit
labels is supported and performs no conversion.

## Merge groups by leaf identity

```python
combined = merge_fields((thermodynamics, magnetic))
# Resolve a known collision explicitly with a complete output-name sequence.
comparison = merge_fields((density_before, density_after),
                           names=("density_before", "density_after"))
```

Pass a nonempty sequence of Fields. Components follow group order and then the
component order within each group. Duplicate output names raise an error;
there is no implicit suffix, overwrite or deduplication policy. Rename selected
inputs first, or supply all output names through `names`.

Inputs must share the **same Mesh object** and exactly the same leaf coverage.
Matching shapes or equivalent but distinct meshes do not establish compatibility.
Input selection order and physical slot order may differ. Every value is read
through its source `slot_of_leaf` directory. Output slots are packed in the
first input's selection order, so `interior()` works even when an input requires
per-leaf window access. Requested region metadata follows that first selection.

Mixed units and different preparation schemes are allowed explicitly. The
caller is responsible for compatible physical meanings, times, normalizations
and numerical reconstructions. Merge does not resample, prepare missing support,
or fill a union of different leaf coverages.

## Evaluate several pointwise outputs together

```python
import numpy as np


def magnetic_products(ctx):
    squared = sum(ctx.field(name)**2 for name in ("b1", "b2", "b3"))
    return {"b_squared": squared,
            "magnitude": np.sqrt(squared),
            "nonzero": squared > 0}


products = derive_many(
    magnetic,
    {"b_squared": "G^2", "magnitude": "G", "nonzero": "1"},
    magnetic_products,
)
```

The callback runs **once per leaf**, returning a mapping with exactly the
specified output names. Output order follows `definitions`, independently of
callback mapping order. Each returned value must be a scalar or an array with
the leaf interior plus common valid halo shape. There is no broadcasting of
other array shapes. Values are converted to float64; nonfinite values propagate.

The second argument, `definitions`, accepts an ordered name-to-unit mapping,
which sets interpretation to `pointwise-derived`. Alternatively pass an ordered
sequence of `FieldDefinition` objects to specify all metadata explicitly.
Empty definitions and duplicate names are rejected before running callbacks.

Multiple input groups use the same explicit binding rules as single-output
`derive`:

```python
products = derive_many(
    {"b": magnetic, "curl": curl_b},
    {"b_dot_curl": "G^2 / coordinate-length", "curl_squared": "G^2 / coordinate-length^2"},
    lambda ctx: {
        "b_dot_curl": sum(ctx.field(b, group="b") * ctx.field(c, group="curl")
                          for b, c in zip(("b1", "b2", "b3"),
                                          ("curl_x", "curl_y", "curl_z"))),
        "curl_squared": sum(ctx.field(c, group="curl")**2
                            for c in ("curl_x", "curl_y", "curl_z")),
    },
)
```

This requires the same Mesh and leaf coverage, with independent slot alignment.
Only supply groups actually needed: every supplied group limits common support.
Existing `derive(inputs, name, func, units=...)` calls still return one component
from a scalar or array callback result.

Recipes are strictly **pointwise**. Arrays are readonly, but the API cannot
statically enforce a Python callback's spatial semantics. Spatial shifts,
`np.gradient`, filtering and reductions across spatial axes violate this
contract. Use `derivative` or another explicit spatial operator. Summing or
combining different components at the same point is valid. Nonlinear formulas
act on already prepared values, including valid support; evaluating before
preparation produces a different reconstruction.

## Valid support, storage, and memory

Selection retains its input's `valid_halo`. Merge and derivation retain the
minimum `valid_halo` across inputs. Outputs allocate exactly this halo, with
`storage_halo == valid_halo`. Invalid padding is never copied or evaluated,
and regional edges never become physical boundaries. Zero-halo outputs allow
interior operations but cannot satisfy a consumer requiring valid support.

Outputs use compact C-contiguous `(slot, x, y, z, component)` float64 storage,
without a NumPy base pointing to a larger parent allocation. Only Mesh,
Selection, field definitions and small provenance tokens are shared. Input
Fields, their value arrays, and callbacks are not retained by the output.

All operations accept `memory_limit`. Admission occurs before output allocation
and before recipe execution, counting Mesh, the unique underlying input array
allocations, output and leaf directory. A small view into a large input backing
is charged for that backing. Multi-output recipes additionally reserve all
returned float64 blocks plus one conversion block; the single-output wrapper
reserves two blocks. These are controlled-array estimates, not process-RSS
caps. Arbitrary allocations and retained arrays inside user callbacks, or other
products held by the caller, are outside the estimate. A failure publishes no
partial Fields and never changes precision or requested components.

## Provenance and magnetic consumers

Selection preserves the source token and preparation scheme. Merge records
ordered source tokens and a `merge(...)` scheme; recipes record ordered source
tokens and a `pointwise(...)` scheme. These describe ancestry, without retaining
Sources or their arrays. Each operation assigns a fresh `value_identity` and
clears `derivation`. This intentionally conservative rule also applies to an
identity selection, renaming only, or a one-group merge.

A curl certificate binds a particular vector group, its definitions and its
scheme. Compute curl **after** composing the magnetic vector:

```python
import simesh as sm
from simesh import applications as app

magnetic = select_fields(combined, ("b1", "b2", "b3"))
curl_b = sm.curl(magnetic)
map_result = app.surface_diagnostics(magnetic, points, curl_field=curl_b)
selected = map_result.threshold(q_min=10.)
lines = app.trace(magnetic, selected, step=.01, max_steps=2000)
```

The vector must have three ordered components with common units. An old curl
from the pre-selection/pre-merge group is rejected. Selecting, reordering or
merging curl values also clears its certificate: it remains a valid field group
but cannot stand in for the certified `curl_field` argument. This prevents a
reordered B or curl from passing provenance checks by shape alone.

QSL/twist diagnostics retain no trajectories. Selection is followed by explicit
application tracing, whose default follows both magnetic directions. These
helpers do not change MHD recovery, thermal conventions or result-file schemas.
