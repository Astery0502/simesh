"""Explicit pointwise recipes producing independently owned native fields."""

from collections.abc import Mapping
import numpy as np

from ..field_ops import _aligned_inputs, _layout, _admit_output
from ..fields import FieldDefinition, Fields, publish, require_fields, _field_index


class DerivedContext:
    """Read-only arrays for one leaf, including common valid support.

    Recipes must be pointwise: no spatial shifts, differentiation, reductions
    over spatial axes, or mutation. Use derivative() for spatial operations.
    """

    def __init__(self, bindings, leaf):
        self._bindings, self._leaf = bindings, leaf

    def field(self, name, *, group=None):
        """Select a name or component index; multiple inputs require group."""
        if group is None:
            if len(self._bindings) != 1:
                raise ValueError("select group explicitly for multiple input groups")
            group = next(iter(self._bindings))
        fields, box, columns = self._bindings[group]
        if type(name) is int or isinstance(name,np.integer):
            if not 0 <= name < len(fields.fields):
                raise ValueError("component index outside the field group")
            return fields.values[(fields.slot_of_leaf[self._leaf], *box, int(name))]
        if name not in columns:
            columns[name] = _field_index(fields.fields, name)
        return fields.values[(fields.slot_of_leaf[self._leaf], *box, columns[name])]


def derive(inputs, name, func, *, units="code", memory_limit=None):
    """Evaluate a pointwise recipe on Fields or a mapping of named Fields.

    The callback receives a DerivedContext once per leaf and returns a scalar
    or an array matching that leaf's interior plus common valid halo. All input
    groups must share the same Mesh and leaf coverage; slot order may differ.
    Their physical meaning and units must be compatible by caller choice.

    Output owns its values and retains no input arrays or callback. The budget
    includes inputs, output and two float64 result blocks, but cannot bound
    arbitrary allocations inside user callbacks. Nonfinite values propagate.
    """
    definition = FieldDefinition(name, units, "pointwise-derived")
    if not callable(func):
        raise TypeError("func must be a pointwise callable")
    return derive_many(inputs, (definition,), lambda ctx: {name: func(ctx)},
                       memory_limit=memory_limit)


def derive_many(inputs, definitions, func, *, memory_limit=None):
    """Evaluate one pointwise callback per leaf for multiple named outputs.

    Definitions is an ordered mapping of output names to unit labels, or a
    sequence of FieldDefinition objects. The callback returns a mapping with
    exactly these names; each value is a scalar or a same-shaped leaf array.
    Mapping order in the callback does not affect component order. Definitions
    supplied as a mapping receive the interpretation ``pointwise-derived``.

    Inputs, spatial semantics, common support and ownership follow derive().
    The budget includes input backing, output, all returned float64 blocks and
    one conversion block. Other callback allocations are the caller's concern.
    The callback and input Fields are not retained in the completed result.
    """
    if isinstance(inputs, Fields):
        groups = {"input": inputs}
    elif isinstance(inputs, Mapping) and inputs:
        groups = dict(inputs)
    else:
        raise TypeError("inputs must be Fields or a nonempty mapping of Fields")
    if not all(isinstance(key, str) and key for key in groups):
        raise ValueError("input group names must be nonempty strings")
    if not callable(func):
        raise TypeError("func must be a pointwise callable")
    if isinstance(definitions, Mapping):
        definitions = tuple(FieldDefinition(name, units, "pointwise-derived")
                            for name, units in definitions.items())
    else:
        definitions = tuple(definitions)
    if not definitions or not all(isinstance(value, FieldDefinition) for value in definitions):
        raise ValueError("provide nonempty output FieldDefinitions or a name-to-units mapping")
    names = tuple(value.name for value in definitions)
    if len(set(names)) != len(names):
        raise ValueError("duplicate output field names")
    fields = _aligned_inputs(groups.values())
    first = fields[0]
    halo, block_shape, boxes = _layout(fields)
    block_bytes = 8*int(np.prod(block_shape))
    _admit_output(fields, block_shape, len(definitions), memory_limit, "derive",
                  scratch=(len(definitions)+1)*block_bytes)
    # Geometry is invariant across callbacks; field names are resolved only on
    # first use, while values access still checks any borrowed input lifetime.
    bindings = {key: (value, box, {}) for (key, value), box in zip(groups.items(), boxes)}
    output = np.empty((len(first.leaf_ids), *block_shape, len(definitions)), dtype=np.float64)
    for slot, leaf in enumerate(first.leaf_ids):
        results = func(DerivedContext(bindings, leaf))
        for value in fields:
            require_fields(value)
        if not isinstance(results, Mapping) or set(results) != set(names):
            raise ValueError("recipe must return a mapping with exactly the defined output names")
        for column, name in enumerate(names):
            result = np.asarray(results[name], dtype=np.float64)
            if result.shape not in ((), block_shape):
                raise ValueError(f"recipe must return a scalar or block shape {block_shape}, got {result.shape}")
            output[slot, ..., column] = result
            del result
        del results
    for value in fields:
        require_fields(value)
    return publish(first.mesh, output, first.selection, definitions, halo, halo,
                   "pointwise(" + ",".join(value.scheme for value in fields) + ")",
                   tuple(value.source for value in fields))
