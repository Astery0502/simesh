"""Explicit pointwise recipes producing independently owned native fields."""

from collections.abc import Mapping
import numpy as np

from ..field_ops import _aligned_inputs, _layout, _admit_output
from ..fields import FieldDefinition, Fields, publish, require_fields, _field_index


class DerivedContext:
    """Read-only arrays for one leaf and the common valid support of all input groups.

    Notes
    -----
    Callbacks must be pointwise: no mutation, spatial shifts, differentiation or
    reduction over spatial axes. Use derivative or reduction APIs for spatial work.
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

    Parameters
    ----------
    inputs : Fields or mapping of str to Fields
        One group or named groups sharing Mesh identity and leaf coverage; slot order
        may differ.
    name : str
        Output field name.
    func : callable
        Called once per leaf with [DerivedContext][simesh.DerivedContext]; return a scalar or array matching its
        interior plus common valid halo.
    units : str
        Unit label for the returned values, without automatic conversion.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent pointwise outputs using common valid support; inputs/callback are not retained.

    Notes
    -----
    Callback allocations beyond the accounted output blocks remain the caller's responsibility; nonfinite recipe values propagate.
    """
    definition = FieldDefinition(name, units, "pointwise-derived")
    if not callable(func):
        raise TypeError("func must be a pointwise callable")
    return derive_many(inputs, (definition,), lambda ctx: {name: func(ctx)},
                       memory_limit=memory_limit)


def derive_many(inputs, definitions, func, *, memory_limit=None):
    """Evaluate one pointwise callback per leaf for multiple named outputs.

    Parameters
    ----------
    inputs : Fields or mapping of str to Fields
        One group or named groups sharing Mesh identity and leaf coverage; slot order
        may differ.
    definitions : mapping or sequence of FieldDefinition
        Ordered names/units or definitions; this order determines output columns. A
        mapping uses pointwise-derived interpretation.
    func : callable
        Called once per leaf with [DerivedContext][simesh.DerivedContext]; return exactly the defined names, each
        mapped to a scalar or matching leaf array. Dictionary order does not change
        output order.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent pointwise outputs using common valid support; inputs/callback are not retained.

    Notes
    -----
    Callback allocations beyond the accounted output blocks remain the caller's responsibility; nonfinite recipe values propagate.
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
