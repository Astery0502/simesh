"""Explicit pointwise recipes producing independently owned fields."""

from collections.abc import Mapping
import numpy as np

from .._field_data import PointwiseLayout, require_field_data
from ..fields import FieldDefinition, Fields, _field_index


class DerivedContext:
    """Read-only arrays for one pointwise batch and the common valid support of all input groups.

    Notes
    -----
    Callbacks must be pointwise: no mutation, spatial shifts, differentiation or
    reduction over spatial axes. Use derivative or reduction APIs for spatial work.
    """

    def __init__(self, bindings):
        self._bindings = bindings

    def field(self, name, *, group=None):
        """Select a name or component index; multiple inputs require group."""
        if group is None:
            if len(self._bindings) != 1:
                raise ValueError("select group explicitly for multiple input groups")
            group = next(iter(self._bindings))
        fields, data, columns = self._bindings[group]
        if type(name) is int or isinstance(name,np.integer):
            if not 0 <= name < len(fields.fields):
                raise ValueError("component index outside the field group")
            return data[..., int(name)]
        if name not in columns:
            columns[name] = _field_index(fields.fields, name)
        return data[..., columns[name]]


def derive(inputs, name, func, *, units="code", memory_limit=None):
    """Evaluate a pointwise recipe on completed native field groups.

    Parameters
    ----------
    inputs : Fields or mapping
        One group or named groups sharing Mesh identity and requested coverage;
        storage slot order may differ.
    name : str
        Output field name.
    func : callable
        Called with [DerivedContext][simesh.DerivedContext]; return a scalar or
        matching batch array including common valid halo.
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
    """Evaluate one pointwise callback per batch for multiple named outputs.

    Parameters
    ----------
    inputs : Fields or mapping
        One group or named groups sharing Mesh identity and requested coverage;
        storage slot order may differ.
    definitions : mapping or sequence of FieldDefinition
        Ordered names/units or definitions; this order determines output columns. A
        mapping uses pointwise-derived interpretation.
    func : callable
        Called with [DerivedContext][simesh.DerivedContext]; return exactly the
        defined names, each mapped to a scalar or matching batch array.
        Dictionary order does not change output order.
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
        raise TypeError("inputs must be Fields or a nonempty mapping")
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
    layout = PointwiseLayout(groups.values())
    fields = layout.inputs
    layout.admit(len(definitions), memory_limit, "derive", scratch_per_cell=8*(len(definitions)+1))
    output = layout.allocate(len(definitions))
    columns = tuple({} for _ in fields)
    for index, arrays in layout.chunks():
        bindings = {key: (value, data, lookup) for key, value, data, lookup in
                    zip(groups, fields, arrays, columns)}
        results = func(DerivedContext(bindings))
        for value in fields:
            require_field_data(value)
        if not isinstance(results, Mapping) or set(results) != set(names):
            raise ValueError("recipe must return a mapping with exactly the defined output names")
        shape = arrays[0].shape[:-1]
        target = (index, Ellipsis)
        for column, name in enumerate(names):
            result = np.asarray(results[name], dtype=np.float64)
            if result.shape not in ((), shape):
                raise ValueError(f"recipe must return a scalar or batch shape {shape}, got {result.shape}")
            output[(*target, column)] = result
            del result
        del bindings, results
    return layout.publish(output, definitions,
                          scheme="pointwise(" + ",".join(value.scheme for value in fields) + ")",
                          source=tuple(value.source for value in fields))
