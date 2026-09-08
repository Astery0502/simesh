"""Centered derivatives of extended inputs, retaining independent outputs."""

import numpy as np
from dataclasses import replace

from .._validation import admit, workers_count, indices
from .._execution import worker_context, run_ranges
from ..fields import FieldDefinition, publish, require_fields, _field_index


def derivative(fields, terms, definitions, *, workers=1, memory_limit=None):
    """Sum ordered (component, axis, coefficient) terms for each output field.

    Components accept indices or unique field names; axes accept 0/1/2 or
    "x"/"y"/"z". A derivative consumes one valid layer. Invalid padding is
    not differentiated. No physical current or unit conversion is inferred.
    """
    from .._kernels.native import differentiate
    fields = require_fields(fields, halo=1)
    workers_count(workers)
    definitions = tuple(definitions)
    if (not definitions or len(terms) != len(definitions) or
            not all(isinstance(definition, FieldDefinition) for definition in definitions)):
        raise ValueError("one nonempty term list per defined output required")
    rows, weights = [], []
    for output, entries in enumerate(terms):
        if not entries:
            raise ValueError("a derivative output requires a term")
        for component, axis, coefficient in entries:
            if isinstance(component, str):
                component = _field_index(fields.fields, component)
            if isinstance(axis, str):
                axis = {"x": 0, "y": 1, "z": 2}.get(axis.lower(), -1)
            if (not isinstance(component, (int, np.integer)) or
                    not 0 <= component < len(fields.fields) or
                    not isinstance(axis, (int, np.integer)) or not 0 <= axis < 3 or
                    not np.isfinite(coefficient)):
                raise ValueError("invalid derivative component, axis or coefficient")
            rows.append((output, component, axis))
            weights.append(coefficient)
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    h = fields.valid_halo - 1
    shape = (len(fields.leaf_ids), *(n+2*h for n in fields.mesh.block_shape), len(definitions))
    required = (fields.mesh.nbytes + fields.nbytes + 8*int(np.prod(shape)) +
                fields.slot_of_leaf.nbytes + rows.nbytes + weights.nbytes)
    admit(required, memory_limit, "derivative")
    output = np.empty(shape, dtype=float)
    backing = fields.values
    # This binding, rather than allocated shape, defines the first valid stencil.
    offset = fields.storage_halo - fields.valid_halo + 1

    def fill(first, last):
        differentiate(backing, fields.slot_of_leaf, fields.leaf_ids[first:last],
                      fields.mesh.spacing, rows, weights, output[first:last], offset)

    with worker_context(workers) as executor:
        run_ranges(len(output), workers, fill, executor)
    return publish(fields.mesh, output, fields.selection, definitions, h, h,
                   fields.scheme+"/centered-extended", fields.source)


def curl(fields, components=(0, 1, 2), *, workers=1, memory_limit=None):
    fields = require_fields(fields, halo=1)
    components = indices(components, len(fields.fields), "components")
    if len(components) != 3:
        raise ValueError("curl requires three ordered vector components")
    x, y, z = components
    units = {fields.fields[i].units for i in components}
    if len(units) != 1:
        raise ValueError("curl components must use common units")
    terms = (((z, 1, 1.), (y, 2, -1.)), ((x, 2, 1.), (z, 0, -1.)),
             ((y, 0, 1.), (x, 1, -1.)))
    definitions = tuple(FieldDefinition("curl_"+axis, next(iter(units))+" / coordinate-length",
                                        "centered-derivative") for axis in "xyz")
    result = derivative(fields, terms, definitions, workers=workers, memory_limit=memory_limit)
    # A small lifetime token binds this actual input group without retaining its
    # potentially large backing. It is not a payload hash or an expression cache.
    return replace(result, derivation=("curl", fields.value_identity, tuple(fields.fields),
                                      tuple(map(int, components)), fields.scheme))
