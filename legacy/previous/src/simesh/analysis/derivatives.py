"""Centered stencils on extended inputs, with independent derived backing."""

import numpy as np

from .fields import PreparedFields, FieldDefinition


def derivative(prepared, terms, definitions, *, budget_bytes=2*1024**3):
    """Each output sums ``(field_slot, axis, coefficient)`` centered derivatives.

    Differentiate the prepared extended input, then publish one fewer halo.
    This differs from exchanging derived interiors across refinement interfaces.
    No magnetic/current normalization is inferred. Coefficients carry any
    explicitly selected conversion; definitions specify output units.
    """
    from simesh.utils.lib.analysis.native import differentiate
    if prepared.halo < 1:
        raise ValueError("derivatives require one valid input layer")
    definitions = tuple(definitions)
    if len(terms) != len(definitions) or not definitions:
        raise ValueError("one nonempty term list per defined output is required")
    records, weights = [], []
    for output, entries in enumerate(terms):
        if not entries:
            raise ValueError("a derivative output requires a term")
        for component, axis, coefficient in entries:
            if (not isinstance(component, (int, np.integer)) or
                    not 0 <= component < len(prepared.fields) or
                    not isinstance(axis, (int, np.integer)) or not 0 <= axis < 3 or
                    not np.isfinite(coefficient)):
                raise ValueError("invalid derivative component, axis or coefficient")
            records.append((output, component, axis))
            weights.append(coefficient)
    rows = np.asarray(records, dtype=np.int64)
    coefficients = np.asarray(weights, dtype=np.float64)
    halo = prepared.halo-1
    shape = (len(prepared.leaf_ids), *(n+2*halo for n in prepared.mesh.block_shape),
             len(definitions))
    required = (prepared.mesh.nbytes + prepared.nbytes + 8*int(np.prod(shape)) +
                prepared.slot_of_leaf.nbytes + rows.nbytes + coefficients.nbytes)
    if required > budget_bytes:
        raise MemoryError(f"derivative needs {required} controlled bytes, budget {budget_bytes}")
    output = np.empty(shape, dtype=np.float64)
    differentiate(prepared.values, prepared.slot_of_leaf, prepared.leaf_ids,
                  prepared.mesh.spacing, rows, coefficients, output)
    directory = np.full(prepared.mesh.leaf_count, -1, dtype=np.int64)
    directory[prepared.leaf_ids] = np.arange(len(prepared.leaf_ids))
    directory.flags.writeable = output.flags.writeable = False
    return PreparedFields(prepared.mesh, output, prepared.leaf_ids, directory,
                          definitions, halo, prepared.strategy+"/centered-extended",
                          prepared.source)


def curl(prepared, components=(0, 1, 2), *, budget_bytes=2*1024**3):
    x,y,z = components
    terms = (((z,1,1.),(y,2,-1.)), ((x,2,1.),(z,0,-1.)), ((y,0,1.),(x,1,-1.)))
    units = {prepared.fields[i].units for i in components}
    if len(units) != 1:
        raise ValueError("curl components must use the same units")
    unit = next(iter(units))+" / coordinate-length"
    return derivative(prepared, terms,
                      tuple(FieldDefinition("curl_"+a, unit, "centered-derivative")
                            for a in "xyz"), budget_bytes=budget_bytes)
