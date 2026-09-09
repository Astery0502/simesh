"""Centered derivatives of extended inputs, retaining independent outputs."""

import numpy as np
from dataclasses import replace

from .._validation import admit, workers_count, indices
from .._execution import worker_context, run_ranges
from ..fields import FieldDefinition, publish, require_fields, _field_index, require_continuous, _input_arrays


def derivative(fields, terms, definitions, *, output=None, workers=1, memory_limit=None):
    """Sum ordered (component, axis, coefficient) terms for each output field.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    terms : sequence
        One ordered list of (component, axis, coefficient) terms per output. Components
        accept names/indices; axes accept x/y/z or 0/1/2.
    definitions : sequence of FieldDefinition
        One definition per output. Physical current and unit conversion are not
        inferred.
    output : ndarray, optional
        Writable contiguous float64 result backing, without input aliases. Failure may
        leave partial writes; keep writable aliases unchanged while consuming returned
        Fields.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent or caller-backed centered derivatives with input.valid_halo - 1
        valid layers.

    Notes
    -----
    To interpolate the derivative afterward, prepare the original input with two valid halo layers.
    """
    from .._kernels.native import differentiate
    fields = require_fields(fields, halo=1)
    workers_count(workers)
    definitions = tuple(definitions)
    if (not definitions or len(terms) != len(definitions) or
            not all(isinstance(definition, FieldDefinition) for definition in definitions)):
        raise ValueError("one nonempty term list per defined output required")
    rows, weights = [], []
    for output_index, entries in enumerate(terms):
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
            rows.append((output_index, component, axis))
            weights.append(coefficient)
    require_continuous(fields, tuple(dict.fromkeys(r[1] for r in rows)), operation="derivative")
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    h = fields.valid_halo - 1
    shape = (len(fields.leaf_ids), *(n+2*h for n in fields.mesh.block_shape), len(definitions))
    required = (fields.mesh.nbytes + fields.nbytes + 8*int(np.prod(shape)) +
                fields.slot_of_leaf.nbytes + rows.nbytes + weights.nbytes)
    admit(required, memory_limit, "derivative")
    owned = output is None
    if owned:
        output = np.empty(shape, dtype=float)
    elif (not isinstance(output,np.ndarray) or output.shape!=shape or output.dtype!=np.float64 or
            not output.flags.c_contiguous or not output.flags.writeable or
            any(np.shares_memory(output,a) for a in _input_arrays(fields))):
        raise ValueError("output must match writable contiguous derivative storage and not alias input")
    backing = fields.values
    # This binding, rather than allocated shape, defines the first valid stencil.
    offset = fields.storage_halo - fields.valid_halo + 1

    def fill(first, last):
        differentiate(backing, fields.slot_of_leaf, fields.leaf_ids[first:last],
                      fields.mesh.spacing, rows, weights, output[first:last], offset)

    with worker_context(workers) as executor:
        run_ranges(len(output), workers, fill, executor)
    return publish(fields.mesh, output if owned else output.view(), fields.selection, definitions, h, h,
                   fields.scheme+"/centered-extended", fields.source)


def _curl_terms(components):
    x,y,z = components
    return (((z, 1, 1.), (y, 2, -1.)), ((x, 2, 1.), (z, 0, -1.)),
            ((y, 0, 1.), (x, 1, -1.)))


def curl(fields, components=(0, 1, 2), *, output=None, workers=1, memory_limit=None):
    """Compute the centered curl of three ordered vector components.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    components : sequence of int
        Exactly three ordered local component indices with common unit labels.
    output : ndarray, optional
        Writable contiguous float64 result backing, without input aliases. Failure may
        leave partial writes; keep writable aliases unchanged while consuming returned
        Fields.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Three curl components in field units per coordinate length, with one fewer valid
        halo. Derivation identity allows use as a matching twist companion.

    Notes
    -----
    To interpolate the derivative afterward, prepare the original input with two valid halo layers.
    """
    fields = require_fields(fields, halo=1)
    components = indices(components, len(fields.fields), "components")
    if len(components) != 3:
        raise ValueError("curl requires three ordered vector components")
    units = {fields.fields[i].units for i in components}
    if len(units) != 1:
        raise ValueError("curl components must use common units")
    terms = _curl_terms(components)
    definitions = tuple(FieldDefinition("curl_"+axis, next(iter(units))+" / coordinate-length",
                                        "centered-derivative") for axis in "xyz")
    result = derivative(fields, terms, definitions, output=output, workers=workers, memory_limit=memory_limit)
    # A small lifetime token binds this actual input group without retaining its
    # potentially large backing. It is not a payload hash or an expression cache.
    return replace(result, derivation=("curl", fields.value_identity, tuple(fields.fields),
                                      tuple(map(int, components)), fields.scheme))
