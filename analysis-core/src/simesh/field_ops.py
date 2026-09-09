"""Owned component selection and leaf-aligned composition of completed Fields."""

from collections.abc import Mapping
from dataclasses import replace
import math
import numpy as np

from ._validation import admit, array_bytes
from .fields import _field_index, publish, require_fields


def _aligned_inputs(inputs):
    fields = tuple(require_fields(value) for value in inputs)
    if not fields:
        raise ValueError("provide at least one field group")
    first = fields[0]
    for value in fields[1:]:
        if (value.mesh is not first.mesh or len(value.leaf_ids) != len(first.leaf_ids) or
                np.any(value.slot_of_leaf[first.leaf_ids] < 0)):
            raise ValueError("inputs must share the same Mesh and leaf coverage")
    return fields


def _layout(fields):
    halo = min(value.valid_halo for value in fields)
    shape = tuple(n + 2*halo for n in fields[0].mesh.block_shape)
    boxes = tuple(tuple(slice(value.storage_halo-halo, value.storage_halo+n+halo)
                        for n in value.mesh.block_shape) for value in fields)
    return halo, shape, boxes


def _admit_output(fields, shape, components, memory_limit, operation, *, scratch=0):
    first = fields[0]
    input_bytes = array_bytes(array for value in fields
                              for array in (value.values, value.leaf_ids, value.slot_of_leaf))
    required = (first.mesh.nbytes + input_bytes +
                8*len(first.leaf_ids)*math.prod(shape)*components +
                first.mesh.leaf_count*8 + scratch)
    admit(required, memory_limit, operation)


def _definitions(definitions, names):
    if names is not None:
        if isinstance(names, (str, Mapping)):
            raise TypeError("names must be an ordered sequence of output names")
        names = tuple(names)
        if len(names) != len(definitions):
            raise ValueError("one output name per selected component required")
        definitions = tuple(replace(value, name=name) for value, name in zip(definitions, names))
    if len({value.name for value in definitions}) != len(definitions):
        raise ValueError("duplicate output field names; supply unique names explicitly")
    return definitions


def select_fields(fields, components=None, *, names=None, memory_limit=None):
    """Copy an ordered component subset into compact, independently owned Fields.

    Select unique names, zero-based indices, or a mixture; a scalar selector is
    also accepted. None selects all components. Empty or repeated selections
    are rejected. Optional names replaces all selected names in output order;
    units and interpretations are preserved without conversion.

    Only valid halo is copied. Results have a fresh value identity and no curl
    derivation certificate, even for an identity selection; compute curl on the
    resulting vector group before passing a retained curl to tracing or QSL.
    """
    fields = require_fields(fields)
    if components is None:
        components = range(len(fields.fields))
    elif isinstance(components, (str, int, np.integer)):
        components = (components,)
    selected = []
    for component in components:
        if isinstance(component, str):
            component = _field_index(fields.fields, component)
        elif (isinstance(component, (bool, np.bool_)) or
              not isinstance(component, (int, np.integer))):
            raise ValueError("components must be field names or integer indices")
        if not 0 <= component < len(fields.fields):
            raise ValueError("component index outside the field group")
        selected.append(int(component))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select at least one component without repeats")
    definitions = _definitions(tuple(fields.fields[i] for i in selected), names)
    halo, shape, (box,) = _layout((fields,))
    _admit_output((fields,), shape, len(selected), memory_limit, "select fields")
    output = np.empty((len(fields.leaf_ids), *shape, len(selected)), dtype=np.float64)
    for slot, leaf in enumerate(fields.leaf_ids):
        for column, component in enumerate(selected):
            output[slot, ..., column] = fields.values[(fields.slot_of_leaf[leaf], *box, component)]
    return publish(fields.mesh, output, fields.selection, definitions, halo, halo,
                   fields.scheme, fields.source)


def merge_fields(inputs, *, names=None, memory_limit=None):
    """Copy an ordered sequence of groups into one independently owned Fields.

    Inputs must share Mesh identity and complete leaf coverage, but may differ
    in physical slot order, selection order, units and preparation scheme.
    Components follow group order and then each group's component order. Names
    must be unique; explicitly rename inputs or supply all output names to
    resolve collisions. Physical compatibility is the caller's responsibility.

    Output follows the first selection, packs its slots, and retains only the
    common valid halo. Source tokens are kept in input order, without retaining
    input Fields. A fresh identity invalidates any input curl certificate.
    """
    if isinstance(inputs, Mapping):
        raise TypeError("inputs must be an ordered sequence of Fields")
    fields = _aligned_inputs(inputs)
    first = fields[0]
    definitions = _definitions(tuple(definition for value in fields for definition in value.fields), names)
    halo, shape, boxes = _layout(fields)
    _admit_output(fields, shape, len(definitions), memory_limit, "merge fields")
    output = np.empty((len(first.leaf_ids), *shape, len(definitions)), dtype=np.float64)
    start = 0
    for value, box in zip(fields, boxes):
        stop = start + len(value.fields)
        for slot, leaf in enumerate(first.leaf_ids):
            output[slot, ..., start:stop] = value.values[(value.slot_of_leaf[leaf], *box, slice(None))]
        start = stop
    return publish(first.mesh, output, first.selection, definitions, halo, halo,
                   "merge(" + ",".join(value.scheme for value in fields) + ")",
                   tuple(value.source for value in fields))
