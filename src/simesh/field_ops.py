"""Owned component selection and leaf-aligned composition of completed Fields."""

from collections.abc import Mapping
from dataclasses import replace
import math
import numpy as np

from ._validation import admit, array_bytes
from .fields import _component_indices, publish, require_fields


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
    # Publication validates the directory with masks and unique slot indices,
    # after recipe temporaries have been released.
    metadata_scratch = max(first.mesh.leaf_count + 8*len(first.leaf_ids),
                           33*len(first.leaf_ids))
    required = (first.mesh.nbytes + input_bytes +
                8*len(first.leaf_ids)*math.prod(shape)*components +
                first.mesh.leaf_count*8 + max(scratch, metadata_scratch))
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

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    names : sequence of str, optional
        Distinct output names in component order.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent compact copy, even for a full selection or rename. For one-time
        sampling, consumer components avoid this field copy.
    """
    fields = require_fields(fields)
    selected = _component_indices(fields, components)
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

    Parameters
    ----------
    inputs : sequence of Fields
        Groups sharing Mesh identity and leaf coverage; slot order may differ.
        Components follow group order, then each group's component order.
    names : sequence of str, optional
        Distinct output names in concatenation order; supply them to resolve name
        collisions.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent concatenation packed in the first group's leaf order, with common
        valid halo. Inputs are not retained; the new value identity does not preserve
        input curl certificates.

    Notes
    -----
    Physical compatibility of units and preparation schemes remains the caller's responsibility.
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
