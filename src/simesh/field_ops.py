"""Owned component selection and composition on matching sample locations."""

from collections.abc import Mapping
from dataclasses import replace
import math
import numpy as np

from ._validation import admit
from .fields import _component_indices
from ._field_data import PointwiseLayout, require_field_data
from .mesh import resolve_selection


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


def select_fields(fields, components=None, *, names=None, region=None, memory_limit=None):
    """Copy an ordered component subset into compact, independently owned Fields.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    names : sequence of str, optional
        Distinct output names in component order.
    region : Selection or array-like, optional
        Explicit target coverage on the original grid. AMR retains complete
        intersecting leaves on the original Mesh.
        No missing values are filled and region edges are not physical boundaries.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent copy, even for a full selection or rename. For one-time
        sampling, consumer components avoid this field copy.
    """
    fields = require_field_data(fields)
    selected = _component_indices(fields, components)
    definitions = _definitions(tuple(fields.fields[i] for i in selected), names)
    if region is not None:
        fields = _select_region(fields, region, len(selected), memory_limit)
    layout = PointwiseLayout((fields,))
    layout.admit(len(selected), memory_limit, "select fields")
    output = layout.allocate(len(selected))
    for index, (data,) in layout.chunks():
        target = (index, Ellipsis)
        for column, component in enumerate(selected):
            output[(*target, column)] = data[..., component]
    return layout.publish(output, definitions, scheme=fields.scheme, source=fields.source,
                          stats=fields.preparation_stats.copy())


def _select_region(fields, region, width, memory_limit):
    selection = resolve_selection(fields.mesh, region)
    if np.any(fields.slot_of_leaf[selection.leaf_ids] < 0):
        raise ValueError("requested region is not covered by the supplied fields")
    admit(fields.mesh.nbytes+fields.nbytes+fields.mesh.leaf_count*8+
          len(selection.leaf_ids)*math.prod(n+2*fields.valid_halo for n in fields.mesh.block_shape)*width*8,
          memory_limit, "field region selection")
    slots = np.full(fields.mesh.leaf_count, -1, dtype=np.int64)
    slots[selection.leaf_ids] = fields.slot_of_leaf[selection.leaf_ids]
    return replace(fields, selection=selection, slot_of_leaf=slots)


def merge_fields(inputs, *, names=None, memory_limit=None):
    """Copy an ordered sequence of groups into one independently owned Fields.

    Parameters
    ----------
    inputs : sequence of Fields
        Groups sharing grid identity and requested coverage; explicitly select
        the same region before merging. Storage slot order may differ.
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
    layout = PointwiseLayout(inputs)
    fields = layout.inputs
    definitions = _definitions(tuple(definition for value in fields for definition in value.fields), names)
    layout.admit(len(definitions), memory_limit, "merge fields")
    output = layout.allocate(len(definitions))
    for index, arrays in layout.chunks():
        target = (index, Ellipsis)
        column = 0
        for data in arrays:
            width = data.shape[-1]
            output[(*target, slice(column,column+width))] = data
            column += width
    return layout.publish(output, definitions,
                          scheme="merge(" + ",".join(value.scheme for value in fields) + ")",
                          source=tuple(value.source for value in fields))
