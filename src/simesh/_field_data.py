"""Bind aligned native field storage separately from pointwise formulas."""

import math
import numpy as np

from .fields import require_fields, _component_indices, _publication_bytes, publish
from ._validation import admit, array_bytes


def require_field_data(value):
    return require_fields(value)


def components(fields, selected=None, *, count=None):
    require_field_data(fields)
    result = _component_indices(fields, selected)
    if count is not None and len(result) != count:
        raise ValueError(f"select {count} components")
    units = {fields.fields[i].units for i in result}
    if len(units) != 1:
        raise ValueError("selected components must have common units")
    return result, next(iter(units))


class PointwiseLayout:
    """Bind matching sample locations; chunks never expose layout-specific IDs."""

    def __init__(self, inputs, *, allow_superset=False):
        self.inputs = tuple(require_field_data(value) for value in inputs)
        if not self.inputs:
            raise ValueError("provide at least one field group")
        self.unique_inputs = tuple({id(value): value for value in self.inputs}.values())
        self.first = first = self.inputs[0]
        for value in self.unique_inputs[1:]:
            if value.mesh is not first.mesh:
                raise ValueError("inputs must share the same Mesh")
            elif ((not allow_superset and len(value.leaf_ids) != len(first.leaf_ids)) or
                  np.any(value.slot_of_leaf[first.leaf_ids] < 0)):
                raise ValueError("inputs must share the same Mesh and leaf coverage")
            elif not allow_superset and not np.array_equal(value.selection.requested_bounds,
                                                           first.selection.requested_bounds):
                raise ValueError("inputs must share requested coverage; select the same region explicitly")
        self.halo = min(value.valid_halo for value in self.inputs)
        self.spatial_shape = tuple(n+2*self.halo for n in first.mesh.block_shape)
        self.prefix = (len(first.leaf_ids), *self.spatial_shape)
        self.chunk_cells = math.prod(self.spatial_shape)

    def admit(self, width, memory_limit, operation, *, scratch_per_cell=0):
        arrays = ((value.values, value.leaf_ids, value.slot_of_leaf)
                  for value in self.unique_inputs)
        inputs = array_bytes(array for group in arrays for array in group)
        metadata = _publication_bytes(self.first.mesh, len(self.first.leaf_ids))
        required = (self.first.mesh.nbytes+inputs+8*math.prod(self.prefix)*width+metadata+
                    self.chunk_cells*(scratch_per_cell))
        admit(required, memory_limit, operation)
        return required

    def allocate(self, width):
        return np.empty((*self.prefix, width))

    def chunks(self):
        boxes = tuple(tuple(slice(value.storage_halo-self.halo,
                                  value.storage_halo+n+self.halo)
                            for n in value.mesh.block_shape) for value in self.unique_inputs)
        for row, leaf in enumerate(self.first.leaf_ids):
            arrays = {id(value): value.values[(value.slot_of_leaf[leaf], *box, slice(None))]
                      for value, box in zip(self.unique_inputs, boxes)}
            yield row, tuple(arrays[id(value)] for value in self.inputs)

    def publish(self, output, definitions, *, scheme, source, stats=None):
        for value in self.unique_inputs:
            require_field_data(value)
        first = self.first
        return publish(first.mesh, output, first.selection, definitions, self.halo, self.halo,
                       scheme, source, stats)
