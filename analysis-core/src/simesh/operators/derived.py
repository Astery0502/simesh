"""Explicit pointwise recipes producing independently owned native fields."""

from collections.abc import Mapping
import numpy as np

from .._validation import admit, array_bytes
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
    includes inputs, output and one float64 result block, but cannot bound
    arbitrary allocations inside user callbacks. Nonfinite values propagate.
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
    definition = FieldDefinition(name, units, "pointwise-derived")
    fields = [require_fields(value) for value in groups.values()]
    first = fields[0]
    for value in fields[1:]:
        if (value.mesh is not first.mesh or len(value.leaf_ids) != len(first.leaf_ids) or
                np.any(value.slot_of_leaf[first.leaf_ids] < 0)):
            raise ValueError("derived inputs must share the same Mesh and leaf coverage")
    halo = min(value.valid_halo for value in fields)
    block_shape = tuple(n + 2*halo for n in first.mesh.block_shape)
    block_bytes = 8*int(np.prod(block_shape))
    input_bytes = array_bytes(array for value in fields
                              for array in (value.values, value.leaf_ids, value.slot_of_leaf))
    required = (first.mesh.nbytes + input_bytes +
                (len(first.leaf_ids)+1)*block_bytes + first.mesh.leaf_count*8)
    admit(required, memory_limit, "derive")
    # Geometry is invariant across callbacks; field names are resolved only on
    # first use, while values access still checks any borrowed input lifetime.
    bindings = {
        key: (value, tuple(slice(value.storage_halo-halo, value.storage_halo+n+halo)
                           for n in value.mesh.block_shape), {})
        for key, value in groups.items()
    }
    output = np.empty((len(first.leaf_ids), *block_shape, 1), dtype=np.float64)
    for slot, leaf in enumerate(first.leaf_ids):
        result = np.asarray(func(DerivedContext(bindings, leaf)), dtype=np.float64)
        if result.shape not in ((), block_shape):
            raise ValueError(f"recipe must return a scalar or block shape {block_shape}, got {result.shape}")
        output[slot, ..., 0] = result
        del result
    return publish(first.mesh, output, first.selection, (definition,), halo, halo,
                   "pointwise(" + ",".join(value.scheme for value in fields) + ")",
                   tuple(value.source for value in fields))
