"""Published native field groups with separate storage and validity contracts."""

from dataclasses import dataclass, field
import numpy as np

from ._validation import frozen_array, indices, array_bytes
from .mesh import Mesh, Selection


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    units: str = "code"
    interpretation: str = "cell-average"

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.name, self.units, self.interpretation)):
            raise ValueError("field name, units and interpretation must be nonempty strings")


class _Lease:
    def __init__(self):
        self.active = True

    def check(self):
        if not self.active:
            raise RuntimeError("prepared batch borrow has expired")


@dataclass(frozen=True, eq=False)
class Fields:
    """Readonly fields on complete leaves, owned or explicitly batch-borrowed.

    ``storage_halo`` locates interiors in the backing. ``valid_halo`` states
    which surrounding values are complete, independently of allocated padding.
    Factories publish this descriptor only after all requested writes succeed.
    """

    mesh: Mesh
    _values: np.ndarray
    selection: Selection
    slot_of_leaf: np.ndarray
    fields: tuple[FieldDefinition, ...]
    storage_halo: int
    valid_halo: int
    scheme: str
    source: object
    preparation_stats: dict = field(default_factory=dict)
    _lease: _Lease | None = None
    value_identity: object = field(default_factory=object)
    derivation: tuple | None = None

    def __post_init__(self):
        if (type(self.storage_halo) is not int or type(self.valid_halo) is not int or
                not 0 <= self.valid_halo <= self.storage_halo):
            raise ValueError("valid_halo must fit the allocated storage_halo")
        expected = (*(n + 2*self.storage_halo for n in self.mesh.block_shape), len(self.fields))
        if (not isinstance(self._values, np.ndarray) or self._values.dtype != np.float64 or
                self._values.ndim != 5 or self._values.shape[1:] != expected or
                not self._values.flags.c_contiguous or not self.fields or
                not all(isinstance(f, FieldDefinition) for f in self.fields) or
                self.selection.mesh is not self.mesh):
            raise ValueError("field backing, definitions and mesh do not agree")
        slots = self.slot_of_leaf
        if (slots.shape != (self.mesh.leaf_count,) or slots.dtype != np.int64 or
                not slots.flags.c_contiguous or np.any(slots < -1) or
                np.any(slots >= len(self._values)) or np.any(slots[self.leaf_ids] < 0)):
            raise ValueError("invalid source-leaf to storage mapping")
        occupied = np.flatnonzero(slots >= 0)
        if (len(occupied) != len(self.leaf_ids) or
                len(np.unique(slots[occupied])) != len(occupied)):
            raise ValueError("field coverage and storage mapping disagree")
        self._values.flags.writeable = False
        slots.flags.writeable = False

    def _check(self):
        if self._lease is not None:
            self._lease.check()

    @property
    def values(self):
        self._check()
        return self._values

    @property
    def leaf_ids(self):
        return self.selection.leaf_ids

    @property
    def nbytes(self):
        self._check()
        return array_bytes((self._values, self.leaf_ids, self.slot_of_leaf))

    def interior(self):
        self._check()
        if (len(self._values) != len(self.leaf_ids) or
                not np.array_equal(self.slot_of_leaf[self.leaf_ids], np.arange(len(self.leaf_ids)))):
            raise ValueError("nonpacked slots require per-leaf window access")
        h = self.storage_halo
        return self._values[:, h:h+self.mesh.block_shape[0], h:h+self.mesh.block_shape[1],
                            h:h+self.mesh.block_shape[2], :]

    def window(self, leaf, lower, upper, *, support=0):
        self._check()
        leaf = int(indices([leaf], self.mesh.leaf_count)[0])
        lo, hi = np.asarray(lower), np.asarray(upper)
        slot = self.slot_of_leaf[leaf]
        if (lo.shape != (3,) or hi.shape != (3,) or lo.dtype.kind not in "iu" or
                hi.dtype.kind not in "iu" or np.any(lo < 0) or np.any(hi > self.mesh.block_shape) or
                np.any(lo > hi) or type(support) is not int or not 0 <= support <= self.valid_halo or slot < 0):
            raise ValueError("window exceeds supplied coverage or valid support")
        h = self.storage_halo
        slices = tuple(slice(int(a)+h-support, int(b)+h+support) for a, b in zip(lo, hi))
        return self._values[(slot, *slices, slice(None))]


def publish(mesh, values, selection, definitions, storage_halo, valid_halo, scheme, source,
            stats=None, lease=None, *, value_identity=None, derivation=None):
    directory = np.full(mesh.leaf_count, -1, dtype=np.int64)
    directory[selection.leaf_ids] = np.arange(len(selection.leaf_ids))
    return Fields(mesh, values, selection, directory, tuple(definitions), storage_halo,
                  valid_halo, scheme, source, stats or {}, lease,
                  object() if value_identity is None else value_identity, derivation)


def require_fields(value, *, halo=0, operation="consumer"):
    if not isinstance(value, Fields):
        raise TypeError("this consumer requires completed Fields")
    value._check()
    if value.valid_halo < halo:
        raise ValueError(f"{operation}: current valid_halo={value.valid_halo}; requires at least "
                         f"{halo} valid halo layers. Use prepare(source, fields=..., "
                         "scheme=...) to fill two valid layers; for derived fields, prepare "
                         "the inputs and recompute, allowing one layer per derivative. "
                         "Allocated or zero-filled storage_halo is not valid support.")
    return value


def _field_index(definitions, name):
    matches = [i for i, definition in enumerate(definitions) if definition.name == name]
    if len(matches) != 1:
        raise ValueError(f"field {name!r} must identify exactly one component (missing or ambiguous)")
    return matches[0]


def _component_indices(fields, components):
    if components is None:
        return tuple(range(len(fields.fields)))
    if isinstance(components, (str, int, np.integer)):
        components = (components,)
    selected = []
    for component in components:
        if isinstance(component, str):
            component = _field_index(fields.fields, component)
        if (isinstance(component, (bool, np.bool_)) or
                not isinstance(component, (int, np.integer)) or
                not 0 <= component < len(fields.fields)):
            raise ValueError("components must be field names or available integer indices")
        selected.append(int(component))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select at least one component without duplicates")
    return tuple(selected)



def source_value_identity(source_identity, field_ids, scheme):
    """Logical immutable primary values, independent of packing or retention."""
    return source_identity, tuple(map(int,field_ids)), scheme


def require_continuous(fields, components=None, *, halo=1, operation="interpolation"):
    require_fields(fields, halo=halo, operation=operation)
    selected = _component_indices(fields, components)
    for i in selected:
        if fields.fields[i].interpretation.startswith("categorical"):
            raise ValueError(f"{operation}: {fields.fields[i].name!r} is categorical; "
                             "select continuous physical components instead")
    return selected


def _input_arrays(fields):
    """Arrays that caller outputs must not overwrite during native consumption."""
    return (fields.values, fields.leaf_ids, fields.slot_of_leaf,
            *(a for a in vars(fields.mesh).values() if isinstance(a,np.ndarray)),
            *(a for a in fields.mesh.forest if isinstance(a,np.ndarray)))
