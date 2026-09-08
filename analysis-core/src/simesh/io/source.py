"""Immutable value sources: reading is independent of halo preparation."""

from collections.abc import Mapping
import numpy as np

from .._validation import indices, array_bytes, admit
from ..fields import FieldDefinition, publish, source_value_identity, _field_index
from ..mesh import resolve_selection
from .._amr.blockio import array_block_reader, read_blocks_into


class Source:
    """An owned source lifetime, a mesh and a stable directory of stored fields.

    Advanced adapters supply a checked block reader. No numerical preparation
    method, support workspace or cache is attached to the source.
    """

    def __init__(self, mesh, definitions, reader, *, validate=None, close=None,
                 read_full=None, read_scratch_bytes=0, metadata_arrays=(), read_scratch=None, io_stats=None):
        self.mesh = mesh
        self.fields = tuple(definitions)
        if not self.fields or not all(isinstance(f, FieldDefinition) for f in self.fields):
            raise ValueError("nonempty FieldDefinition sequence required")
        if reader.shape != (mesh.leaf_count, len(self.fields), *mesh.block_shape):
            raise ValueError("reader shape disagrees with source mesh and fields")
        self._reader = reader
        self._validate = validate
        self._close = close
        self._read_full = read_full
        self.read_scratch_bytes = int(read_scratch_bytes)
        self._memory_arrays = (*reader.memory_arrays, *metadata_arrays)
        self.identity = object()
        self._closed = False
        self._read_scratch = read_scratch
        self.io_stats = {} if io_stats is None else io_stats

    @property
    def nbytes(self):
        return array_bytes(self._memory_arrays)

    def validate(self):
        if self._closed:
            raise OSError("source is closed")
        if self._validate is not None:
            self._validate()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                if self._close is not None:
                    self._close()
            finally:
                self._reader = self._read_full = self._read_scratch = None
                self._validate = self._close = None
                self._memory_arrays = ()

    def read_footprint(self, count, field_count):
        """I/O-only temporary storage for a declared batch, excluding its output."""
        return (self.read_scratch_bytes if self._read_scratch is None else
                self._read_scratch(count,field_count))

    def __enter__(self):
        try:
            self.validate()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def field_ids(self, fields=None):
        if fields is None:
            return np.arange(len(self.fields), dtype=np.int64)
        selected = (fields,) if isinstance(fields, str) else tuple(fields)
        if not selected:
            raise ValueError("select at least one field")
        if all(isinstance(x, str) for x in selected):
            ids = [_field_index(self.fields, name) for name in selected]
            return indices(ids, len(self.fields), "fields")
        return indices(selected, len(self.fields), "fields")

    def read_into(self, leaf_ids, field_ids, output):
        """Read ordered native interiors into field-major caller storage."""
        self.validate()
        ids = indices(leaf_ids, self.mesh.leaf_count)
        fields = indices(field_ids, len(self.fields), "field_ids")
        if (not len(fields) or not isinstance(output, np.ndarray) or output.dtype != np.float64 or
                output.shape != (len(ids), len(fields), *self.mesh.block_shape) or
                not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("output must be writable native field-major interiors")
        zero = np.zeros(3, dtype=np.int64)
        upper = np.asarray(self.mesh.block_shape, dtype=np.int64)
        read_blocks_into(self._reader, zero, upper, ids, fields, output, zero)
        self.validate()

    def _read_padded_into(self, leaf_ids, field_ids, output, offset):
        """Internal read boundary for a private field-major preparation buffer."""
        self.validate()
        zero = np.zeros(3, dtype=np.int64)
        read_blocks_into(self._reader, zero, np.asarray(self.mesh.block_shape, dtype=np.int64),
                         leaf_ids, field_ids, output, offset)
        self.validate()


def definitions_from_names(names, units=None):
    if units is not None and not isinstance(units, (str, Mapping)):
        raise ValueError("units must be a string or a field-name mapping")
    return tuple(FieldDefinition(name, units if isinstance(units, str) else (units or {}).get(name, "code"))
                 for name in names)


def source_from_arrays(mesh, values, fields, *, units=None, copy=True, memory_limit=None):
    """Create a source from native float64 (leaf, component, x, y, z) data.

    With ``copy=False``, callers must keep the backing unchanged until close.
    Detached prepared products do not borrow that input backing.
    """
    definitions = tuple(fields)
    if all(isinstance(f, str) for f in definitions):
        definitions = definitions_from_names(definitions, units)
    elif units is not None:
        raise ValueError("FieldDefinition already specifies units")
    if (not isinstance(values, np.ndarray) or values.dtype != np.float64 or
            values.shape != (mesh.leaf_count, len(definitions), *mesh.block_shape) or
            not values.flags.c_contiguous or type(copy) is not bool):
        raise ValueError("array source requires native contiguous field-major float64 values")
    admit(mesh.nbytes + values.nbytes*(2 if copy else 1), memory_limit, "array source")
    backing = values.copy() if copy else values
    if copy:
        backing.flags.writeable = False
    return Source(mesh, definitions, array_block_reader(backing))


def read_fields(source, fields=None, *, region=None, leaf_ids=None, memory_limit=None):
    """Publish detached interior fields without constructing halo workspaces."""
    selection = resolve_selection(source.mesh, region, leaf_ids)
    field_ids = source.field_ids(fields)
    count = len(selection.leaf_ids)
    cells = int(np.prod(source.mesh.block_shape))
    size = count*len(field_ids)*cells*8
    # Read and component-adjacent publication can coexist.
    admit(source.mesh.nbytes + source.nbytes + 2*size + source.read_footprint(count,len(field_ids)) +
          8*(source.mesh.leaf_count+count), memory_limit, "read fields")
    source.validate()
    complete = np.array_equal(selection.leaf_ids, np.arange(source.mesh.leaf_count))
    if complete and source._read_full is not None:
        raw, stats = source._read_full(field_ids)
    else:
        raw = np.empty((count, len(field_ids), *source.mesh.block_shape), dtype=float)
        source.read_into(selection.leaf_ids, field_ids, raw)
        stats = {"selected_load_count": count, "read_value_bytes": size}
    source.validate()
    values = np.ascontiguousarray(np.moveaxis(raw, 1, -1))
    return publish(source.mesh, values, selection, tuple(source.fields[i] for i in field_ids),
                   0, 0, "interior", source.identity, stats,
                   value_identity=source_value_identity(source.identity,field_ids,"interior"))
