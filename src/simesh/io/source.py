"""Immutable value sources: reading is independent of halo preparation."""

from collections.abc import Mapping
import numpy as np

from .._validation import indices, array_bytes, admit
from ..fields import FieldDefinition, publish, source_value_identity, _field_index
from ..mesh import resolve_selection
from .._amr.blockio import array_block_reader, read_blocks_into, make_block_reader
from .metadata import SnapshotMetadata
from ._boundary import boundary_configuration


class Source:
    """Immutable input lifetime and stable stored-field directory.

    Use open_amrvac or source_from_arrays.

    Reading is separate from numerical preparation; caching is opt-in.

    Attributes
    ----------
    mesh : Mesh
        Shared original geometry.
    fields : tuple of FieldDefinition
        Available stored components; integer selectors are local to this directory.
    boundary : tuple of tuple of str
        Immutable rules in field-directory order, with faces x-, x+, y-, y+, z-, z+.
        continuous copies the nearest interior cell; symmetric mirrors cell centers;
        asymmetric mirrors and negates (odd parity). Vector parity is explicit per
        component, never inferred from names. periodic faces follow Mesh.periodic.
    identity : object
        In-memory source association, not a file hash.
    io_stats : dict
        Adapter-specific read/cache observations.

    Notes
    -----
    - Constructor boundary inputs may be a mode string for all nonperiodic faces,
      a field-name mapping to mode strings or six-face rows, or a (field, 6) table.
      Omitted fields default to continuous; explicit rows must use periodic exactly
      on periodic faces. No physical rules are inferred from snapshot headers.
    - Use with/close; detached metadata and owned Fields survive closure.
    - Borrowed adapters must close before their parent. Parent closure or detected
      input changes invalidate reads, including cache hits.
    """

    def __init__(self, mesh, definitions, reader, *, validate=None, close=None,
                 read_full=None, read_native=None, read_scratch_bytes=0, metadata_arrays=(), read_scratch=None, io_stats=None,
                 metadata=None, boundary=None):
        if metadata is not None and not isinstance(metadata, SnapshotMetadata):
            raise TypeError("metadata must be SnapshotMetadata or None")
        self._metadata = metadata
        self.mesh = mesh
        self.fields = tuple(definitions)
        if not self.fields or not all(isinstance(f, FieldDefinition) for f in self.fields):
            raise ValueError("nonempty FieldDefinition sequence required")
        if reader.shape != (mesh.leaf_count, len(self.fields), *mesh.block_shape):
            raise ValueError("reader shape disagrees with source mesh and fields")
        self._boundary, self._boundary_modes = boundary_configuration(mesh, self.fields, boundary)
        self._reader = reader
        self._validate = validate
        self._close = close
        # Accept read_full for older adapters; final-storage transfers use read_native.
        self._read_native = read_native
        self.field_origins = tuple(range(len(self.fields)))
        self.read_scratch_bytes = int(read_scratch_bytes)
        self._memory_arrays = tuple(a for a in (*reader.memory_arrays, *metadata_arrays, self._boundary_modes)
                                    if isinstance(a, np.ndarray))
        self.identity = object()
        self._closed = False
        self._read_scratch = read_scratch
        self.io_stats = {} if io_stats is None else io_stats

    @property
    def boundary(self):
        """Detached immutable physical halo configuration; see Source."""
        return self._boundary

    @property
    def metadata(self):
        """Detached snapshot description, available after close; None if unspecified."""
        return self._metadata

    @property
    def nbytes(self):
        """Accounted input backing and metadata arrays, excluding the shared Mesh."""
        return array_bytes((*self._memory_arrays, self._boundary_modes))

    def validate(self):
        """Reject a closed Source or a change detected by its input adapter.
        """
        if self._closed:
            raise OSError("source is closed")
        if self._validate is not None:
            self._validate()

    def close(self):
        """Release input resources; detached metadata and completed owned fields survive.
        """
        if not self._closed:
            self._closed = True
            try:
                if self._close is not None:
                    self._close()
            finally:
                self._reader = self._read_native = self._read_scratch = None
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
        """Resolve a name or sequence of names/local integers to stored component IDs.
        """
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
        """Read ordered interiors into writable (leaf, component, x, y, z) float64 storage.

        Failure may leave partial writes; this method does not prepare halos.
        """
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

    def read_native_into(self, leaf_ids, field_ids, output, *, storage_halo=0):
        """Read interiors into component-last float64 caller storage.

        Allocated padding remains invalid. Output must not alias input; failure may
        leave partial writes.
        """
        self.validate()
        ids = indices(leaf_ids, self.mesh.leaf_count)
        fields = indices(field_ids, len(self.fields), "field_ids")
        h = storage_halo
        if (type(h) is not int or h < 0 or not len(fields) or
                not isinstance(output, np.ndarray) or output.dtype != np.float64 or
                output.shape != (len(ids), *(n+2*h for n in self.mesh.block_shape), len(fields)) or
                not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("output must match writable component-last native storage")
        if any(np.shares_memory(output, a) for a in (*self._memory_arrays, ids, fields)):
            raise ValueError("output must not alias source or request storage")
        box = tuple(slice(h, h+n) for n in self.mesh.block_shape)
        target = output[(slice(None), *box, slice(None))]
        if self._read_native is not None:
            stats = self._read_native(ids, fields, target)
        else:
            raw = np.empty((1, len(fields), *self.mesh.block_shape))
            for row, leaf in enumerate(ids):
                self.read_into(np.asarray([leaf], dtype=np.int64), fields, raw)
                target[row] = np.moveaxis(raw[0], 0, -1)
            stats = {"selected_load_count": len(ids), "peak_read_buffer_bytes": raw.nbytes}
        self.validate()
        return stats

    def value_identity(self, field_ids, scheme):
        return source_value_identity(self.identity,
            tuple(self.field_origins[i] for i in field_ids), scheme)

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


def source_from_arrays(mesh, values, fields, *, units=None, copy=True, memory_limit=None, metadata=None, boundary=None):
    """Create a source from native float64 (leaf, component, x, y, z) data.

    Parameters
    ----------
    mesh : Mesh
        Validated Cartesian 3D mesh; its periodic flags control halo topology only.
    values : ndarray
        C-contiguous float64 array in (leaf, component, x, y, z) order.
    fields : sequence
        Field names or FieldDefinition objects.
    units : str or mapping, optional
        Labels only; no numerical conversion. Omit with explicit FieldDefinition objects.
    copy : bool
        Own a copy, or borrow backing that the caller keeps unchanged until close.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.
    metadata : SnapshotMetadata, optional
        Detached description; does not verify the supplied array values.
    boundary : str, mapping or array-like, optional
        Explicit physical halo rules in this field directory; see Source.boundary
        and the Source constructor input contract. Defaults to continuous.

    Returns
    -------
    Source
        Array-backed immutable input with the supplied Mesh and optional detached metadata.
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
    admit(mesh.nbytes + values.nbytes*(2 if copy else 1) + 6*len(definitions),
          memory_limit, "array source")
    backing = values.copy() if copy else values
    if copy:
        backing.flags.writeable = False
    def read_native(ids, selected, target):
        for row, leaf in enumerate(ids):
            for column, component in enumerate(selected):
                target[row, ..., column] = backing[leaf, component]
        return {"selected_load_count": len(ids), "peak_read_buffer_bytes": 0}
    return Source(mesh, definitions, array_block_reader(backing), read_native=read_native, metadata=metadata, boundary=boundary)


def read_fields(source, fields=None, *, region=None, leaf_ids=None, memory_limit=None):
    """Publish detached interior fields without constructing halo workspaces.

    Parameters
    ----------
    source : Source
        Open immutable input; field selectors address its current directory.
    fields : str or sequence, optional
        Names or local Source indices of stored fields; integers must be supplied as a
        sequence.
    region : Selection or array-like, optional
        Complete leaves intersecting a physical (2, 3) box, or an existing selection.
    leaf_ids : sequence of int, optional
        Ordered original leaf IDs; mutually exclusive with region.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Detached component-last interiors with zero valid halo. Use prepare before
        linear interpolation or derivatives; zero-order sampling needs no halo.
    """
    selection = resolve_selection(source.mesh, region, leaf_ids)
    field_ids = source.field_ids(fields)
    count = len(selection.leaf_ids)
    cells = int(np.prod(source.mesh.block_shape))
    size = count*len(field_ids)*cells*8
    required = (source.mesh.nbytes + source.nbytes + size +
                source.read_footprint(count,len(field_ids)) + cells*len(field_ids)*8 +
                8*(source.mesh.leaf_count+count))
    admit(required, memory_limit, "read fields")
    values = np.empty((count, *source.mesh.block_shape, len(field_ids)))
    stats = source.read_native_into(selection.leaf_ids, field_ids, values)
    stats.update(controlled_upper_bytes=required, publication_copy_bytes=0)
    return publish(source.mesh, values, selection, tuple(source.fields[i] for i in field_ids),
                   0, 0, "interior", source.identity, stats,
                   value_identity=source.value_identity(field_ids,"interior"))


def select_source(source, fields):
    """Borrow an ordered Source subset without copying values.

    Parameters
    ----------
    source : Source
        Open immutable input; field selectors address its current directory.
    fields : str or sequence, optional
        Names or local Source indices of stored fields; integers must be supplied as a
        sequence.

    Returns
    -------
    Source
        Borrowed ordered directory adapter without a value copy; Source lifetime rules
        apply. Local indices retain their original value association.
    """
    source.validate()
    chosen = source.field_ids(fields)
    chosen.flags.writeable = False
    def read(state, lower, upper, ids, columns, output, offset):
        source.validate()
        read_blocks_into(source._reader, lower, upper, ids, chosen[columns], output, offset)
        source.validate()
    def native(ids, columns, target):
        source.validate()
        return source._read_native(ids, chosen[columns], target)
    reader = make_block_reader(source, (source.mesh.leaf_count,len(chosen),*source.mesh.block_shape),
        read, memory_arrays=(*source._memory_arrays,chosen))
    result = Source(source.mesh, tuple(source.fields[i] for i in chosen), reader,
        validate=source.validate, read_native=native if source._read_native is not None else None,
        read_scratch=source.read_footprint, io_stats=source.io_stats, metadata=source.metadata,
        boundary=tuple(source.boundary[i] for i in chosen))
    result.identity = source.identity
    result.field_origins = tuple(source.field_origins[i] for i in chosen)
    return result
