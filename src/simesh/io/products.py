"""Explicit ordinary AMRVAC file products from completed fields or file interiors."""

from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .metadata import SnapshotMetadata
from ..fields import require_fields
from ..mesh import _root_bounds, _root_spans, _root_physical_bounds
from .._validation import admit


_BATCH_BYTES = 8 * 1024**2
_BATCH_LEAVES = 256


def _export_plan(mesh, metadata, names, root_bounds, retained_bytes, memory_limit):
    if any(mesh.periodic):
        raise ValueError("AMRVAC export does not support periodic meshes")
    if not isinstance(metadata, (SnapshotMetadata, dict)):
        raise ValueError('explicit SnapshotMetadata or header dictionary required')
    header = metadata.to_header() if isinstance(metadata, SnapshotMetadata) else deepcopy(metadata)
    if (header.get('datfile_version') != 5 or header.get('ndim') != 3 or
            header.get('geometry') != 'Cartesian_3D' or
            np.any(header.get('periodic', True))):
        raise ValueError('metadata must describe nonperiodic Cartesian 3D v5 data')
    expected = {'xmin': mesh.lower, 'xmax': mesh.upper,
                'domain_nx': mesh.root_shape*np.asarray(mesh.block_shape),
                'block_nx': mesh.block_shape}
    if any(not np.array_equal(header.get(key), value) for key, value in expected.items()):
        raise ValueError('metadata geometry must match the Fields mesh')
    try:
        encoded = [name.encode('ascii') for name in names]
    except UnicodeEncodeError:
        raise ValueError('AMRVAC field names must be ASCII') from None
    if (not names or len(set(names)) != len(names) or
            any(not name or len(name) > 16 or name.strip() != name or b'\x00' in name for name in encoded)):
        raise ValueError('AMRVAC field names must be distinct, unpadded, NUL-free and contain 1 to 16 ASCII bytes')

    box = _root_bounds(mesh, ((0, 0, 0), mesh.root_shape) if root_bounds is None else root_bounds)
    roots = (1 if np.all(box[0] == 0) and np.array_equal(box[1], mesh.root_shape)
             else math.prod(map(int, box[1] - box[0])))
    # Includes clipped Morton maps and vectorized root-span temporaries.
    admit(retained_bytes + 256*roots, memory_limit, 'AMRVAC root selection')
    spans = _root_spans(mesh, box)
    count = int(np.sum(spans[:, 3] - spans[:, 2]))
    nodes = int(np.sum(spans[:, 1] - spans[:, 0]))
    # Conservative bound for live plan, serialized index arrays and construction
    # temporaries. Payload batches and reader scratch are admitted separately.
    required = retained_bytes + 256*roots + 160*count + 16*nodes
    admit(required, memory_limit, 'AMRVAC export index')
    ids = np.empty(count, dtype=np.int64)
    flags = np.empty(nodes, dtype=np.bool_)
    leaf_offset = node_offset = 0
    for start, stop, first, last in spans:
        leaves, size = last - first, stop - start
        ids[leaf_offset:leaf_offset+leaves] = np.arange(first, last)
        flags[node_offset:node_offset+size] = mesh.node_leaves[start:stop] >= 0
        leaf_offset += leaves
        node_offset += size
    leaf_nodes = mesh.leaf_nodes[ids]
    levels = mesh.forest.node_levels[leaf_nodes]
    coordinates = mesh.forest.node_coords[leaf_nodes]
    coordinates -= box[0] * np.left_shift(1, levels - 1)[:, None]
    coordinates += 1
    bounds = _root_physical_bounds(mesh, box)
    if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('cropped physical bounds are not representable')
    header.update(xmin=bounds[0], xmax=bounds[1],
                  domain_nx=(box[1]-box[0])*np.asarray(mesh.block_shape),
                  nw=len(names), w_names=list(names), nleafs=count,
                  nparents=nodes-count, levmax=int(levels.max()), staggered=False)
    maximum = np.iinfo(np.int32).max
    if (count > maximum or nodes-count > maximum or len(names) > maximum or
            np.any(header['domain_nx'] > maximum) or
            np.any(np.asarray(mesh.block_shape) > maximum) or
            np.any(coordinates > maximum) or np.any(levels > maximum)):
        raise ValueError('AMRVAC geometry exceeds the signed 32-bit file representation')
    return header, flags, (levels, coordinates), ids, required


def _batch_count(count, block_bytes, required, memory_limit, read_scratch=0):
    size = min(count, _BATCH_LEAVES, max(1, _BATCH_BYTES // block_bytes))
    # One block's disk-layout bytes plus reader scratch coexist with the batch.
    fixed = required + block_bytes + read_scratch
    admit(fixed + block_bytes, memory_limit, 'AMRVAC export batch')
    if memory_limit is not None:
        size = min(size, (memory_limit-fixed)//block_bytes)
    return size


def _destination(path, overwrite):
    if type(overwrite) is not bool:
        raise ValueError('overwrite must be boolean')
    destination = Path(path)
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(destination)
    return destination


def _publish(destination, overwrite, plan, batches):
    from ._v5.writer import write_datfile_from_batches

    header, flags, tree, _, _ = plan
    fd, temporary = tempfile.mkstemp(prefix='.simesh-', suffix='.dat', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            written = write_datfile_from_batches(stream, batches, header, flags, tree)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return written
    finally:
        Path(temporary).unlink(missing_ok=True)


def _field_batches(fields, ids, size):
    shape = (len(fields.fields), *fields.mesh.block_shape)
    buffer = np.empty((size, *shape), dtype=np.float64)
    h = fields.storage_halo
    box = tuple(slice(h, h+n) for n in fields.mesh.block_shape)
    for start in range(0, len(ids), size):
        backing = fields.values  # Validate borrowed lifetime for each batch.
        chosen = ids[start:start+size]
        for row, leaf in enumerate(chosen):
            buffer[row] = np.moveaxis(backing[(fields.slot_of_leaf[leaf], *box, slice(None))], -1, 0)
        yield buffer[:len(chosen)]


def write_amrvac(path, fields, *, metadata, root_bounds=None, overwrite=False, memory_limit=None):
    """Export complete-mesh or root-aligned regional Fields interiors as AMRVAC v5.

    Parameters
    ----------
    path : str or Path
        Destination file; its parent must exist.
    fields : Fields
        Completed ordinary fields covering every exported leaf; extra coverage is
        ignored. Storage order is independent of original leaf IDs. Halo is omitted.
        The original Mesh must be nonperiodic, including for regional export.
    metadata : SnapshotMetadata or dict
        Explicit model/time header matching the original Fields Mesh.
    root_bounds : array-like, optional
        Zero-based integer (2, 3) root-block box with exclusive upper bounds. Omit
        to require complete original-mesh coverage. Levels and physical coordinates
        are retained; only output geometry and indices are rebased.
    overwrite : bool
        Allow replacing an existing destination.
    memory_limit : int, optional
        Accounted-array budget, including input Fields and Mesh, export indices and
        bounded payload scratch; not a process RSS limit.

    Returns
    -------
    dict
        Written header. Publication is atomic; failures retain an existing file.

    Notes
    -----
    The file omits units, preparation/derivation provenance, halo and CT values;
    export does not certify a solver restart. New file edges delimit its domain,
    so derivative results near cropped edges need not match the original domain.
    Input Fields and their original Mesh are unchanged.
    """
    fields = require_fields(fields)
    destination = _destination(path, overwrite)
    if root_bounds is None and (len(fields.leaf_ids) != fields.mesh.leaf_count or
                               np.any(fields.slot_of_leaf < 0)):
        raise ValueError('AMRVAC export requires complete original-mesh coverage')
    plan = _export_plan(fields.mesh, metadata, [f.name for f in fields.fields], root_bounds,
                        fields.nbytes + fields.mesh.nbytes, memory_limit)
    ids = plan[3]
    if np.any(fields.slot_of_leaf[ids] < 0):
        raise ValueError('AMRVAC export requires complete selected-root coverage')
    block_bytes = 8*len(fields.fields)*math.prod(fields.mesh.block_shape)
    size = _batch_count(len(ids), block_bytes, plan[4], memory_limit)
    return _publish(destination, overwrite, plan, _field_batches(fields, ids, size))


def _source_batches(source, ids, field_ids, size):
    buffer = np.empty((size, len(field_ids), *source.mesh.block_shape), dtype=np.float64)
    for start in range(0, len(ids), size):
        chosen = ids[start:start+size]
        target = buffer[:len(chosen)]
        source.read_into(chosen, field_ids, target)
        yield target
    source.validate()


def crop_amrvac(path, output_path, *, root_bounds, fields=None, overwrite=False, memory_limit=None):
    """Copy selected root subtrees and stored fields into an independent v5 snapshot.

    Parameters
    ----------
    path : str or Path
        Nonperiodic Cartesian 3D AMRVAC v5 input. Only ordinary cell fields are
        read; any saved ghosts and CT tail are omitted from output.
    output_path : str or Path
        Destination file with an existing parent, distinct from the input file.
    root_bounds : array-like
        Nonempty zero-based integer (2, 3) root-block box, upper bounds exclusive.
    fields : str or sequence, optional
        Stored field names or local input-header indices, in output order. Integer
        selectors must be supplied as a sequence; None selects all input fields.
    overwrite : bool
        Allow replacing an existing destination.
    memory_limit : int, optional
        Accounted-array budget for global input Mesh/index, export index and bounded
        read/write scratch; not a process RSS limit. No whole field payload is loaded.

    Returns
    -------
    dict
        Written header, with original time/model metadata and updated domain/tree/fields.
        Publication is atomic; failures retain an existing destination.

    Notes
    -----
    No interpolation, halo preparation or derived-field computation is performed.
    The output has the same boundary/provenance limitations as write_amrvac.
    """
    from .amrvac import open_amrvac

    destination = _destination(output_path, overwrite)
    if Path(path).resolve() == destination.resolve() or (
            destination.exists() and os.path.samefile(path, destination)):
        raise ValueError('input and output must be different files')
    with open_amrvac(path, fields=fields, memory_limit=memory_limit) as source:
        plan = _export_plan(source.mesh, source.metadata, [f.name for f in source.fields],
                            root_bounds, source.mesh.nbytes + source.nbytes, memory_limit)
        ids = plan[3]
        field_ids = source.field_ids()
        block_bytes = 8*len(field_ids)*math.prod(source.mesh.block_shape)
        size = _batch_count(len(ids), block_bytes, plan[4], memory_limit,
                            source.read_footprint(min(len(ids), _BATCH_LEAVES), len(field_ids)))
        return _publish(destination, overwrite, plan, _source_batches(source, ids, field_ids, size))
