"""Explicit ordinary AMRVAC file products from completed native fields."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile

import numpy as np

from .metadata import SnapshotMetadata
from ..fields import require_fields
from .._validation import admit


def write_amrvac(path, fields, *, metadata, overwrite=False, memory_limit=None):
    """Export full-mesh Fields interiors using explicit AMRVAC v5 metadata.

    Parameters
    ----------
    path : str or Path
        Destination file or directory; its parent must exist.
    fields : Fields
        Complete original-mesh coverage; ordinary interiors are serialized in SFC order,
        without halo or CT values.
    metadata : SnapshotMetadata or dict
        Explicit model/time header matching the complete original Mesh.
    overwrite : bool
        Allow replacing an existing destination; otherwise an existing file is refused.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    dict
        Written header. Publication is atomic and failures retain an existing
        destination; the memory estimate includes an interior conversion copy.

    Notes
    -----
    The file cannot encode field-unit labels, preparation schemes or derivation
    provenance; record these separately. Export is not certification of a solver
    restart state.
    """
    require_fields(fields)
    if type(overwrite) is not bool or not isinstance(metadata, (SnapshotMetadata, dict)):
        raise ValueError('explicit SnapshotMetadata or header dictionary and boolean overwrite required')
    mesh = fields.mesh
    if len(fields.leaf_ids) != mesh.leaf_count or np.any(fields.slot_of_leaf < 0):
        raise ValueError('AMRVAC export requires complete original-mesh coverage')
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
    maximum = np.iinfo(np.int32).max
    if (np.any(expected['domain_nx'] > maximum) or
            np.any(np.asarray(mesh.block_shape) > maximum) or
            np.any(mesh.forest.node_coords[mesh.leaf_nodes] >= maximum) or
            np.any(mesh.forest.node_levels > maximum)):
        raise ValueError('AMRVAC geometry exceeds the signed 32-bit file representation')
    names = [definition.name for definition in fields.fields]
    try:
        encoded = [name.encode('ascii') for name in names]
    except UnicodeEncodeError:
        raise ValueError('AMRVAC field names must be ASCII') from None
    if len(set(names)) != len(names) or any(len(name) > 16 for name in encoded):
        raise ValueError('AMRVAC field names must be distinct and at most 16 ASCII bytes')
    destination = Path(path)
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(destination)
    shape = (mesh.leaf_count, len(names), *mesh.block_shape)
    required = (fields.nbytes + mesh.nbytes + 8*int(np.prod(shape)) +
                8*int(np.prod(shape[1:])) + mesh.leaf_count*128)
    admit(required, memory_limit, 'AMRVAC export')
    data = np.empty(shape)
    h = fields.storage_halo
    box = tuple(slice(h, h+n) for n in mesh.block_shape)
    backing = fields.values
    for leaf, slot in enumerate(fields.slot_of_leaf):
        data[leaf] = np.moveaxis(backing[(slot, *box, slice(None))], -1, 0)
    forest = mesh.forest
    nodes = mesh.leaf_nodes
    header.update(nw=len(names), w_names=names, nleafs=mesh.leaf_count,
                  nparents=len(mesh.node_leaves)-mesh.leaf_count,
                  levmax=int(forest.node_levels.max()), staggered=False)
    tree = (forest.node_levels[nodes], forest.node_coords[nodes]+1)
    from ._v5.writer import write_datfile_from_sfc

    fd, temporary = tempfile.mkstemp(prefix='.simesh-', suffix='.dat', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            written = write_datfile_from_sfc(stream, data, header,
                                           mesh.node_leaves >= 0, tree)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return written
    finally:
        Path(temporary).unlink(missing_ok=True)
