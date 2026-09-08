"""Explicit snapshots and file products across the stateful compatibility edge."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile

import numpy as np

from .source import source_from_arrays, definitions_from_names
from ..fields import FieldDefinition, require_fields
from ..mesh import mesh_from_forest
from .._validation import admit


def source_from_dataset(dataset, fields=None, *, units=None, memory_limit=None):
    """Snapshot loaded 3D Dataset interiors into an independent immutable Source.

    Field names address loaded columns, including materialized derived columns.
    No ghost values or mutable Dataset/AMRMesh references cross this boundary.
    Load the desired original fields first; this adapter performs no file reads.
    """
    from ..amrvac.amrvac_dataset import AMRVACDataSet

    if not isinstance(dataset, AMRVACDataSet) or dataset.data is None:
        raise ValueError('source_from_dataset requires a Dataset with loaded fields')
    if (int(dataset.ndim) != 3 or dataset.geometry != 'Cartesian_3D' or
            np.any(dataset.periodic)):
        raise ValueError('native Source requires nonperiodic Cartesian 3D geometry')
    names = list(dataset.loaded_field_names if fields is None else
                 [fields] if isinstance(fields, str) else fields)
    if not names or len(set(names)) != len(names):
        raise ValueError('select distinct loaded field names')
    columns = dataset._columns_for_field_names(names)
    block = tuple(map(int, dataset.block_nx))
    count = int(dataset.nleafs)
    shape = (count, len(names), *block)
    # Dataset's padded backing can be larger than its public interior view.
    live = np.asarray(dataset.data).nbytes
    if dataset.ghost_width:
        live = max(live, dataset.mesh.padded_view().nbytes)
        coarse = dataset.mesh.datac
        if coarse is not None:
            live += np.asarray(coarse).nbytes
    required = live + 8*int(np.prod(shape)) + count*4096 + len(dataset.is_leaf)*512
    admit(required, memory_limit, 'Dataset snapshot')
    mesh = mesh_from_forest(
        np.asarray(dataset.domain_nx, dtype=np.int64)//np.asarray(block),
        np.asarray(dataset.is_leaf, dtype=bool), lower=dataset.physical_domain[0],
        upper=dataset.physical_domain[1], block_shape=block,
    )
    values = np.empty(shape)
    for target, column in enumerate(columns):
        values[:, target] = dataset.data[:, column]
    values.flags.writeable = False
    definitions = definitions_from_names(names, units)
    derived = set(dataset.derived_field_names)
    definitions = tuple(FieldDefinition(d.name, d.units,
        'materialized-derived' if d.name in derived else 'cell-average') for d in definitions)
    return source_from_arrays(mesh, values, definitions, copy=False)


def write_amrvac(path, fields, *, metadata, overwrite=False, memory_limit=None):
    """Export full-mesh Fields interiors using explicit AMRVAC v5 metadata.

    The output stores ordinary cell-centered values in original SFC order;
    ghost layers and CT face values are not serialized. Unit labels, derivation
    provenance and preparation schemes are not representable in the dat header.
    Metadata describes the physical model/time and must match the mesh geometry.
    This is a data product, not certification of a solver restart state.

    One field-major interior copy is admitted explicitly. A temporary sibling
    file is published only after serialization succeeds; overwrite=False also
    refuses destinations created concurrently. The temporary file is removed on
    failure, leaving an existing destination intact.
    """
    require_fields(fields)
    if type(overwrite) is not bool or not isinstance(metadata, dict):
        raise ValueError('explicit metadata dictionary and boolean overwrite required')
    mesh = fields.mesh
    if len(fields.leaf_ids) != mesh.leaf_count or np.any(fields.slot_of_leaf < 0):
        raise ValueError('AMRVAC export requires complete original-mesh coverage')
    header = deepcopy(metadata)
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
    shape = (mesh.leaf_count, len(names), *mesh.block_shape)
    required = (fields.nbytes + mesh.nbytes + 8*int(np.prod(shape)) +
                96*int(np.prod(shape[1:])) + mesh.leaf_count*128)
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
    tree = (forest.node_levels[nodes], forest.node_coords[nodes]+1,
            np.zeros(mesh.leaf_count, dtype=np.int64))
    from ..amrvac.datio import write_datfile_from_sfc

    destination = Path(path)
    if not overwrite and destination.exists():
        raise FileExistsError(destination)
    fd, temporary = tempfile.mkstemp(prefix='.simesh-', suffix='.dat', dir=destination.parent)
    os.close(fd)
    try:
        written = write_datfile_from_sfc(temporary, data, header,
                                       mesh.node_leaves >= 0, tree, overwrite=True)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return written
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
