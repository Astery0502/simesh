"""AMRVAC v5 ordinary-field sources, independent of preparation strategies."""

import os
import numpy as np

from ._v5.index import read_amrvac_v5_index, bind_amrvac_v5_forest
from ._v5.reader import make_amrvac_v5_ordinary_block_reader
from .source import Source, definitions_from_names
from .metadata import SnapshotMetadata
from ..mesh import _from_validated
from .._validation import admit, array_bytes


def open_amrvac(path, *, fields=None, units=None, memory_limit=None, boundary=None):
    """Open immutable Cartesian 3D v5 ordinary fields with halo periodicity.

    Parameters
    ----------
    path : str or Path
        Ordinary Cartesian 3D AMRVAC v5 snapshot; CT face values are not exposed.
        Axis-periodic metadata is retained by Mesh for exact-phase halo preparation
        only. Periodic coordinate and trajectory semantics are not provided.
    fields : str or sequence, optional
        Names or local Source indices of stored fields; integers must be supplied as a
        sequence.
    units : str or mapping, optional
        Field unit labels, without numerical scaling.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    boundary : str, mapping or array-like, optional
        Explicit physical halo rules for the full file field directory, before
        selecting fields; see Source. The snapshot stores only periodic flags,
        so omitted physical rules default to continuous.

    Returns
    -------
    Source
        Metadata/index input without loaded field payload or halos; use its
        context-manager lifetime.
    """
    fd = os.open(os.fspath(path), os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(fd)
        if index.dimension_count != 3 or index.geometry not in ("Cartesian", "Cartesian_3D"):
            raise ValueError("source profile requires Cartesian 3D v5 ordinary fields")
        metadata_bytes = array_bytes(index)
        admit(metadata_bytes + index.leaf_count*2048 + len(index.forest_flags)*512,
              memory_limit, "source metadata")
        binding = bind_amrvac_v5_forest(index)
        mesh = _from_validated(binding.root_shape, binding.coord_to_rank, binding.forest,
                               index.domain_lower, index.domain_upper, index.block_cell_counts,
                               periodic=index.periodic)
        reader = make_amrvac_v5_ordinary_block_reader(fd, index, binding)
        definitions = definitions_from_names(index.field_names, units)
        offsets = np.r_[index.block_offsets, index.file_identity[2]]
        maximum_record = max((int(b)-int(a) for a, b in zip(offsets[:-1], offsets[1:])), default=0)

        def validate():
            st = os.fstat(fd)
            identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if identity != index.file_identity:
                raise OSError("source file changed; open a new immutable source")

        def read_native(ids, fields, target):
            from ._bulk import read_full_interiors
            from ._v5.reader import _read_amrvac_v5_blocks_into
            if len(ids) == mesh.leaf_count and np.array_equal(ids, np.arange(mesh.leaf_count)):
                _, stats = read_full_interiors(reader, fields, output=np.moveaxis(target,-1,1))
                return stats
            # The Source boundary has validated the final backing. The ordinary
            # reader's NumPy transfers support this strided interior view while
            # retaining selected-field I/O, header and identity checks.
            zero = np.zeros(3,dtype=np.int64)
            _read_amrvac_v5_blocks_into(reader.state,zero,np.asarray(mesh.block_shape,dtype=np.int64),
                ids,fields,np.moveaxis(target,-1,1),zero,native_output=True)
            return {"selected_load_count":len(ids),"read_value_bytes":target.nbytes}

        source = Source(mesh, definitions, reader, validate=validate, close=lambda: os.close(fd),
                        read_native=read_native, read_scratch_bytes=2*max(maximum_record, 16*1024**2),
                        metadata_arrays=(*index, binding.root_shape, binding.rank_to_coord, binding.coord_to_rank),
                        metadata=SnapshotMetadata._from_amrvac_index(index, path), boundary=boundary)
        admit(mesh.nbytes+source.nbytes, memory_limit, "source metadata")
        if fields is not None:
            from .source import select_source
            selected = select_source(source, fields)
            selected._close = source.close
            admit(mesh.nbytes+selected.nbytes,memory_limit,"source metadata")
            return selected
        return source
    except BaseException:
        os.close(fd)
        raise
