"""Owned file contexts around explicit native analysis source adapters."""

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import replace
import os
import time
import numpy as np

from simesh.analysis.fields import FieldDefinition
from simesh.analysis.mesh import indices,frozen_array


@contextmanager
def open_source(path,*,field_names=None,field_indices=None,field_units=None,
                support_capacity=128,value_cache_capacity=0,budget_bytes=2*1024**3):
    """Open immutable Cartesian 3D, nonperiodic v5 ordinary fields for analysis.

    Staggered tails are validated/skipped, not exposed as CT fields. Selected
    file fields become source columns 0..K-1; original IDs remain recorded.
    The context owns its fd. Detached products survive context exit; new reads
    must finish before exit. No payload is eagerly loaded by this operation.
    value_cache_capacity optionally retains that many ordinary-interior blocks;
    it is separate from PreparedPool's completed two-halo values. File changes
    or context exit reject subsequent preparation/borrows, including cache hits.
    """
    from simesh_rewrite.amrvac_dat import read_amrvac_v5_index,bind_amrvac_v5_forest
    from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_ordinary_block_reader
    from simesh_rewrite.blockio import make_block_reader,read_blocks_into
    from simesh.analysis.providers import make_source
    if field_names is not None and field_indices is not None:
        raise ValueError("choose field_names or original file field_indices")
    if field_units is not None and not isinstance(field_units,(str,Mapping)):
        raise ValueError("field_units must be a unit string or field-name mapping")
    fd = os.open(os.fspath(path),os.O_RDONLY)
    active = True
    try:
        index = read_amrvac_v5_index(fd)
        if index.dimension_count!=3 or index.geometry!='Cartesian_3D' or np.any(index.periodic):
            raise ValueError("analysis source requires nonperiodic Cartesian 3D; retain canonical APIs for other workflows")
        if field_names is not None:
            if isinstance(field_names,str):
                field_names = [field_names]
            selected = []
            for name in field_names:
                matches = [i for i,value in enumerate(index.field_names) if value==name]
                if len(matches)!=1:
                    raise ValueError(f"field name is missing or ambiguous: {name}")
                selected.append(matches[0])
            original = indices(np.asarray(selected,dtype=np.int64),index.field_count,"field_indices")
        else:
            original = indices(np.arange(index.field_count,dtype=np.int64) if field_indices is None else field_indices,
                               index.field_count,"field_indices")
        if not len(original):
            raise ValueError("select at least one ordinary field")
        original = frozen_array(original,np.int64)
        if type(value_cache_capacity) is not int or value_cache_capacity<0:
            raise ValueError("value_cache_capacity must be a nonnegative integer")
        cache_bound = (min(value_cache_capacity,index.leaf_count)*
                       (len(original)*8*int(np.prod(index.block_cell_counts))+128)+
                       (8*index.leaf_count if value_cache_capacity else 0))
        # Metadata is O(nodes/leaves), admitted before constructing geometry.
        metadata_bytes = sum(a.nbytes for a in index if isinstance(a,np.ndarray))
        metadata_upper = metadata_bytes+index.leaf_count*2048+len(index.forest_flags)*512
        if metadata_upper+cache_bound>budget_bytes:
            raise MemoryError("analysis metadata exceeds the supplied budget")
        binding = bind_amrvac_v5_forest(index)
        raw_reader = make_amrvac_v5_ordinary_block_reader(fd,index,binding)
        def read_selected(state,lower,upper,ids,fields,output,out_lower):
            read_blocks_into(raw_reader,lower,upper,ids,original[fields],output,out_lower)
        reader = make_block_reader(None,(index.leaf_count,len(original),*index.block_cell_counts),read_selected,
                                  memory_arrays=(*raw_reader.memory_arrays,original))
        definitions = []
        for field in original:
            name = index.field_names[field]
            units = field_units if isinstance(field_units,str) else (field_units or {}).get(name,"code")
            if not isinstance(units,str):
                raise ValueError("each field unit must be a string")
            definitions.append(FieldDefinition(name,units))
        maximum_record = max((int(end)-int(start) for start,end in
            zip(index.block_offsets,np.r_[index.block_offsets[1:],index.file_identity[2]])),default=0)
        # One raw run plus endian copy; record/header Python objects are bounded
        # by the adapter's transfer capacity and included conservatively.
        backend_bound = 2*maximum_record+max(57,support_capacity)*4096
        extra = metadata_bytes+binding.rank_to_coord.nbytes+binding.coord_to_rank.nbytes+binding.root_shape.nbytes
        def validate_values():
            if not active:
                raise OSError("analysis source context is closed")
            st = os.fstat(fd)
            identity = (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            if identity != index.file_identity:
                raise OSError("analysis source values changed; open a new source and numerical pools")
        source = make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
            index.domain_lower,index.domain_upper,index.block_cell_counts,reader,definitions,
            support_capacity=support_capacity,extra_resident_bytes=extra,
            backend_scratch_bytes=backend_bound,original_field_ids=tuple(map(int,original)),
            value_cache_capacity=value_cache_capacity,validate_values=validate_values)
        if source.mesh.nbytes+source.resident_bytes+source.scratch_bytes(np.arange(len(original)),2)>budget_bytes:
            raise MemoryError("minimum source/preparation working set exceeds budget")
        yield source
    finally:
        active = False
        os.close(fd)


def open_prepared(path,*,field_names=None,field_indices=None,field_units=None,
                  budget_bytes=2*1024**3):
    """Read selected full-domain interiors and use the canonical bulk provider.

    The returned immutable native product owns its canonical C backing and needs
    no open file. All raw/reader, padded/coarse/geometry storage is admitted.
    Use open_source plus a PreparedPool for bounded original-file access.
    """
    from .analysis import prepare_resident
    start = time.perf_counter()
    with open_source(path,field_names=field_names,field_indices=field_indices,field_units=field_units,
                     budget_bytes=budget_bytes) as source:
        opened = time.perf_counter()
        mesh = source.mesh
        count = mesh.leaf_count
        k = len(source.fields)
        block = np.asarray(mesh.block_shape,dtype=np.int64)
        raw_bytes = count*k*8*int(np.prod(block))
        retained = count*k*8*(int(np.prod(block+4))+int(np.prod(block//2+4)))
        geometry = count*1400+len(mesh.node_leaves)*260+4096
        read_scratch = source.scratch_bytes(np.arange(k,dtype=np.int64),0)
        required = max(source.resident_bytes+mesh.nbytes+raw_bytes+read_scratch,
                       source.resident_bytes+mesh.nbytes+raw_bytes+retained+geometry+16*count)
        if required>budget_bytes:
            raise MemoryError(f"resident file workflow needs {required} controlled bytes; use bounded source")
        raw = np.empty((count,k,*block),dtype=float)
        # Dense private loading benefits from sequential header/payload order on
        # the measured large-file path; every one-record callback stays checked.
        source.read_interiors(np.arange(count,dtype=np.int64),np.arange(k,dtype=np.int64),raw,batch_size=1)
        read = time.perf_counter()
        product = prepare_resident(mesh,mesh.roots.shape,mesh.node_leaves>=0,raw,source.fields,
                                   budget_bytes=budget_bytes-source.resident_bytes)
        stats = {**product.preparation_stats,"open_seconds":opened-start,"read_seconds":read-opened,
                 "file_to_prepared_seconds":time.perf_counter()-start,"workflow_controlled_upper_bytes":required,
                 "original_field_ids":source.original_field_ids}
        return replace(product,preparation_stats=stats)
