"""Explicit adapter to the bundled, contract-preserving rewrite provider.

The numerical core never imports this adapter. Source arrays and borrowed file
descriptors remain unchanged/alive until preparation ends.
"""

import time

import numpy as np

from .fields import FieldSource
from .mesh import MeshIndex, frozen_array, indices
from simesh_rewrite.blockio import read_blocks_into
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.completed_primary import (
    make_completed_primary_consumer, execute_selected_refined_halos_with_consumer,
)
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.refined_halo import _allocate_workspace


def make_source(root_shape, coord_to_rank, forest, lower, upper, block_shape,
                reader, definitions, *, support_capacity=128, extra_resident_bytes=0,
                backend_scratch_bytes=0, original_field_ids=None):
    """Assemble validated 3D forest/reader and continuous two-layer preparation."""
    root_shape = frozen_array(root_shape, np.int64)
    coord_to_rank = frozen_array(coord_to_rank, np.int64)
    lower, upper = frozen_array(lower, float), frozen_array(upper, float)
    block = frozen_array(block_shape, np.int64)
    domain = root_shape * block
    f = forest
    validate_refined_all_touch_2to1(root_shape, coord_to_rank, f.root_node_ids,
        f.node_levels, f.node_coords, f.child_node_ids, f.node_leaf_ids, f.leaf_node_ids)
    leaf_count = len(f.leaf_node_ids)
    if reader.shape != (leaf_count, len(definitions), *block.tolist()):
        raise ValueError("reader, field definitions and geometry disagree")
    if np.any(block < 4) or np.any(block % 2):
        raise ValueError("bootstrap supports even 3D blocks of at least four cells")
    bounds, spacing = refined_leaf_geometry(lower, upper, root_shape, domain, block,
        f.node_levels, f.node_coords, f.leaf_node_ids, np.arange(leaf_count, dtype=np.int64))
    scales = np.left_shift(1, f.node_levels-1)
    dx = np.ldexp((upper-lower)/domain, -(f.node_levels-1)[:, None])
    lower_indices = f.node_coords * block
    upper_indices = lower_indices + block
    node_lower = lower + lower_indices*dx
    node_upper = lower + upper_indices*dx
    node_upper = np.where(upper_indices == domain*scales[:, None], upper, node_upper)
    mesh = MeshIndex(lower, upper, tuple(block.tolist()),
        frozen_array(f.root_node_ids[coord_to_rank], np.int64),
        frozen_array(f.child_node_ids, np.int64), frozen_array(f.node_leaf_ids, np.int64),
        frozen_array(node_lower, float), frozen_array(node_upper, float),
        frozen_array(f.leaf_node_ids, np.int64), frozen_array(bounds, float),
        frozen_array(spacing, float))
    capacity = min(max(57, support_capacity), leaf_count)
    # Query fixed provider overhead using a zero-capacity, zero-field workspace;
    # admission never allocates the requested full payload just to count bytes.
    _, fixed_arrays = _allocate_workspace(0, 0, tuple(block), tuple(block+4))
    fixed_bytes = sum(a.nbytes for a in fixed_arrays) + 4*3*8
    del fixed_arrays
    provider_geometry = sum(a.nbytes for a in f if isinstance(a, np.ndarray))
    resident = (sum(a.nbytes for a in reader.memory_arrays) + provider_geometry +
                root_shape.nbytes + coord_to_rank.nbytes + block.nbytes + domain.nbytes +
                extra_resident_bytes)

    def scratch_bytes(fields, halo):
        if halo == 0:
            return (capacity*8*len(fields)*int(np.prod(block)) + backend_scratch_bytes +
                    capacity*4096 + 32*leaf_count + 4096)
        return (capacity*(8*len(fields)*int(np.prod(block+2*halo))+1854) +
                8*len(fields)*int(np.prod(block+1)) + fixed_bytes + 8*len(fields) +
                32*leaf_count + 4096 + backend_scratch_bytes)

    def read_interiors(ids, fields, output, *, batch_size=None):
        ids = indices(ids,leaf_count)
        fields = indices(fields,len(definitions),"field_ids")
        if (not isinstance(output,np.ndarray) or output.dtype!=np.float64 or
                output.shape!=(len(ids),len(fields),*block) or not output.flags.c_contiguous or
                not output.flags.writeable):
            raise ValueError("raw destination must match selected field-major interiors")
        if any(np.shares_memory(output,a) for a in (*reader.memory_arrays,ids,fields)):
            raise ValueError("raw destination must not alias source/selector arrays")
        zero = np.zeros(3,dtype=np.int64)
        batch = capacity if batch_size is None else batch_size
        if type(batch) is not int or batch<1 or batch>capacity:
            raise ValueError("raw I/O batch size must fit the admitted support capacity")
        calls = 0
        start = time.perf_counter()
        for first in range(0,len(ids),batch):
            stop = min(first+batch,len(ids))
            read_blocks_into(reader,zero,block,ids[first:stop],fields,output[first:stop],zero)
            calls += 1
        return {"reader_call_count":calls,"selected_load_count":len(ids),
                "read_value_bytes":len(ids)*len(fields)*8*int(np.prod(block)),
                "total_seconds":time.perf_counter()-start}

    def fill(ids, fields, halo, output):
        if halo == 0:
            temporary = np.empty((min(capacity,len(ids)),len(fields),*block),dtype=float)
            stats = {"reader_call_count":0,"selected_load_count":0,"read_value_bytes":0}
            start = time.perf_counter()
            for first in range(0,len(ids),capacity):
                stop = min(first+capacity,len(ids))
                current = read_interiors(ids[first:stop],fields,temporary[:stop-first])
                output[first:stop] = np.moveaxis(temporary[:stop-first],1,-1)
                for name in stats:
                    stats[name] += current[name]
            return {**stats,"total_seconds":time.perf_counter()-start,
                    "scratch_admission_bytes":scratch_bytes(fields,halo)}
        order = np.argsort(ids, kind="stable")
        sorted_ids = np.ascontiguousarray(ids[order])
        widths = np.full(3, halo, dtype=np.int64)
        modes = np.zeros((len(fields), 6), dtype=np.uint8)
        normals = np.full(3, -1, dtype=np.int64)
        packing_seconds = 0.

        def consume(state, offset, primary_ids, payload, valid_lo, valid_hi,
                    interior_lo, interior_hi):
            nonlocal packing_seconds
            started = time.perf_counter()
            count = len(primary_ids)
            if not (np.all(valid_lo == 0) and np.all(valid_hi == block+2*halo)):
                raise RuntimeError("provider did not complete the requested two-layer box")
            # Advanced indexing assignment scatters a view; no full output gather.
            output[order[offset:offset+count]] = np.moveaxis(payload[:count], 1, -1)
            packing_seconds += time.perf_counter()-started

        consumer = make_completed_primary_consumer(None, consume, output_arrays=(output,))
        start = time.perf_counter()
        stats = execute_selected_refined_halos_with_consumer(reader, consumer, sorted_ids,
            fields, root_shape, coord_to_rank, f.root_node_ids, f.node_levels, f.node_coords,
            f.child_node_ids, f.node_leaf_ids, f.leaf_node_ids, widths, widths, modes,
            normals, capacity)
        return {**stats._asdict(), "total_seconds": time.perf_counter()-start,
                "packing_seconds": packing_seconds,
                "read_value_bytes": stats.selected_load_count*len(fields)*8*int(np.prod(block)),
                "scratch_admission_bytes": scratch_bytes(fields, halo)}

    def plan_builder(source, ids, capacity, budget):
        from .geometry_plans import _build
        return _build(source, ids, capacity, budget, root_shape, coord_to_rank, f)

    return FieldSource(mesh, tuple(definitions), fill, scratch_bytes, resident,
                       "rewrite-ratio2-minmod-exactphase-cont-v1",
                       memory_arrays=reader.memory_arrays,read_interiors=read_interiors,
                       original_field_ids=original_field_ids,plan_builder=plan_builder)
