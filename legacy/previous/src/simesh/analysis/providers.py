"""Direct analysis preparation with the bundled exact-phase transfer primitives.

The numerical core never imports this adapter. Source arrays and borrowed file
descriptors remain unchanged/alive until preparation ends. Validated geometry
belongs to the source lifetime; private prepared values are published by callers
only after this adapter completes. Standalone rewrite executors remain unchanged.
"""

import time

import numpy as np

from .fields import FieldSource
from .mesh import MeshIndex, frozen_array, indices
from simesh_rewrite.blockio import read_blocks_into
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite import refined_halo as halo_provider
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.refined_halo import _allocate_workspace
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes


def make_source(root_shape, coord_to_rank, forest, lower, upper, block_shape,
                reader, definitions, *, support_capacity=128, extra_resident_bytes=0,
                backend_scratch_bytes=0, original_field_ids=None,
                value_cache_capacity=0, validate_values=None):
    """Assemble validated 3D forest/reader and continuous two-layer preparation.

    Optional value_cache_capacity retains raw interiors for one ordered field
    group at a time. The reader/source values remain immutable for this entire
    source lifecycle; validate_values may raise when that promise ends.
    Forest arrays likewise remain immutable; topology and fixed transfer scope
    are validated here rather than before every batch's numerical operations.
    """
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
    if type(value_cache_capacity) is not int or value_cache_capacity < 0:
        raise ValueError("value_cache_capacity must be a nonnegative integer")
    value_cache = None
    if value_cache_capacity:
        from .value_cache import InteriorValueCache
        value_cache = InteriorValueCache(reader,value_cache_capacity,validate_values)
        reader = value_cache.reader
        # Miss staging and bounded selector/LRU temporaries are independent of
        # the retained cache payload already included in reader.memory_arrays.
        backend_scratch_bytes += (capacity*len(definitions)*8*int(np.prod(block))+
                                  64*(capacity+value_cache.capacity)+4096)
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
        if validate_values is not None:
            validate_values()
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
        cache_before = dict(value_cache.stats) if value_cache is not None else {}
        start = time.perf_counter()
        for first in range(0,len(ids),batch):
            stop = min(first+batch,len(ids))
            read_blocks_into(reader,zero,block,ids[first:stop],fields,output[first:stop],zero)
            calls += 1
        loaded = value_cache.stats['value_cache_misses']-cache_before['value_cache_misses'] if value_cache else len(ids)
        return {"reader_call_count":calls,"selected_load_count":len(ids),
                "read_value_bytes":loaded*len(fields)*8*int(np.prod(block)),
                "requested_value_bytes":len(ids)*len(fields)*8*int(np.prod(block)),
                "total_seconds":time.perf_counter()-start}

    def fill(ids, fields, halo, output):
        if validate_values is not None:
            validate_values()
        ids = indices(ids,leaf_count)
        fields = indices(fields,len(definitions),"field_ids")
        if type(halo) is not int or halo not in (0,2) or not len(fields):
            raise ValueError("prepare nonempty fields with zero or two halo layers")
        if (not isinstance(output,np.ndarray) or output.dtype!=np.float64 or
                output.shape!=(len(ids),*(block+2*halo),len(fields)) or
                not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("destination must match selected component-adjacent prepared blocks")
        metadata = (root_shape,coord_to_rank,lower,upper,block,domain,ids,fields,
                    *(a for a in f if isinstance(a,np.ndarray)))
        if any(np.shares_memory(output,a) for a in (*reader.memory_arrays,*metadata)):
            raise ValueError("destination must not alias source or request arrays")
        cache_before = dict(value_cache.stats) if value_cache is not None else {}
        if halo == 0:
            temporary = np.empty((min(capacity,len(ids)),len(fields),*block),dtype=float)
            stats = {"reader_call_count":0,"selected_load_count":0,"read_value_bytes":0,"requested_value_bytes":0}
            start = time.perf_counter()
            for first in range(0,len(ids),capacity):
                stop = min(first+capacity,len(ids))
                current = read_interiors(ids[first:stop],fields,temporary[:stop-first])
                output[first:stop] = np.moveaxis(temporary[:stop-first],1,-1)
                for name in stats:
                    stats[name] += current[name]
            cache_stats = ({key:value-cache_before[key] for key,value in value_cache.stats.items()}
                           if value_cache is not None else {})
            return {**stats,**cache_stats,"total_seconds":time.perf_counter()-start,
                    "scratch_admission_bytes":scratch_bytes(fields,halo)}
        order = np.argsort(ids, kind="stable")
        sorted_ids = np.ascontiguousarray(ids[order])
        widths = np.full(3, halo, dtype=np.int64)
        modes = np.zeros((len(fields), 6), dtype=np.uint8)
        normals = np.full(3, -1, dtype=np.int64)
        start = time.perf_counter()
        w, arrays = _allocate_workspace(capacity,len(fields),tuple(block),tuple(block+2*halo))
        zero = np.zeros(3,dtype=np.int64)
        interior_upper = widths+block
        fill_directed_halo_target_boxes(widths,interior_upper,zero,block+2*halo,
            halo_provider.CANONICAL_DIRECTIONS,w.target_lower,w.target_upper)
        first = chunks = loads = maximum = 0
        planning_seconds = reader_seconds = ghost_seconds = packing_seconds = 0.
        while first < len(sorted_ids):
            stamp = time.perf_counter()
            # Geometry, 2:1 balance, even block shape and continuous two-halo
            # transfer are fixed at source construction. Bind each batch once;
            # do not repeat the standalone executor's all-request preflight.
            # Active source-slot guards remain inside this binding operation.
            count, selected = halo_provider._prepare_refined_halo_chunk(
                w,sorted_ids[first:first+capacity],root_shape,coord_to_rank,
                f.root_node_ids,f.node_levels,f.node_coords,f.child_node_ids,
                f.node_leaf_ids,f.leaf_node_ids,widths,interior_upper,modes,normals,
                validate_actions=False)
            planning_seconds += time.perf_counter()-stamp
            if count < 1:
                raise RuntimeError("preparation support does not fit its workspace")
            stamp = time.perf_counter()
            read_blocks_into(reader,zero,block,w.selected_leaf_ids[:selected],fields,
                             w.payload[:selected],widths)
            reader_seconds += time.perf_counter()-stamp
            stamp = time.perf_counter()
            halo_provider._apply_chunk_actions_unchecked(w,count,selected,widths,
                                                        interior_upper,modes,normals)
            ghost_seconds += time.perf_counter()-stamp
            stamp = time.perf_counter()
            output[order[first:first+count]] = np.moveaxis(w.payload[:count],1,-1)
            packing_seconds += time.perf_counter()-stamp
            first += count
            chunks += 1
            loads += selected
            maximum = max(maximum,selected)
        if validate_values is not None:
            validate_values()
        cache_stats = ({key:value-cache_before[key] for key,value in value_cache.stats.items()}
                       if value_cache is not None else {})
        return {"primary_count":len(ids),"chunk_count":chunks,"reader_call_count":chunks,
                "consumer_call_count":chunks,"selected_load_count":loads,
                "maximum_selected_slots":maximum,
                "managed_array_bytes":sum(a.nbytes for a in (*arrays,zero,widths,interior_upper,modes,normals)),
                **cache_stats,"total_seconds":time.perf_counter()-start,
                "planning_seconds":planning_seconds,"reader_seconds":reader_seconds,
                "ghost_seconds":ghost_seconds,"packing_seconds":packing_seconds,
                "read_value_bytes":cache_stats.get('value_cache_misses',loads)*len(fields)*8*int(np.prod(block)),
                "requested_value_bytes":loads*len(fields)*8*int(np.prod(block)),
                "scratch_admission_bytes": scratch_bytes(fields, halo)}

    def plan_builder(source, ids, capacity, budget):
        from .geometry_plans import _build
        return _build(source, ids, capacity, budget, root_shape, coord_to_rank, f)

    return FieldSource(mesh, tuple(definitions), fill, scratch_bytes, resident,
                       "rewrite-ratio2-minmod-exactphase-cont-v1",
                       memory_arrays=reader.memory_arrays,read_interiors=read_interiors,
                       original_field_ids=original_field_ids,validate_values=validate_values,
                       plan_builder=plan_builder)
