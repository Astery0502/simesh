"""Whole-domain native derivative delivery, independent of later slice geometry."""

import time
import numpy as np

from .derivatives import curl
from .fields import PreparedFields, prepare, _footprint, _request


def global_curl(source, *, field_ids=(0,1,2), batch_size=128, output=None,
                budget_bytes=2*1024**3):
    """Compute all leaf interiors plus one derived halo into owned or supplied backing.

    A supplied array/memmap is borrowed as output for its caller-owned lifetime.
    Failure raises without a returned product; completed sink batches may remain.
    The source is not kept alive by the successfully returned derived product.
    Ordered batches have no cache replacement or borrowing state.
    """
    mesh = source.mesh
    if len(field_ids) != 3:
        raise ValueError("global curl needs exactly three ordered vector components")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    shape = (mesh.leaf_count, *(n+2 for n in mesh.block_shape), 3)
    output_bytes = 8*int(np.prod(shape))
    batch_bytes = 8*min(batch_size,mesh.leaf_count)*int(np.prod(shape[1:]))
    directory_bytes = mesh.leaf_count*8
    reserve = output_bytes + batch_bytes + 4*directory_bytes
    if output is not None:
        if (not isinstance(output,np.ndarray) or output.dtype != np.float64 or
                output.shape != shape or not output.flags.c_contiguous or
                not output.flags.writeable):
            raise ValueError(f"output must be writable contiguous float64 {shape}")
        if any(np.shares_memory(output,a) for a in source.memory_arrays):
            raise ValueError("output must not alias immutable source backing")
    _, fields = _request(source,np.empty(0,dtype=np.int64),field_ids,2)
    _, preparation_bytes = _footprint(source,min(batch_size,mesh.leaf_count),fields,2)
    peak_controlled = preparation_bytes + reserve
    if peak_controlled > budget_bytes:
        raise MemoryError(f"global curl needs {peak_controlled} controlled bytes, budget {budget_bytes}")
    target = np.empty(shape,dtype=float) if output is None else output
    ids = np.arange(mesh.leaf_count,dtype=np.int64)
    start = time.perf_counter()
    derivative_seconds = output_seconds = 0.
    definitions = None
    for first in range(0,len(ids),batch_size):
        selected = ids[first:first+batch_size]
        primary = prepare(source,selected,fields,budget_bytes=budget_bytes-reserve)
        stamp = time.perf_counter()
        derived = curl(primary,budget_bytes=budget_bytes-output_bytes-source.resident_bytes)
        derivative_seconds += time.perf_counter()-stamp
        stamp = time.perf_counter()
        target[first:first+len(selected)] = derived.values
        output_seconds += time.perf_counter()-stamp
        definitions = derived.fields
        del derived, primary
    values = target.view()
    values.flags.writeable = False
    ids.flags.writeable = False
    stats = {"global_leaves":len(ids),"total_seconds":time.perf_counter()-start,
             "derivative_seconds":derivative_seconds,"output_seconds":output_seconds,
             "controlled_upper_bytes":peak_controlled,"output_bytes":output_bytes,
             "backing":"caller-array" if output is not None else "owned-array"}
    return PreparedFields(mesh,values,ids,ids,definitions,1,
                          source.strategy+"/centered-extended",source.identity,stats,owner=target)
