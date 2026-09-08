"""Sequential record batches for owned, full-domain analysis input arrays."""

import math
import time
import numpy as np

from simesh.io._v5.reader import _current_identity, _pread_exact
from simesh._validation import indices


READ_BATCH_BYTES = 16 * 1024**2


def _copy_records(raw, offset, count, stride, ghosts, state, fields, output):
    """Decode a run with one stored shape, preserving every floating-point bit."""
    lower = tuple(int(v) for v in ghosts[:3])
    upper = tuple(int(v) for v in ghosts[3:])
    if any(v < 0 for v in (*lower, *upper)):
        raise ValueError("block record has negative saved ghost extents")
    shape = tuple(n+a+b for n,a,b in zip(state.shape[2:],lower,upper))
    cells = math.prod(shape)
    ordinary_bytes = state.shape[1]*cells*8
    tail_bytes = 24*math.prod(n+1 for n in shape) if state.staggered else 0
    if 24+ordinary_bytes+tail_bytes != stride:
        raise ValueError("block record length disagrees with its fields, ghosts or staggered tail")
    nx,ny,nz = shape
    stored = np.ndarray((count,state.shape[1],nz,ny,nx),
                        dtype=np.dtype(state.byte_order+'u8'),buffer=raw,offset=offset+24,
                        strides=(stride,cells*8,nx*ny*8,nx*8,8))
    x,y,z = (slice(lo,lo+n) for lo,n in zip(lower,state.shape[2:]))
    target = output.view(np.uint64)
    if len(fields) == 1 or np.all(np.diff(fields) == 1):
        selected = stored[:,int(fields[0]):int(fields[-1])+1,z,y,x]
        target[...] = selected.transpose(0,1,4,3,2)
    else:
        for column, field in enumerate(fields):
            target[:,column] = stored[:,int(field),z,y,x].transpose(0,3,2,1)


def read_full_interiors(reader, field_ids, *, chunk_bytes=READ_BATCH_BYTES):
    """Read contiguous file spans, returning owned native field-major interiors.

    The input is the checked ordinary-field file reader used by analysis. Full
    records, including unselected fields and staggered tails, are read to reduce
    small I/O operations. Only selected ordinary interiors enter the result.
    Record headers and lengths are validated before decoding each private batch;
    failure returns no result. The source identity is checked before and after.
    """
    state = reader.state
    fields = indices(field_ids,state.shape[1],"field_ids")
    if not len(fields) or type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("select nonempty fields and a positive read batch size")
    if _current_identity(state.file_descriptor,callback_lifecycle=True) != state.file_identity:
        raise OSError("analysis source changed before bulk reading")
    offsets = np.r_[state.block_offsets,state.file_identity[2]]
    count = state.shape[0]
    output = np.empty((count,len(fields),*state.shape[2:]),dtype=np.float64)
    first = calls = transferred = peak = 0
    read_seconds = decode_seconds = 0.
    started = time.perf_counter()
    while first < count:
        # End at a record boundary. A single unusually large record is allowed
        # and included in the caller's peak-buffer estimate.
        end_byte = min(int(offsets[-1]),int(offsets[first])+chunk_bytes)
        stop = int(np.searchsorted(offsets,end_byte,side='right'))-1
        stop = min(count,max(first+1,stop))
        sizes = np.diff(offsets[first:stop+1])
        if np.any(sizes < 24):
            raise ValueError("block record is shorter than its ghost header")
        size = int(offsets[stop]-offsets[first])
        stamp = time.perf_counter()
        raw = _pread_exact(state.file_descriptor,size,int(offsets[first]),section="record batch")
        read_seconds += time.perf_counter()-stamp
        stamp = time.perf_counter()
        stride = int(sizes[0])
        if np.all(sizes == stride):
            headers = np.ndarray((stop-first,6),dtype=np.dtype(state.byte_order+'i4'),
                                 buffer=raw,strides=(stride,4))
            uniform = bool(np.all(headers == headers[0]))
            if uniform:
                _copy_records(raw,0,stop-first,stride,headers[0],state,fields,output[first:stop])
            del headers
        else:
            uniform = False
        if not uniform:
            for row, record_size in enumerate(sizes):
                offset = int(offsets[first+row]-offsets[first])
                header = np.frombuffer(raw,dtype=np.dtype(state.byte_order+'i4'),count=6,offset=offset)
                _copy_records(raw,offset,1,int(record_size),header,state,fields,output[first+row:first+row+1])
                del header
        decode_seconds += time.perf_counter()-stamp
        calls += 1
        transferred += size
        peak = max(peak,size)
        first = stop
        del raw
    if _current_identity(state.file_descriptor,callback_lifecycle=True) != state.file_identity:
        raise OSError("analysis source changed during bulk reading")
    return output, {"record_batch_count":calls,"file_bytes_requested":transferred,
                    "selected_value_bytes":output.nbytes,"peak_read_buffer_bytes":peak,
                    "read_seconds":read_seconds,"decode_seconds":decode_seconds,
                    "total_seconds":time.perf_counter()-started}
