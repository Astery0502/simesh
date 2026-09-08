"""Private bounded ordinary-interior reuse for an immutable provider lifecycle.

The adapter binds one ordered field selection. Changing that selection discards
all keys. Source-value changes require a new source/cache, never a geometry hit.
Only the provider's checked full-interior transfers use this adapter.
"""
import time
import numpy as np
from simesh._amr.blockio import make_block_reader, read_blocks_into


class InteriorValueCache:
    def __init__(self, reader, capacity, validate_values=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('interior cache capacity must be a positive integer')
        self.capacity = min(capacity, reader.shape[0])
        self.backing_reader = reader
        self.validate_values = validate_values
        self.values = np.empty((self.capacity, *reader.shape[1:]), dtype=float)
        self.directory = np.full(reader.shape[0], -1, dtype=np.int64)
        self.leaves = np.full(self.capacity, -1, dtype=np.int64)
        self.age = np.zeros(self.capacity, dtype=np.int64)
        self.fields = None
        self.clock = 0
        self.stats = {'value_cache_hits':0, 'value_cache_misses':0,
                      'value_cache_read_seconds':0., 'value_cache_copy_seconds':0.}
        self.memory_arrays = (self.values,self.directory,self.leaves,self.age)
        self.reader = make_block_reader(self, reader.shape, self._read,
            memory_arrays=(*reader.memory_arrays,*self.memory_arrays))

    def _read(self, state, lower, upper, ids, fields, output, out_lower):
        if not len(ids) or not len(fields):
            read_blocks_into(self.backing_reader,lower,upper,ids,fields,output,out_lower)
            return
        if self.validate_values is not None:
            self.validate_values()
        if np.any(lower) or tuple(upper)!=self.backing_reader.shape[2:]:
            raise ValueError('provider value cache requires complete interiors')
        key = tuple(map(int,fields))
        if key!=self.fields:
            self.directory.fill(-1)
            self.leaves.fill(-1)
            self.age.fill(0)
            self.fields = key
        slots = self.directory[ids]
        hit_rows = np.flatnonzero(slots>=0)
        miss_rows = np.flatnonzero(slots<0)
        k = len(fields)
        # The provider bounds each transfer by its admitted support capacity.
        temporary = np.empty((len(miss_rows),k,*self.backing_reader.shape[2:]))
        start = time.perf_counter()
        read_blocks_into(self.backing_reader,lower,upper,ids[miss_rows],fields,
                         temporary,np.zeros(3,dtype=np.int64))
        if self.validate_values is not None:
            self.validate_values()
        self.stats['value_cache_read_seconds'] += time.perf_counter()-start
        # Publish only after the complete miss read succeeds. On a failure both
        # hits and keys remain usable only under the source's validity contract.
        start = time.perf_counter()
        region = tuple(slice(int(a),int(a+b)) for a,b in zip(out_lower,upper))
        for row in hit_rows:
            output[(row,slice(None),*region)] = self.values[slots[row],:k]
        for pos,row in enumerate(miss_rows):
            output[(row,slice(None),*region)] = temporary[pos]
        self.clock += 1
        self.age[slots[hit_rows]] = self.clock
        # Hit values are already copied to the caller before any eviction.
        keep = min(len(miss_rows),self.capacity)
        if keep:
            destinations = np.argsort(self.age,kind='stable')[:keep]
            old = self.leaves[destinations]
            self.directory[old[old>=0]] = -1
            for slot,pos in zip(destinations,range(len(miss_rows)-keep,len(miss_rows))):
                leaf = ids[miss_rows[pos]]
                self.values[slot,:k] = temporary[pos]
                self.leaves[slot] = leaf
                self.directory[leaf] = slot
                self.age[slot] = self.clock
        self.stats['value_cache_copy_seconds'] += time.perf_counter()-start
        self.stats['value_cache_hits'] += len(hit_rows)
        self.stats['value_cache_misses'] += len(miss_rows)

    def close(self):
        self.values = self.directory = self.leaves = self.age = None
        self.reader = self.backing_reader = self.validate_values = None
        self.memory_arrays = ()


def cache_source(source, *, capacity, memory_limit=None):
    """Explicit raw-value cache borrowing an immutable source and its Mesh.

    Closing this adapter releases its cache, not the parent source. Parent close
    or mutation invalidates reads, including hits. A new ordered field request
    rebinds the cache and discards its previous keys.
    """
    from .source import Source
    from .._validation import admit
    source.validate()
    if type(capacity) is not int or capacity<1:
        raise ValueError("cache capacity must be a positive integer")
    count=min(capacity,source.mesh.leaf_count)
    block_bytes=8*int(np.prod(source.mesh.block_shape))
    cache_bytes=count*len(source.fields)*block_bytes+source.mesh.leaf_count*8+count*16
    admit(source.mesh.nbytes+source.nbytes+cache_bytes,memory_limit,"raw cache")
    cache=InteriorValueCache(source._reader,count,source.validate)
    cached=Source(source.mesh,source.fields,cache.reader,validate=source.validate,close=cache.close,
        read_scratch=lambda n,k:source.read_footprint(n,k)+n*k*block_bytes+64*(n+count),
        io_stats=cache.stats)
    cached.identity=source.identity
    return cached
