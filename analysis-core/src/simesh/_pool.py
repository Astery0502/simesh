"""Explicit coordinator-owned prepared slots; direct kernels know only Fields."""

from contextlib import contextmanager
import numpy as np

from .fields import Fields, _Lease, FieldDefinition, source_value_identity
from .mesh import Selection
from ._validation import frozen_array, indices, admit
from .preparation import exact


class PreparedPool:
    """Opt-in exact-phase cache with stable synchronous leases.

    The source is borrowed and must remain open/immutable. Closing the pool
    releases its arrays and workspace without closing that source.
    """
    def __init__(self, source, fields=None, *, capacity=128, scheme,
                 support_capacity=128, memory_limit=None):
        source.validate()
        if scheme != "exact-phase":
            raise ValueError("prepared pools use explicit exact-phase preparation")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.field_ids = frozen_array(source.field_ids(fields),np.int64)
        self.fields = tuple(source.fields[i] for i in self.field_ids)
        self.source, self.mesh = source, source.mesh
        self.capacity = min(capacity,self.mesh.leaf_count)
        self.storage_halo = self.valid_halo = 2
        self.scheme = exact.SCHEME
        self.value_identity = source_value_identity(source.identity,self.field_ids,self.scheme)
        block,support,scratch = exact.parameters(self.mesh,len(self.fields),support_capacity)
        shape = (self.capacity,*(block+4),len(self.fields))
        self.controlled_bytes = (self.mesh.nbytes+source.nbytes+source.read_footprint(support,len(self.fields))+scratch+
                                8*int(np.prod(shape))+64*(self.capacity+self.mesh.leaf_count))
        admit(self.controlled_bytes,memory_limit,"prepared pool")
        self._values = np.empty(shape)
        self._workspace = exact.Workspace.allocate(self.mesh,len(self.fields),block,support)
        self._directory = np.full(self.mesh.leaf_count,-1,np.int64)
        self._leaves = np.full(self.capacity,-1,np.int64)
        self._age = np.zeros(self.capacity,np.int64)
        self._clock = 0
        self._active = self._closed = False
        self.prepared_count = 0
        self.read_value_bytes = 0
        self.fill_seconds = 0.

    def _ensure(self, ids, touch):
        self.source.validate()
        if len(ids)>self.capacity:
            raise MemoryError("simultaneous coverage exceeds pool capacity")
        missing = ids[self._directory[ids]<0]
        if len(missing):
            protected = self._directory[ids]
            candidates = np.argsort(self._age,kind="stable")
            slots = candidates[~np.isin(candidates,protected[protected>=0])][:len(missing)]
            slots.sort()
            old = self._leaves[slots]
            self._directory[old[old>=0]] = -1
            self._leaves[slots] = -1
            starts = np.r_[0,np.flatnonzero(np.diff(slots)!=1)+1,len(slots)]
            for first,last in zip(starts[:-1],starts[1:]):
                run = slots[first:last]
                selection = Selection(self.mesh,missing[first:last])
                stats = exact.fill(self.source,selection,self.field_ids,
                                   self._values[run[0]:run[-1]+1],self._workspace)
                self.read_value_bytes += stats["read_value_bytes"]
                self.fill_seconds += stats["total_seconds"]
            self._leaves[slots] = missing
            self._directory[missing] = slots
            self.prepared_count += len(missing)
        self._clock += 1
        touched = ids if touch else missing
        self._age[self._directory[touched]] = self._clock

    @contextmanager
    def borrow(self, leaf_ids, *, touch=True):
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._active:
            raise RuntimeError("share the current lease; nested borrows are not allowed")
        if type(touch) is not bool:
            raise ValueError("touch must be boolean")
        ids = indices(leaf_ids,self.mesh.leaf_count)
        self._ensure(ids,touch)
        self._active = True
        lease = _Lease()
        view = self._values.view()
        directory = np.full(self.mesh.leaf_count,-1,np.int64)
        directory[ids] = self._directory[ids]
        try:
            yield Fields(self.mesh,view,Selection(self.mesh,ids),directory,self.fields,2,2,
                         self.scheme,self.source.identity,{},lease,self.value_identity)
        finally:
            lease.active = False
            self._active = False

    @property
    def resident_leaf_ids(self):
        if self._closed:
            raise RuntimeError("pool is closed")
        return self._leaves[self._leaves>=0].copy()

    def clear(self):
        if self._closed or self._active:
            raise RuntimeError("cannot clear a closed or borrowed pool")
        self._directory.fill(-1)
        self._leaves.fill(-1)
        self._age.fill(0)

    def close(self):
        if self._active:
            raise RuntimeError("cannot close a borrowed pool")
        self._closed = True
        self._workspace = self._values = self._directory = self._leaves = self._age = None
        self.source = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("pool is closed")
        self.source.validate()
        return self

    def __exit__(self,*exc):
        self.close()


class CurlPool:
    """An explicit independent curl cache sharing its primary pool's lease."""
    def __init__(self, primary, *, memory_limit=None):
        if not isinstance(primary,PreparedPool) or primary._closed or len(primary.fields)!=3:
            raise ValueError("CurlPool needs an open three-component primary pool")
        if len({f.units for f in primary.fields})!=1:
            raise ValueError("curl vector components require common units")
        self.primary = primary
        shape = (primary.capacity,*(n+2 for n in primary.mesh.block_shape),3)
        size = 8*int(np.prod(shape))
        self.controlled_bytes = primary.controlled_bytes+2*size+primary.mesh.leaf_count*16
        admit(self.controlled_bytes,memory_limit,"B/curl pool")
        self._values = np.empty(shape)
        self._keys = np.full(primary.capacity,-1,np.int64)
        self._limit = memory_limit
        self.derived_count = 0
        self._definitions = tuple(FieldDefinition("curl_"+axis,primary.fields[0].units+" / coordinate-length",
                                                  "centered-derivative") for axis in "xyz")

    @contextmanager
    def borrow(self, ids, *, touch=True):
        if self._values is None:
            raise RuntimeError("curl pool is closed")
        from .operators.derivatives import curl
        from ._validation import remaining
        from dataclasses import replace
        with self.primary.borrow(ids,touch=touch) as primary:
            slots = primary.slot_of_leaf[primary.leaf_ids]
            missing = primary.leaf_ids[self._keys[slots]!=primary.leaf_ids]
            if len(missing):
                directory = np.full(primary.mesh.leaf_count,-1,np.int64)
                directory[missing] = primary.slot_of_leaf[missing]
                subset = replace(primary,selection=Selection(primary.mesh,missing),slot_of_leaf=directory)
                derived = curl(subset,memory_limit=remaining(self._limit,
                    self.primary.source.nbytes+self._values.nbytes))
                destinations = primary.slot_of_leaf[missing]
                self._values[destinations] = derived.values
                self._keys[destinations] = missing
                self.derived_count += len(missing)
            values = self._values.view()
            companion = Fields(primary.mesh,values,primary.selection,primary.slot_of_leaf,
                self._definitions,1,1,primary.scheme+"/centered-extended",primary.source,
                {},primary._lease,object(),("curl",primary.value_identity,tuple(primary.fields),(0,1,2),primary.scheme))
            yield primary,companion

    def close(self):
        if self.primary._active:
            raise RuntimeError("cannot close a borrowed curl pool")
        self._values = self._keys = None

    def clear(self):
        if self._values is None:
            raise RuntimeError("curl pool is closed")
        self.primary.clear()
        self._keys.fill(-1)

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
