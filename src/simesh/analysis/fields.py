"""Independent field backing, explicit publication and bounded scoped borrowing."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .mesh import MeshIndex, indices, frozen_array


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    units: str
    interpretation: str = "cell-average"


@dataclass(frozen=True, eq=False)
class FieldSource:
    """Borrowed immutable source; ``fill(ids,fields,halo,out)`` completes out.

    out is component-adjacent and private until success. Reader failures may
    modify it, but cannot publish it. The source owner controls close; a caller
    must keep the source alive during preparation, not during detached queries.
    scratch_bytes bounds provider arrays/conversion for the admitted field set.
    resident_bytes counts source/provider backing separately from core storage.
    validate_values, when supplied, must raise if the source lifecycle changed
    or ended. Array owners otherwise promise immutability until pools close.
    """

    mesh: MeshIndex
    fields: tuple[FieldDefinition, ...]
    fill: Callable
    scratch_bytes: Callable
    resident_bytes: int
    strategy: str
    identity: object = field(default_factory=object)
    memory_arrays: tuple = ()
    read_interiors: Callable | None = None
    original_field_ids: tuple | None = None
    validate_values: Callable | None = None


@dataclass(frozen=True, eq=False)
class PreparedFields:
    mesh: MeshIndex
    values: np.ndarray
    leaf_ids: np.ndarray
    slot_of_leaf: np.ndarray
    fields: tuple[FieldDefinition, ...]
    halo: int
    strategy: str
    source: object
    preparation_stats: object = None
    owner: object = None
    owner_extra_bytes: int = 0

    @property
    def nbytes(self):
        # Whole-domain products intentionally share one IDs/directory array.
        directory_bytes = 0 if self.leaf_ids is self.slot_of_leaf else self.slot_of_leaf.nbytes
        return (self.values.nbytes + self.leaf_ids.nbytes + directory_bytes +
                self.owner_extra_bytes)

    def interior(self):
        if (len(self.values) != len(self.leaf_ids) or
                not np.array_equal(self.slot_of_leaf[self.leaf_ids], np.arange(len(self.leaf_ids)))):
            raise ValueError("nonpacked borrowed slots: use window(leaf, lower, upper) for direct views")
        h = self.halo
        return self.values[:, h:h+self.mesh.block_shape[0],
                           h:h+self.mesh.block_shape[1],
                           h:h+self.mesh.block_shape[2], :]

    def window(self, leaf, lower, upper, *, support=0):
        """Stable view backed by this owned product; no new preparation."""
        leaf = int(indices([leaf], self.mesh.leaf_count)[0])
        slot = self.slot_of_leaf[leaf]
        lower, upper = np.asarray(lower), np.asarray(upper)
        if (lower.shape != (3,) or upper.shape != (3,) or
                lower.dtype.kind not in "iu" or upper.dtype.kind not in "iu" or
                np.any(lower < 0) or np.any(upper > self.mesh.block_shape) or
                np.any(lower > upper) or not isinstance(support, int) or
                not 0 <= support <= self.halo or slot < 0):
            raise ValueError("window or support is outside prepared coverage")
        slices = tuple(slice(int(a)+self.halo-support, int(b)+self.halo+support)
                       for a, b in zip(lower, upper))
        return self.values[(slot, *slices, slice(None))]


def _request(source, leaf_ids, field_ids, halo):
    ids = indices(leaf_ids, source.mesh.leaf_count)
    fields = indices(field_ids, len(source.fields), "field_ids")
    if not fields.size:
        raise ValueError("at least one field is required")
    if type(halo) is not int or halo not in (0,2):
        raise ValueError("primary fields support interior-only (0) or two valid halo layers")
    return ids, fields


def _footprint(source, count, fields, halo):
    shape = (count, *(n + 2*halo for n in source.mesh.block_shape), len(fields))
    output = 8 * int(np.prod(shape)) + 8 * (source.mesh.leaf_count + count)
    total = output + source.mesh.nbytes + source.resident_bytes + source.scratch_bytes(fields, halo)
    return shape, total


def prepare(source, leaf_ids, field_ids, *, halo=2, budget_bytes=2*1024**3):
    """Detach a complete native field product, with complete controlled admission."""
    ids, fields = _request(source, leaf_ids, field_ids, halo)
    if source.validate_values is not None:
        source.validate_values()
    shape, total = _footprint(source, len(ids), fields, halo)
    if total > budget_bytes:
        raise MemoryError(f"preparation needs {total} controlled bytes, budget {budget_bytes}")
    values = np.empty(shape, dtype=np.float64)
    stats = source.fill(ids, fields, halo, values)
    directory = np.full(source.mesh.leaf_count, -1, dtype=np.int64)
    directory[ids] = np.arange(len(ids))
    values.flags.writeable = False
    directory.flags.writeable = False
    return PreparedFields(source.mesh, values, frozen_array(ids, np.int64), directory,
                          tuple(source.fields[i] for i in fields), halo, source.strategy,
                          source.identity, stats)


class _Borrow:
    def __init__(self, product):
        self._product = product

    def __getattr__(self, name):
        if self._product is None:
            raise RuntimeError("prepared borrow has expired")
        return getattr(self._product, name)


class PreparedPool:
    """Coordinator-only cache. Escaped arrays expire when their borrow ends.

    An active borrow freezes the entire pool. Workers share that one borrow;
    additional borrows, preparation, clear and close are rejected until return.
    A miss failure invalidates all overwritten candidate slots before publication.
    """

    def __init__(self, source, field_ids, capacity, *, halo=2, budget_bytes=2*1024**3):
        _, fields = _request(source, np.empty(0, dtype=np.int64), field_ids, halo)
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        capacity = min(capacity, source.mesh.leaf_count)
        shape, total = _footprint(source, capacity, fields, halo)
        # Misses write contiguous final slots without another padded copy.
        # Selector/recency storage is included conservatively.
        total += 64 * (capacity + source.mesh.leaf_count)
        if total > budget_bytes:
            raise MemoryError(f"pool needs {total} controlled bytes, budget {budget_bytes}")
        self.source, self.field_ids, self.halo = source, frozen_array(fields,np.int64), halo
        self.capacity, self.controlled_bytes = capacity, total
        self._values = np.empty(shape, dtype=np.float64)
        self._directory = np.full(source.mesh.leaf_count, -1, dtype=np.int64)
        self._leaves = np.full(capacity, -1, dtype=np.int64)
        self._age = np.zeros(capacity, dtype=np.int64)
        self._clock = self._active = 0
        self._closed = False
        self.prepared_count = 0

    def _ensure(self, ids, *, touch=True):
        if self._closed:
            raise RuntimeError("pool is closed")
        if self.source.validate_values is not None:
            self.source.validate_values()
        missing = ids[self._directory[ids] < 0]
        if len(ids) > self.capacity:
            raise MemoryError("requested simultaneous borrow exceeds pool capacity")
        if missing.size and self._active:
            raise RuntimeError("cannot prepare while a borrow is active")
        if missing.size:
            protected = self._directory[ids]
            candidates = np.argsort(self._age, kind="stable")
            slots = candidates[~np.isin(candidates, protected[protected >= 0])][:len(missing)]
            # Group contiguous slots for one provider call per run; source order
            # and storage order are bound explicitly, not assumed equal.
            slots.sort()
            old = self._leaves[slots]
            self._directory[old[old >= 0]] = -1
            self._leaves[slots] = -1
            starts = np.r_[0, np.flatnonzero(np.diff(slots) != 1)+1, len(slots)]
            for first, last in zip(starts[:-1], starts[1:]):
                run = slots[first:last]
                self.source.fill(missing[first:last], self.field_ids, self.halo,
                                 self._values[run[0]:run[-1]+1])
            self._leaves[slots] = missing
            self._directory[missing] = slots
            self.prepared_count += len(missing)
        self._clock += 1
        accessed = ids if touch else missing
        self._age[self._directory[accessed]] = self._clock

    @contextmanager
    def borrow(self, leaf_ids, *, touch=True):
        """Freeze requested coverage; `touch=False` does not claim cache hits.

        Consumers exposing the complete directory use a non-touching lease.
        Successful newly prepared entries always receive a fresh timestamp.
        """
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._active:
            raise RuntimeError("share the active borrow; nested borrows are not admitted")
        if type(touch) is not bool:
            raise ValueError("touch must be a boolean")
        ids = indices(leaf_ids, self.source.mesh.leaf_count)
        self._ensure(ids,touch=touch)
        self._active += 1
        values = self._values.view()
        values.flags.writeable = False
        # A lease grants only its requested coverage, even if the pool has more.
        directory = np.full(self.source.mesh.leaf_count, -1, dtype=np.int64)
        directory[ids] = self._directory[ids]
        directory.flags.writeable = False
        product = PreparedFields(self.source.mesh, values, frozen_array(ids, np.int64),
                                 directory, tuple(self.source.fields[i] for i in self.field_ids),
                                 self.halo, self.source.strategy, self.source.identity)
        lease = _Borrow(product)
        try:
            yield lease
        finally:
            lease._product = None
            self._active -= 1

    def clear(self):
        if self._closed:
            raise RuntimeError("pool is closed")
        if self._active:
            raise RuntimeError("cannot clear a borrowed pool")
        self._directory.fill(-1)
        self._leaves.fill(-1)
        self._age.fill(0)

    @property
    def resident_leaf_ids(self):
        if self._closed:
            raise RuntimeError("pool is closed")
        return self._leaves[self._leaves >= 0].copy()

    def close(self):
        if self._active:
            raise RuntimeError("cannot close a borrowed pool")
        self._closed = True
        self._values = self._directory = self._leaves = self._age = None
        self.source = None


def iter_prepared(source, leaf_ids, field_ids, *, capacity=128, budget_bytes=2*1024**3):
    """Yield bounded native batches; each borrow expires when iteration advances.

    Consumers can synchronously write requested interiors to an array/memmap
    sink. Retaining all yielded arrays is outside this borrowing contract.
    """
    ids = indices(leaf_ids, source.mesh.leaf_count)
    pool = PreparedPool(source, field_ids, capacity, budget_bytes=budget_bytes)
    try:
        for first in range(0, len(ids), pool.capacity):
            with pool.borrow(ids[first:first+pool.capacity]) as batch:
                yield batch
    finally:
        pool.close()
