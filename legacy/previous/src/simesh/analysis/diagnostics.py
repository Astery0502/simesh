"""Independent derivative groups retained beside magnetic preparation."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import numpy as np

from .derivatives import curl
from .fields import PreparedFields, PreparedPool, FieldDefinition, _Borrow


@dataclass(frozen=True)
class WithCurl:
    primary: PreparedFields
    curl: PreparedFields


def with_curl(primary, *, budget_bytes=2*1024**3):
    """Retain an independently owned curl group for repeated coupled consumers."""
    if primary.halo < 2 or len(primary.fields) != 3:
        raise ValueError("coupled B/curl requires three B components and two primary halos")
    return WithCurl(primary,curl(primary,budget_bytes=budget_bytes))


class CurlPool:
    """One-halo curl companion for a borrowed two-halo magnetic pool.

    Close releases the companion, not the caller-owned primary pool. Both groups
    are frozen under the primary's coordinator lease during worker consumption.
    Source identity and numerical interpretation are fixed by that primary pool.
    """

    def __init__(self, primary, *, budget_bytes=2*1024**3):
        if not isinstance(primary,PreparedPool) or primary.halo < 2 or len(primary.field_ids)!=3:
            raise ValueError("CurlPool requires a three-component two-halo PreparedPool")
        self.primary = primary
        self.source, self.field_ids = primary.source, primary.field_ids
        self.capacity, self.halo = primary.capacity, primary.halo
        shape = (self.capacity,*(n+2 for n in self.source.mesh.block_shape),3)
        values_bytes = 8*int(np.prod(shape))
        self.controlled_bytes = primary.controlled_bytes+2*values_bytes+self.capacity*24
        if self.controlled_bytes > budget_bytes:
            raise MemoryError(f"B/curl pool needs {self.controlled_bytes} controlled bytes")
        self._values = np.empty(shape,dtype=float)
        self._keys = np.full(self.capacity,-1,dtype=np.int64)
        self._budget = budget_bytes
        self.derived_count = 0
        units = {self.source.fields[i].units for i in self.field_ids}
        if len(units)!=1:
            raise ValueError("magnetic components must share units")
        unit = next(iter(units))+" / coordinate-length"
        self._definitions = tuple(FieldDefinition("curl_"+axis,unit,"centered-derivative") for axis in "xyz")

    @property
    def resident_leaf_ids(self):
        return self.primary.resident_leaf_ids

    @contextmanager
    def borrow(self, ids, *, touch=True):
        if self._values is None:
            raise RuntimeError("curl companion is closed")
        with self.primary.borrow(ids,touch=touch) as primary:
            slots = primary.slot_of_leaf[primary.leaf_ids]
            missing = primary.leaf_ids[self._keys[slots] != primary.leaf_ids]
            if len(missing):
                subset = replace(primary._product,leaf_ids=missing)
                computed = curl(subset,budget_bytes=self._budget-self.source.resident_bytes-self._values.nbytes)
                destinations = primary.slot_of_leaf[missing]
                self._values[destinations] = computed.values
                self._keys[destinations] = missing
                self.derived_count += len(missing)
                del computed
            values = self._values.view()
            values.flags.writeable = False
            companion = PreparedFields(primary.mesh,values,primary.leaf_ids,primary.slot_of_leaf,
                self._definitions,1,primary.strategy+"/centered-extended",primary.source)
            # The containing context is the borrow scope for both groups.
            borrowed = _Borrow(companion)
            try:
                yield WithCurl(primary,borrowed)
            finally:
                borrowed._product = None

    def clear(self):
        if self._values is None:
            raise RuntimeError("curl companion is closed")
        self.primary.clear()
        self._keys.fill(-1)

    def close(self):
        if self.primary._active:
            raise RuntimeError("cannot close a borrowed curl companion")
        self._values = self._keys = None
        self.source = None
