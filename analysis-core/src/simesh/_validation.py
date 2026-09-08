"""Small boundary checks shared by the independent data interfaces."""

import numpy as np


def frozen_array(value, dtype):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def indices(value, size, name="leaf_ids"):
    array = np.asarray(value)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be one-dimensional integer indices")
    if np.any(array < 0) or np.any(array >= size):
        raise ValueError(f"{name} outside available range")
    result = np.ascontiguousarray(array, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must not contain duplicates")
    return result


def workers_count(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    return workers


def admit(required, memory_limit, operation):
    if memory_limit is not None:
        if type(memory_limit) is not int or memory_limit < 1:
            raise ValueError("memory_limit must be a positive integer or None")
        if required > memory_limit:
            raise MemoryError(f"{operation} needs at most {required} controlled bytes; limit {memory_limit}")


def array_bytes(arrays):
    """Count shared NumPy backing once within one explicitly supplied group."""
    seen = set()
    total = 0
    for array in arrays:
        if not isinstance(array, np.ndarray):
            continue
        base = array
        while isinstance(base.base, np.ndarray):
            base = base.base
        if id(base) not in seen:
            seen.add(id(base))
            total += base.nbytes
    return total


def remaining(memory_limit, reserved, operation="retained outputs"):
    if memory_limit is None:
        return None
    admit(reserved+1, memory_limit, operation)
    return memory_limit-reserved
