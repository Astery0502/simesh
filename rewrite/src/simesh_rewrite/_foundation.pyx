"""Typed implementations of the FND-001 semantic primitives."""

from libc.stdint cimport int64_t
from libc.stddef cimport size_t
from libc.string cimport memcpy


cpdef int64_t ravel_cell_unchecked(
    const int64_t[::1] index,
    const int64_t[::1] shape,
):
    return (index[0] * shape[1] + index[1]) * shape[2] + index[2]


cpdef tuple unravel_cell_unchecked(
    int64_t offset,
    const int64_t[::1] shape,
):
    cdef int64_t plane = shape[1] * shape[2]
    cdef int64_t i = offset // plane
    cdef int64_t remainder = offset - i * plane
    cdef int64_t j = remainder // shape[2]
    cdef int64_t k = remainder - j * shape[2]
    return i, j, k


cpdef void copy_region_into_unchecked(
    const double[:, :, :, :, ::1] source,
    const int64_t[::1] source_lower,
    double[:, :, :, :, ::1] destination,
    const int64_t[::1] destination_lower,
    const int64_t[::1] extent,
):
    cdef Py_ssize_t slot, field, i, j
    cdef Py_ssize_t source_i, source_j
    cdef Py_ssize_t destination_i, destination_j
    cdef size_t row_bytes

    if extent[0] == 0 or extent[1] == 0 or extent[2] == 0:
        return

    row_bytes = <size_t>extent[2] * sizeof(double)

    for slot in range(source.shape[0]):
        for field in range(source.shape[1]):
            for i in range(extent[0]):
                source_i = source_lower[0] + i
                destination_i = destination_lower[0] + i
                for j in range(extent[1]):
                    source_j = source_lower[1] + j
                    destination_j = destination_lower[1] + j
                    memcpy(
                        &destination[
                            slot,
                            field,
                            destination_i,
                            destination_j,
                            destination_lower[2],
                        ],
                        &source[
                            slot,
                            field,
                            source_i,
                            source_j,
                            source_lower[2],
                        ],
                        row_bytes,
                    )
