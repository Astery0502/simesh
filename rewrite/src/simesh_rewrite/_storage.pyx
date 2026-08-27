"""Exact selected in-memory transfers for STO-001."""

import cython

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libc.string cimport memcpy


cpdef int64_t validate_indices_unchecked(
    const int64_t[::1] indices,
    int64_t limit,
):
    cdef int64_t position
    for position in range(indices.shape[0]):
        if indices[position] < 0 or indices[position] >= limit:
            return position
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void gather_blocks_into_unchecked(
    const double[:, :, :, :, ::1] backing,
    const int64_t[::1] source_lower,
    const int64_t[::1] extent,
    const int64_t[::1] block_ids,
    const int64_t[::1] field_ids,
    double[:, :, :, :, ::1] destination,
    const int64_t[::1] destination_lower,
):
    cdef int64_t slot, field_slot, i, j
    cdef int64_t block_id, field_id
    cdef size_t row_bytes, plane_bytes, slab_bytes
    if (
        block_ids.shape[0] == 0
        or field_ids.shape[0] == 0
        or extent[0] == 0
        or extent[1] == 0
        or extent[2] == 0
    ):
        return
    if (
        source_lower[1] == 0
        and source_lower[2] == 0
        and destination_lower[1] == 0
        and destination_lower[2] == 0
        and extent[1] == backing.shape[3]
        and extent[1] == destination.shape[3]
        and extent[2] == backing.shape[4]
        and extent[2] == destination.shape[4]
    ):
        slab_bytes = <size_t>(extent[0] * extent[1] * extent[2]) * sizeof(double)
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            for field_slot in range(field_ids.shape[0]):
                field_id = field_ids[field_slot]
                memcpy(
                    &destination[slot, field_slot, destination_lower[0], 0, 0],
                    &backing[block_id, field_id, source_lower[0], 0, 0],
                    slab_bytes,
                )
        return

    if (
        source_lower[2] == 0
        and destination_lower[2] == 0
        and extent[2] == backing.shape[4]
        and extent[2] == destination.shape[4]
    ):
        plane_bytes = <size_t>(extent[1] * extent[2]) * sizeof(double)
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            for field_slot in range(field_ids.shape[0]):
                field_id = field_ids[field_slot]
                for i in range(extent[0]):
                    memcpy(
                        &destination[
                            slot,
                            field_slot,
                            destination_lower[0] + i,
                            destination_lower[1],
                            0,
                        ],
                        &backing[
                            block_id,
                            field_id,
                            source_lower[0] + i,
                            source_lower[1],
                            0,
                        ],
                        plane_bytes,
                    )
        return

    row_bytes = <size_t>extent[2] * sizeof(double)
    for slot in range(block_ids.shape[0]):
        block_id = block_ids[slot]
        for field_slot in range(field_ids.shape[0]):
            field_id = field_ids[field_slot]
            for i in range(extent[0]):
                for j in range(extent[1]):
                    memcpy(
                        &destination[
                            slot,
                            field_slot,
                            destination_lower[0] + i,
                            destination_lower[1] + j,
                            destination_lower[2],
                        ],
                        &backing[
                            block_id,
                            field_id,
                            source_lower[0] + i,
                            source_lower[1] + j,
                            source_lower[2],
                        ],
                        row_bytes,
                    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void scatter_blocks_from_unchecked(
    const double[:, :, :, :, ::1] source,
    const int64_t[::1] source_lower,
    const int64_t[::1] extent,
    const int64_t[::1] block_ids,
    const int64_t[::1] field_ids,
    double[:, :, :, :, ::1] backing,
    const int64_t[::1] destination_lower,
):
    cdef int64_t slot, field_slot, i, j
    cdef int64_t block_id, field_id
    cdef size_t row_bytes, plane_bytes, slab_bytes
    if (
        block_ids.shape[0] == 0
        or field_ids.shape[0] == 0
        or extent[0] == 0
        or extent[1] == 0
        or extent[2] == 0
    ):
        return
    if (
        source_lower[1] == 0
        and source_lower[2] == 0
        and destination_lower[1] == 0
        and destination_lower[2] == 0
        and extent[1] == source.shape[3]
        and extent[1] == backing.shape[3]
        and extent[2] == source.shape[4]
        and extent[2] == backing.shape[4]
    ):
        slab_bytes = <size_t>(extent[0] * extent[1] * extent[2]) * sizeof(double)
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            for field_slot in range(field_ids.shape[0]):
                field_id = field_ids[field_slot]
                memcpy(
                    &backing[block_id, field_id, destination_lower[0], 0, 0],
                    &source[slot, field_slot, source_lower[0], 0, 0],
                    slab_bytes,
                )
        return

    if (
        source_lower[2] == 0
        and destination_lower[2] == 0
        and extent[2] == source.shape[4]
        and extent[2] == backing.shape[4]
    ):
        plane_bytes = <size_t>(extent[1] * extent[2]) * sizeof(double)
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            for field_slot in range(field_ids.shape[0]):
                field_id = field_ids[field_slot]
                for i in range(extent[0]):
                    memcpy(
                        &backing[
                            block_id,
                            field_id,
                            destination_lower[0] + i,
                            destination_lower[1],
                            0,
                        ],
                        &source[
                            slot,
                            field_slot,
                            source_lower[0] + i,
                            source_lower[1],
                            0,
                        ],
                        plane_bytes,
                    )
        return

    row_bytes = <size_t>extent[2] * sizeof(double)
    for slot in range(block_ids.shape[0]):
        block_id = block_ids[slot]
        for field_slot in range(field_ids.shape[0]):
            field_id = field_ids[field_slot]
            for i in range(extent[0]):
                for j in range(extent[1]):
                    memcpy(
                        &backing[
                            block_id,
                            field_id,
                            destination_lower[0] + i,
                            destination_lower[1] + j,
                            destination_lower[2],
                        ],
                        &source[
                            slot,
                            field_slot,
                            source_lower[0] + i,
                            source_lower[1] + j,
                            source_lower[2],
                        ],
                        row_bytes,
                    )
