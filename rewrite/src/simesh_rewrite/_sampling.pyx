"""Exact native-grid placement for SAM-001."""

import cython

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libc.string cimport memcpy


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int64_t validate_selected_level1_placement_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[:, ::1] rank_to_coord,
    const int64_t[::1] block_ids,
):
    cdef int64_t slot, block_id, axis, coordinate
    for slot in range(block_ids.shape[0]):
        block_id = block_ids[slot]
        if block_id < 0 or block_id >= rank_to_coord.shape[0]:
            return slot
        for axis in range(3):
            coordinate = rank_to_coord[block_id, axis]
            if coordinate < 0 or coordinate >= root_shape[axis]:
                return slot
        if coord_to_rank[
            rank_to_coord[block_id, 0],
            rank_to_coord[block_id, 1],
            rank_to_coord[block_id, 2],
        ] != block_id:
            return slot
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void place_level1_blocks_unchecked(
    const double[:, :, :, :, ::1] payload,
    const int64_t[::1] payload_valid_lower,
    const int64_t[::1] block_ids,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, ::1] rank_to_coord,
    double[:, :, :, ::1] uniform_grid,
):
    cdef int64_t slot, field, i, j, block_id
    cdef int64_t global_lower[3]
    cdef size_t row_bytes = <size_t>block_cell_counts[2] * sizeof(double)

    with nogil:
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            global_lower[0] = (
                rank_to_coord[block_id, 0] * block_cell_counts[0]
            )
            global_lower[1] = (
                rank_to_coord[block_id, 1] * block_cell_counts[1]
            )
            global_lower[2] = (
                rank_to_coord[block_id, 2] * block_cell_counts[2]
            )
            for field in range(payload.shape[1]):
                for i in range(block_cell_counts[0]):
                    for j in range(block_cell_counts[1]):
                        memcpy(
                            &uniform_grid[
                                field,
                                global_lower[0] + i,
                                global_lower[1] + j,
                                global_lower[2],
                            ],
                            &payload[
                                slot,
                                field,
                                payload_valid_lower[0] + i,
                                payload_valid_lower[1] + j,
                                payload_valid_lower[2],
                            ],
                            row_bytes,
                        )
