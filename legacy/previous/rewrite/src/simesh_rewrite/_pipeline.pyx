"""Allocation-free semantic preflight for the M0 pipeline."""

import cython

from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int64_t validate_level1_mor_top_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[:, ::1] rank_to_coord,
    const int64_t[:, ::1] face_neighbor_ids,
):
    cdef int64_t block_id, axis, coordinate
    cdef int64_t x, y, z, expected
    for block_id in range(rank_to_coord.shape[0]):
        x = rank_to_coord[block_id, 0]
        y = rank_to_coord[block_id, 1]
        z = rank_to_coord[block_id, 2]
        if (
            x < 0 or x >= root_shape[0]
            or y < 0 or y >= root_shape[1]
            or z < 0 or z >= root_shape[2]
            or coord_to_rank[x, y, z] != block_id
        ):
            return block_id
        for axis in range(3):
            coordinate = rank_to_coord[block_id, axis]
            if coordinate == 0:
                expected = -1
            elif axis == 0:
                expected = coord_to_rank[x - 1, y, z]
            elif axis == 1:
                expected = coord_to_rank[x, y - 1, z]
            else:
                expected = coord_to_rank[x, y, z - 1]
            if face_neighbor_ids[block_id, 2 * axis] != expected:
                return block_id

            if coordinate + 1 == root_shape[axis]:
                expected = -1
            elif axis == 0:
                expected = coord_to_rank[x + 1, y, z]
            elif axis == 1:
                expected = coord_to_rank[x, y + 1, z]
            else:
                expected = coord_to_rank[x, y, z + 1]
            if face_neighbor_ids[block_id, 2 * axis + 1] != expected:
                return block_id
    return -1
