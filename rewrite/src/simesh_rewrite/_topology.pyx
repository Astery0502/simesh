"""Typed MOR-map validation and level-1 face topology for TOP-001."""

import cython

from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_t _validate_level1_maps(
    int64_t extent_x,
    int64_t extent_y,
    int64_t extent_z,
    int64_t volume,
    const int64_t* coord_to_rank,
    const int64_t* rank_to_coord,
) noexcept nogil:
    cdef int64_t rank, x, y, z, forward_offset
    for rank in range(volume):
        x = rank_to_coord[rank * 3]
        y = rank_to_coord[rank * 3 + 1]
        z = rank_to_coord[rank * 3 + 2]
        if x < 0 or x >= extent_x:
            return rank
        if y < 0 or y >= extent_y:
            return rank
        if z < 0 or z >= extent_z:
            return rank
        forward_offset = (x * extent_y + y) * extent_z + z
        if coord_to_rank[forward_offset] != rank:
            return rank
    return -1


cpdef int64_t validate_level1_maps_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[:, ::1] rank_to_coord,
):
    cdef int64_t result
    cdef int64_t extent_x = root_shape[0]
    cdef int64_t extent_y = root_shape[1]
    cdef int64_t extent_z = root_shape[2]
    cdef int64_t volume = rank_to_coord.shape[0]
    cdef const int64_t* forward_ptr = &coord_to_rank[0, 0, 0]
    cdef const int64_t* inverse_ptr = &rank_to_coord[0, 0]
    with nogil:
        result = _validate_level1_maps(
            extent_x,
            extent_y,
            extent_z,
            volume,
            forward_ptr,
            inverse_ptr,
        )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_level1_face_neighbors(
    int64_t extent_x,
    int64_t extent_y,
    int64_t extent_z,
    int64_t volume,
    const int64_t* coord_to_rank,
    const int64_t* rank_to_coord,
    int64_t* face_neighbor_ids,
) noexcept nogil:
    cdef int64_t rank, x, y, z, forward_offset, output_offset
    cdef int64_t x_stride = extent_y * extent_z
    for rank in range(volume):
        x = rank_to_coord[rank * 3]
        y = rank_to_coord[rank * 3 + 1]
        z = rank_to_coord[rank * 3 + 2]
        forward_offset = (x * extent_y + y) * extent_z + z
        output_offset = rank * 6
        face_neighbor_ids[output_offset] = (
            -1 if x == 0 else coord_to_rank[forward_offset - x_stride]
        )
        face_neighbor_ids[output_offset + 1] = (
            -1 if x + 1 == extent_x else coord_to_rank[forward_offset + x_stride]
        )
        face_neighbor_ids[output_offset + 2] = (
            -1 if y == 0 else coord_to_rank[forward_offset - extent_z]
        )
        face_neighbor_ids[output_offset + 3] = (
            -1 if y + 1 == extent_y else coord_to_rank[forward_offset + extent_z]
        )
        face_neighbor_ids[output_offset + 4] = (
            -1 if z == 0 else coord_to_rank[forward_offset - 1]
        )
        face_neighbor_ids[output_offset + 5] = (
            -1 if z + 1 == extent_z else coord_to_rank[forward_offset + 1]
        )


cpdef void fill_level1_face_neighbors_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[:, ::1] rank_to_coord,
    int64_t[:, ::1] face_neighbor_ids,
):
    cdef int64_t extent_x = root_shape[0]
    cdef int64_t extent_y = root_shape[1]
    cdef int64_t extent_z = root_shape[2]
    cdef int64_t volume = rank_to_coord.shape[0]
    cdef const int64_t* forward_ptr = &coord_to_rank[0, 0, 0]
    cdef const int64_t* inverse_ptr = &rank_to_coord[0, 0]
    cdef int64_t* output_ptr = &face_neighbor_ids[0, 0]
    with nogil:
        _fill_level1_face_neighbors(
            extent_x,
            extent_y,
            extent_z,
            volume,
            forward_ptr,
            inverse_ptr,
            output_ptr,
        )
