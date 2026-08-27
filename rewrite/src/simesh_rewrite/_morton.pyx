"""Clipped recursive Morton traversal for MOR-001."""

import cython

from libc.stdint cimport int64_t, uint64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_t _visit_morton(
    uint64_t origin_x,
    uint64_t origin_y,
    uint64_t origin_z,
    int bit,
    uint64_t extent_x,
    uint64_t extent_y,
    uint64_t extent_z,
    int64_t rank,
    int64_t* coord_to_rank,
    int64_t* rank_to_coord,
) noexcept nogil:
    cdef int child
    cdef uint64_t forward_offset
    cdef uint64_t step, child_x, child_y, child_z

    if bit < 0:
        forward_offset = (origin_x * extent_y + origin_y) * extent_z + origin_z
        coord_to_rank[forward_offset] = rank
        rank_to_coord[rank * 3] = <int64_t>origin_x
        rank_to_coord[rank * 3 + 1] = <int64_t>origin_y
        rank_to_coord[rank * 3 + 2] = <int64_t>origin_z
        return rank + 1

    step = (<uint64_t>1) << bit
    for child in range(8):
        child_x = origin_x + (step if child & 1 else 0)
        child_y = origin_y + (step if child & 2 else 0)
        child_z = origin_z + (step if child & 4 else 0)
        if child_x >= extent_x or child_y >= extent_y or child_z >= extent_z:
            continue
        rank = _visit_morton(
            child_x,
            child_y,
            child_z,
            bit - 1,
            extent_x,
            extent_y,
            extent_z,
            rank,
            coord_to_rank,
            rank_to_coord,
        )
    return rank


cpdef void fill_level1_morton_unchecked(
    const int64_t[::1] root_shape,
    int64_t[:, :, ::1] coord_to_rank,
    int64_t[:, ::1] rank_to_coord,
):
    cdef uint64_t maximum_coordinate
    cdef uint64_t shifted
    cdef uint64_t extent_x = <uint64_t>root_shape[0]
    cdef uint64_t extent_y = <uint64_t>root_shape[1]
    cdef uint64_t extent_z = <uint64_t>root_shape[2]
    cdef int64_t* forward_ptr = &coord_to_rank[0, 0, 0]
    cdef int64_t* inverse_ptr = &rank_to_coord[0, 0]
    cdef int bit = -1
    cdef int64_t count
    cdef int64_t expected

    maximum_coordinate = extent_x - 1
    if extent_y - 1 > maximum_coordinate:
        maximum_coordinate = extent_y - 1
    if extent_z - 1 > maximum_coordinate:
        maximum_coordinate = extent_z - 1
    shifted = maximum_coordinate
    while shifted:
        bit += 1
        shifted >>= 1

    with nogil:
        count = _visit_morton(
            0,
            0,
            0,
            bit,
            extent_x,
            extent_y,
            extent_z,
            0,
            forward_ptr,
            inverse_ptr,
        )
    expected = root_shape[0] * root_shape[1] * root_shape[2]
    if count != expected:
        raise RuntimeError("internal Morton traversal count mismatch")
