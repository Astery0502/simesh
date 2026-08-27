"""Selected level-1 Cartesian block geometry for GEO-001."""

import cython

from libc.math cimport isfinite
from libc.stdint cimport int64_t


cdef inline double _face_coordinate(
    double domain_lower,
    double domain_upper,
    double spacing,
    int64_t face_index,
    int64_t domain_cells,
) noexcept nogil:
    if face_index == 0:
        return domain_lower
    if face_index == domain_cells:
        return domain_upper
    return domain_lower + <double>face_index * spacing


cpdef int64_t validate_selected_geometry_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[:, ::1] rank_to_coord,
    const int64_t[::1] block_ids,
    const double[::1] spacing,
):
    cdef int64_t result
    cdef int64_t slot_count = block_ids.shape[0]
    if slot_count == 0:
        return -1
    with nogil:
        result = _validate_selected_geometry(
            &domain_lower[0],
            &domain_upper[0],
            &domain_cell_counts[0],
            &block_cell_counts[0],
            &coord_to_rank[0, 0, 0],
            &rank_to_coord[0, 0],
            rank_to_coord.shape[0],
            &block_ids[0],
            slot_count,
            &spacing[0],
        )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_t _validate_selected_geometry(
    const double* domain_lower,
    const double* domain_upper,
    const int64_t* domain_cell_counts,
    const int64_t* block_cell_counts,
    const int64_t* coord_to_rank,
    const int64_t* rank_to_coord,
    int64_t block_count,
    const int64_t* block_ids,
    int64_t slot_count,
    const double* spacing,
) noexcept nogil:
    cdef int64_t slot, block_id, axis, coordinate, lower_index, upper_index
    cdef int64_t root_extent, forward_offset
    cdef double lower_value, upper_value
    cdef int64_t root_y = domain_cell_counts[1] // block_cell_counts[1]
    cdef int64_t root_z = domain_cell_counts[2] // block_cell_counts[2]
    for slot in range(slot_count):
        block_id = block_ids[slot]
        if block_id < 0 or block_id >= block_count:
            return slot
        for axis in range(3):
            coordinate = rank_to_coord[block_id * 3 + axis]
            root_extent = domain_cell_counts[axis] // block_cell_counts[axis]
            if coordinate < 0 or coordinate >= root_extent:
                return slot
        forward_offset = (
            rank_to_coord[block_id * 3] * root_y
            + rank_to_coord[block_id * 3 + 1]
        ) * root_z + rank_to_coord[block_id * 3 + 2]
        if coord_to_rank[forward_offset] != block_id:
            return slot
        for axis in range(3):
            coordinate = rank_to_coord[block_id * 3 + axis]
            lower_index = coordinate * block_cell_counts[axis]
            upper_index = lower_index + block_cell_counts[axis]
            lower_value = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing[axis],
                lower_index,
                domain_cell_counts[axis],
            )
            upper_value = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing[axis],
                upper_index,
                domain_cell_counts[axis],
            )
            if not isfinite(lower_value) or not isfinite(upper_value):
                return slot
            if lower_value >= upper_value:
                return slot
    return -1


cpdef void fill_level1_block_geometry_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, ::1] rank_to_coord,
    const int64_t[::1] block_ids,
    const double[::1] spacing,
    double[:, :, ::1] block_bounds,
    double[::1] cell_spacing,
):
    cdef int64_t slot_count = block_ids.shape[0]
    cdef int64_t axis
    for axis in range(3):
        cell_spacing[axis] = spacing[axis]
    if slot_count == 0:
        return
    with nogil:
        _fill_level1_block_geometry(
            &domain_lower[0],
            &domain_upper[0],
            &domain_cell_counts[0],
            &block_cell_counts[0],
            &rank_to_coord[0, 0],
            &block_ids[0],
            slot_count,
            &spacing[0],
            &block_bounds[0, 0, 0],
        )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_level1_block_geometry(
    const double* domain_lower,
    const double* domain_upper,
    const int64_t* domain_cell_counts,
    const int64_t* block_cell_counts,
    const int64_t* rank_to_coord,
    const int64_t* block_ids,
    int64_t slot_count,
    const double* spacing,
    double* block_bounds,
) noexcept nogil:
    cdef int64_t slot, block_id, axis, coordinate, lower_index, upper_index
    cdef int64_t output_offset
    for slot in range(slot_count):
        block_id = block_ids[slot]
        for axis in range(3):
            coordinate = rank_to_coord[block_id * 3 + axis]
            lower_index = coordinate * block_cell_counts[axis]
            upper_index = lower_index + block_cell_counts[axis]
            output_offset = slot * 6 + axis
            block_bounds[output_offset] = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing[axis],
                lower_index,
                domain_cell_counts[axis],
            )
            block_bounds[output_offset + 3] = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing[axis],
                upper_index,
                domain_cell_counts[axis],
            )
