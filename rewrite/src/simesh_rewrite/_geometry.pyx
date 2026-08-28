# cython: boundscheck=False, wraparound=False

"""Selected Cartesian block geometry for GEO-001 and GEO-002."""

import cython

from libc.float cimport DBL_MIN
from libc.math cimport isfinite, ldexp
from libc.stdint cimport INT64_MAX, int64_t


cdef inline double _face_coordinate(
    double domain_lower,
    double domain_upper,
    double spacing,
    int64_t face_index,
    int64_t domain_cells,
) noexcept nogil:
    cdef volatile double offset
    if face_index == 0:
        return domain_lower
    if face_index == domain_cells:
        return domain_upper
    offset = <double>face_index * spacing
    return domain_lower + offset


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


cpdef tuple validate_selected_refined_geometry_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] root_shape,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const int64_t[::1] leaf_ids,
    const double[::1] base_spacing,
):
    cdef int status
    cdef int64_t bad_slot
    cdef int64_t slot_count = leaf_ids.shape[0]
    if slot_count == 0:
        return 0, -1
    with nogil:
        status = _validate_selected_refined_geometry(
            &domain_lower[0],
            &domain_upper[0],
            &root_shape[0],
            &domain_cell_counts[0],
            &block_cell_counts[0],
            &node_levels[0],
            node_levels.shape[0],
            &node_coords[0, 0],
            &leaf_node_ids[0],
            leaf_node_ids.shape[0],
            &leaf_ids[0],
            slot_count,
            &base_spacing[0],
            &bad_slot,
        )
    return status, bad_slot


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _validate_selected_refined_geometry(
    const double* domain_lower,
    const double* domain_upper,
    const int64_t* root_shape,
    const int64_t* domain_cell_counts,
    const int64_t* block_cell_counts,
    const int64_t* node_levels,
    int64_t node_count,
    const int64_t* node_coords,
    const int64_t* leaf_node_ids,
    int64_t leaf_count,
    const int64_t* leaf_ids,
    int64_t slot_count,
    const double* base_spacing,
    int64_t* bad_slot,
) noexcept nogil:
    cdef int64_t slot, leaf_id, node_id, level, shift, scale
    cdef int64_t axis, coordinate, total_cells, lower_index, upper_index
    cdef double spacing, lower_value, upper_value
    for slot in range(slot_count):
        leaf_id = leaf_ids[slot]
        if leaf_id < 0 or leaf_id >= leaf_count:
            bad_slot[0] = slot
            return 1
        node_id = leaf_node_ids[leaf_id]
        if node_id < 0 or node_id >= node_count:
            bad_slot[0] = slot
            return 1
        level = node_levels[node_id]
        if level < 1 or level > 63:
            bad_slot[0] = slot
            return 1
        shift = level - 1
        scale = (<int64_t>1) << shift
        for axis in range(3):
            coordinate = node_coords[node_id * 3 + axis]
            if coordinate < 0 or coordinate // scale >= root_shape[axis]:
                bad_slot[0] = slot
                return 1
            if domain_cell_counts[axis] > INT64_MAX // scale:
                bad_slot[0] = slot
                return 2
            total_cells = domain_cell_counts[axis] * scale
            if coordinate > INT64_MAX // block_cell_counts[axis]:
                bad_slot[0] = slot
                return 2
            lower_index = coordinate * block_cell_counts[axis]
            if lower_index > INT64_MAX - block_cell_counts[axis]:
                bad_slot[0] = slot
                return 2
            upper_index = lower_index + block_cell_counts[axis]
            if upper_index > total_cells:
                bad_slot[0] = slot
                return 1
            spacing = ldexp(base_spacing[axis], -<int>shift)
            if not isfinite(spacing) or spacing < DBL_MIN:
                bad_slot[0] = slot
                return 1
            lower_value = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing,
                lower_index,
                total_cells,
            )
            upper_value = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing,
                upper_index,
                total_cells,
            )
            if (
                not isfinite(lower_value)
                or not isfinite(upper_value)
                or lower_value >= upper_value
            ):
                bad_slot[0] = slot
                return 1
    bad_slot[0] = -1
    return 0


cpdef void fill_refined_leaf_geometry_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const int64_t[::1] leaf_ids,
    const double[::1] base_spacing,
    double[:, :, ::1] leaf_bounds,
    double[:, ::1] leaf_cell_spacing,
):
    cdef int64_t slot_count = leaf_ids.shape[0]
    if slot_count == 0:
        return
    with nogil:
        _fill_refined_leaf_geometry(
            &domain_lower[0],
            &domain_upper[0],
            &domain_cell_counts[0],
            &block_cell_counts[0],
            &node_levels[0],
            &node_coords[0, 0],
            &leaf_node_ids[0],
            &leaf_ids[0],
            slot_count,
            &base_spacing[0],
            &leaf_bounds[0, 0, 0],
            &leaf_cell_spacing[0, 0],
        )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_refined_leaf_geometry(
    const double* domain_lower,
    const double* domain_upper,
    const int64_t* domain_cell_counts,
    const int64_t* block_cell_counts,
    const int64_t* node_levels,
    const int64_t* node_coords,
    const int64_t* leaf_node_ids,
    const int64_t* leaf_ids,
    int64_t slot_count,
    const double* base_spacing,
    double* leaf_bounds,
    double* leaf_cell_spacing,
) noexcept nogil:
    cdef int64_t slot, leaf_id, node_id, shift, scale, axis
    cdef int64_t coordinate, total_cells, lower_index, upper_index
    cdef int64_t bounds_offset, spacing_offset
    cdef double spacing
    for slot in range(slot_count):
        leaf_id = leaf_ids[slot]
        node_id = leaf_node_ids[leaf_id]
        shift = node_levels[node_id] - 1
        scale = (<int64_t>1) << shift
        for axis in range(3):
            coordinate = node_coords[node_id * 3 + axis]
            total_cells = domain_cell_counts[axis] * scale
            lower_index = coordinate * block_cell_counts[axis]
            upper_index = lower_index + block_cell_counts[axis]
            spacing = ldexp(base_spacing[axis], -<int>shift)
            bounds_offset = slot * 6 + axis
            spacing_offset = slot * 3 + axis
            leaf_bounds[bounds_offset] = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing,
                lower_index,
                total_cells,
            )
            leaf_bounds[bounds_offset + 3] = _face_coordinate(
                domain_lower[axis],
                domain_upper[axis],
                spacing,
                upper_index,
                total_cells,
            )
            leaf_cell_spacing[spacing_offset] = spacing
