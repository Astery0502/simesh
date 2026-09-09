# cython: boundscheck=False, wraparound=False

"""Exact grouped refined point kernels for SAM-004 and SAM-005."""

import cython

from libc.float cimport DBL_MIN
from libc.math cimport isfinite, ldexp
from libc.stdint cimport INT64_MAX, int64_t

from ._sampling_core cimport (
    canonical_face,
    fixed_lerp,
    source_cell_index_range,
    trilinear_axis_stencil_from_point,
)


cdef inline int _slot_axis_geometry(
    int64_t slot,
    int64_t axis,
    const double* domain_lower,
    const double* domain_upper,
    const int64_t* domain_cell_counts,
    const int64_t* block_cell_counts,
    const int64_t* node_levels,
    int64_t node_count,
    const int64_t* node_coords,
    const int64_t* leaf_node_ids,
    int64_t leaf_count,
    const int64_t* slot_leaf_ids,
    const double* base_spacing,
    int64_t* coordinate,
    int64_t* level_cells,
    int64_t* global_lower,
    double* spacing,
    double* block_lower,
    double* block_upper,
) noexcept nogil:
    """Return 0, invalid 1, or integer overflow 2 for one slot axis."""
    cdef int64_t leaf_id = slot_leaf_ids[slot]
    cdef int64_t node_id, level, shift, scale, upper_index

    if leaf_id < 0 or leaf_id >= leaf_count:
        return 1
    node_id = leaf_node_ids[leaf_id]
    if node_id < 0 or node_id >= node_count:
        return 1
    level = node_levels[node_id]
    if level < 1 or level > 63:
        return 1
    shift = level - 1
    scale = (<int64_t>1) << shift
    if domain_cell_counts[axis] > INT64_MAX // scale:
        return 2
    level_cells[0] = domain_cell_counts[axis] * scale
    coordinate[0] = node_coords[node_id * 3 + axis]
    if coordinate[0] < 0:
        return 1
    if coordinate[0] > INT64_MAX // block_cell_counts[axis]:
        return 2
    global_lower[0] = coordinate[0] * block_cell_counts[axis]
    if global_lower[0] > INT64_MAX - block_cell_counts[axis]:
        return 2
    upper_index = global_lower[0] + block_cell_counts[axis]
    if upper_index > level_cells[0]:
        return 1

    spacing[0] = ldexp(base_spacing[axis], -<int>shift)
    if not isfinite(spacing[0]) or spacing[0] < DBL_MIN:
        return 1
    block_lower[0] = canonical_face(
        domain_lower[axis],
        domain_upper[axis],
        spacing[0],
        global_lower[0],
        level_cells[0],
    )
    block_upper[0] = canonical_face(
        domain_lower[axis],
        domain_upper[axis],
        spacing[0],
        upper_index,
        level_cells[0],
    )
    if (
        not isfinite(block_lower[0])
        or not isfinite(block_upper[0])
        or block_lower[0] >= block_upper[0]
    ):
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple validate_refined_point_groups_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const int64_t[::1] slot_leaf_ids,
    const double[:, ::1] points,
    const int64_t[::1] point_indices,
    const int64_t[::1] slot_point_offsets,
    const double[::1] base_spacing,
    bint trilinear,
):
    """Return ``(status, bad_slot, bad_point)`` without mutating arrays.

    Status 1 is invalid selected geometry, 2 is signed integer overflow,
    3 is an invalid point index or owner, and 4 is an invalid stencil.
    Structural ndarray and group-offset checks belong to the Python boundary.
    """
    cdef int status
    cdef int64_t slot, axis, order, point_index, cell, local_cell, left_index
    cdef int64_t coordinate[3]
    cdef int64_t level_cells[3]
    cdef int64_t global_lower[3]
    cdef double spacing[3]
    cdef double block_lower[3]
    cdef double block_upper[3]
    cdef double point, weight
    cdef int64_t bad_slot = -1
    cdef int64_t bad_point = -1

    for slot in range(slot_leaf_ids.shape[0]):
        for axis in range(3):
            status = _slot_axis_geometry(
                slot,
                axis,
                &domain_lower[0],
                &domain_upper[0],
                &domain_cell_counts[0],
                &block_cell_counts[0],
                &node_levels[0],
                node_levels.shape[0],
                &node_coords[0, 0],
                &leaf_node_ids[0],
                leaf_node_ids.shape[0],
                &slot_leaf_ids[0],
                &base_spacing[0],
                &coordinate[axis],
                &level_cells[axis],
                &global_lower[axis],
                &spacing[axis],
                &block_lower[axis],
                &block_upper[axis],
            )
            if status != 0:
                bad_slot = slot
                return status, bad_slot, bad_point

        for order in range(
            slot_point_offsets[slot],
            slot_point_offsets[slot + 1],
        ):
            point_index = point_indices[order]
            if point_index < 0 or point_index >= points.shape[0]:
                bad_slot = slot
                bad_point = point_index
                return 3, bad_slot, bad_point
            for axis in range(3):
                point = points[point_index, axis]
                if (
                    not isfinite(point)
                    or point < domain_lower[axis]
                    or point >= domain_upper[axis]
                    or point < block_lower[axis]
                    or point >= block_upper[axis]
                ):
                    bad_slot = slot
                    bad_point = point_index
                    return 3, bad_slot, bad_point
                if trilinear:
                    trilinear_axis_stencil_from_point(
                        point,
                        block_lower[axis],
                        spacing[axis],
                        &left_index,
                        &weight,
                    )
                    if (
                        left_index < -1
                        or left_index > block_cell_counts[axis] - 1
                        or not isfinite(weight)
                        or weight < 0.0
                        or weight >= 1.0
                    ):
                        bad_slot = slot
                        bad_point = point_index
                        return 4, bad_slot, bad_point
                else:
                    cell = source_cell_index_range(
                        point,
                        domain_lower[axis],
                        domain_upper[axis],
                        spacing[axis],
                        global_lower[axis],
                        block_cell_counts[axis],
                        level_cells[axis],
                    )
                    local_cell = cell - global_lower[axis]
                    if local_cell < 0 or local_cell >= block_cell_counts[axis]:
                        bad_slot = slot
                        bad_point = point_index
                        return 3, bad_slot, bad_point
    return 0, bad_slot, bad_point


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void sample_refined_zero_order_point_groups_unchecked(
    const double[:, :, :, :, ::1] payload,
    const int64_t[::1] interior_lower,
    const int64_t[::1] slot_leaf_ids,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const double[::1] base_spacing,
    const double[:, ::1] points,
    const int64_t[::1] point_indices,
    const int64_t[::1] slot_point_offsets,
    double[:, ::1] point_values,
):
    cdef int64_t slot, axis, order, point_index, field, leaf_id, node_id
    cdef int64_t shift, scale, global_cell
    cdef int64_t level_cells[3]
    cdef int64_t global_lower[3]
    cdef int64_t local[3]
    cdef double spacing[3]

    with nogil:
        for slot in range(slot_leaf_ids.shape[0]):
            leaf_id = slot_leaf_ids[slot]
            node_id = leaf_node_ids[leaf_id]
            shift = node_levels[node_id] - 1
            scale = (<int64_t>1) << shift
            for axis in range(3):
                level_cells[axis] = domain_cell_counts[axis] * scale
                global_lower[axis] = (
                    node_coords[node_id, axis] * block_cell_counts[axis]
                )
                spacing[axis] = ldexp(base_spacing[axis], -<int>shift)
            for order in range(
                slot_point_offsets[slot],
                slot_point_offsets[slot + 1],
            ):
                point_index = point_indices[order]
                for axis in range(3):
                    global_cell = source_cell_index_range(
                        points[point_index, axis],
                        domain_lower[axis],
                        domain_upper[axis],
                        spacing[axis],
                        global_lower[axis],
                        block_cell_counts[axis],
                        level_cells[axis],
                    )
                    local[axis] = global_cell - global_lower[axis]
                for field in range(payload.shape[1]):
                    point_values[point_index, field] = payload[
                        slot,
                        field,
                        interior_lower[0] + local[0],
                        interior_lower[1] + local[1],
                        interior_lower[2] + local[2],
                    ]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void sample_refined_trilinear_point_groups_unchecked(
    const double[:, :, :, :, ::1] payload,
    const int64_t[::1] interior_lower,
    const int64_t[::1] slot_leaf_ids,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const double[::1] base_spacing,
    const double[:, ::1] points,
    const int64_t[::1] point_indices,
    const int64_t[::1] slot_point_offsets,
    double[:, ::1] point_values,
):
    cdef int64_t slot, axis, order, point_index, field, leaf_id, node_id
    cdef int64_t shift, scale, left_index
    cdef int64_t level_cells[3]
    cdef int64_t global_lower[3]
    cdef int64_t left[3]
    cdef int64_t right[3]
    cdef double spacing[3]
    cdef double block_lower[3]
    cdef double weight[3]
    cdef double c00, c01, c10, c11, c0, c1

    with nogil:
        for slot in range(slot_leaf_ids.shape[0]):
            leaf_id = slot_leaf_ids[slot]
            node_id = leaf_node_ids[leaf_id]
            shift = node_levels[node_id] - 1
            scale = (<int64_t>1) << shift
            for axis in range(3):
                level_cells[axis] = domain_cell_counts[axis] * scale
                global_lower[axis] = (
                    node_coords[node_id, axis] * block_cell_counts[axis]
                )
                spacing[axis] = ldexp(base_spacing[axis], -<int>shift)
                block_lower[axis] = canonical_face(
                    domain_lower[axis],
                    domain_upper[axis],
                    spacing[axis],
                    global_lower[axis],
                    level_cells[axis],
                )
            for order in range(
                slot_point_offsets[slot],
                slot_point_offsets[slot + 1],
            ):
                point_index = point_indices[order]
                for axis in range(3):
                    trilinear_axis_stencil_from_point(
                        points[point_index, axis],
                        block_lower[axis],
                        spacing[axis],
                        &left_index,
                        &weight[axis],
                    )
                    left[axis] = interior_lower[axis] + left_index
                    right[axis] = left[axis] + 1
                for field in range(payload.shape[1]):
                    c00 = fixed_lerp(
                        payload[slot, field, left[0], left[1], left[2]],
                        payload[slot, field, left[0], left[1], right[2]],
                        weight[2],
                    )
                    c01 = fixed_lerp(
                        payload[slot, field, left[0], right[1], left[2]],
                        payload[slot, field, left[0], right[1], right[2]],
                        weight[2],
                    )
                    c10 = fixed_lerp(
                        payload[slot, field, right[0], left[1], left[2]],
                        payload[slot, field, right[0], left[1], right[2]],
                        weight[2],
                    )
                    c11 = fixed_lerp(
                        payload[slot, field, right[0], right[1], left[2]],
                        payload[slot, field, right[0], right[1], right[2]],
                        weight[2],
                    )
                    c0 = fixed_lerp(c00, c01, weight[1])
                    c1 = fixed_lerp(c10, c11, weight[1])
                    point_values[point_index, field] = fixed_lerp(
                        c0,
                        c1,
                        weight[0],
                    )
