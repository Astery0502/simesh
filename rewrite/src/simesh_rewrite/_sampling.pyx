"""Exact native-grid placement for SAM-001."""

import cython

from libc.math cimport floor, isfinite
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


cdef inline double _canonical_face(
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t face_index,
    int64_t domain_cells,
) noexcept nogil:
    cdef volatile double offset
    if face_index == 0:
        return domain_lower
    if face_index == domain_cells:
        return domain_upper
    offset = <double>face_index * native_spacing
    return domain_lower + offset


cdef inline double _sample_center(
    double sample_lower,
    double output_spacing,
    int64_t output_index,
) noexcept nogil:
    cdef volatile double factor = <double>output_index + 0.5
    cdef volatile double offset = factor * output_spacing
    return sample_lower + offset


cdef inline int64_t _source_cell_index_binary(
    double center,
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t domain_cells,
) noexcept nogil:
    cdef int64_t lower = 0
    cdef int64_t upper = domain_cells - 1
    cdef int64_t middle
    while lower < upper:
        middle = lower + (upper - lower + 1) // 2
        if _canonical_face(
            domain_lower,
            domain_upper,
            native_spacing,
            middle,
            domain_cells,
        ) <= center:
            lower = middle
        else:
            upper = middle - 1
    return lower


cdef inline int64_t _source_cell_index(
    double center,
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t domain_cells,
) noexcept nogil:
    cdef volatile double delta = center - domain_lower
    cdef volatile double ratio = delta / native_spacing
    cdef int64_t candidate
    if ratio <= 0.0:
        candidate = 0
    elif ratio >= <double>(domain_cells - 1):
        candidate = domain_cells - 1
    else:
        candidate = <int64_t>ratio
    if _canonical_face(
        domain_lower,
        domain_upper,
        native_spacing,
        candidate,
        domain_cells,
    ) > center or (
        candidate + 1 < domain_cells
        and _canonical_face(
            domain_lower,
            domain_upper,
            native_spacing,
            candidate + 1,
            domain_cells,
        ) <= center
    ):
        return _source_cell_index_binary(
            center,
            domain_lower,
            domain_upper,
            native_spacing,
            domain_cells,
        )
    return candidate


cdef inline int64_t _output_owner_block(
    int64_t output_index,
    double sample_lower,
    double output_spacing,
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t domain_cells,
    int64_t block_cells,
) noexcept nogil:
    return _source_cell_index(
        _sample_center(sample_lower, output_spacing, output_index),
        domain_lower,
        domain_upper,
        native_spacing,
        domain_cells,
    ) // block_cells


cdef inline int64_t _lower_bound_output_owner(
    int64_t target_block_coordinate,
    int64_t output_cells,
    double sample_lower,
    double output_spacing,
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t domain_cells,
    int64_t block_cells,
) noexcept nogil:
    cdef int64_t lower = 0
    cdef int64_t upper = output_cells
    cdef int64_t middle
    while lower < upper:
        middle = lower + (upper - lower) // 2
        if _output_owner_block(
            middle,
            sample_lower,
            output_spacing,
            domain_lower,
            domain_upper,
            native_spacing,
            domain_cells,
            block_cells,
        ) < target_block_coordinate:
            lower = middle + 1
        else:
            upper = middle
    return lower


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void sample_level1_zero_order_unchecked(
    const double[:, :, :, :, ::1] payload,
    const int64_t[::1] payload_valid_lower,
    const int64_t[::1] block_ids,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, ::1] rank_to_coord,
    const double[::1] native_spacing,
    const double[::1] sample_lower,
    const double[::1] output_spacing,
    double[:, :, :, ::1] uniform_grid,
):
    cdef int64_t slot, field, i, j, k, block_id
    cdef int64_t global_i, global_j, global_k
    cdef int64_t local_i, local_j, local_k
    cdef int64_t coordinate[3]
    cdef int64_t output_lower[3]
    cdef int64_t output_upper[3]

    with nogil:
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            coordinate[0] = rank_to_coord[block_id, 0]
            coordinate[1] = rank_to_coord[block_id, 1]
            coordinate[2] = rank_to_coord[block_id, 2]
            output_lower[0] = _lower_bound_output_owner(
                coordinate[0],
                uniform_grid.shape[1],
                sample_lower[0],
                output_spacing[0],
                domain_lower[0],
                domain_upper[0],
                native_spacing[0],
                domain_cell_counts[0],
                block_cell_counts[0],
            )
            output_upper[0] = _lower_bound_output_owner(
                coordinate[0] + 1,
                uniform_grid.shape[1],
                sample_lower[0],
                output_spacing[0],
                domain_lower[0],
                domain_upper[0],
                native_spacing[0],
                domain_cell_counts[0],
                block_cell_counts[0],
            )
            output_lower[1] = _lower_bound_output_owner(
                coordinate[1],
                uniform_grid.shape[2],
                sample_lower[1],
                output_spacing[1],
                domain_lower[1],
                domain_upper[1],
                native_spacing[1],
                domain_cell_counts[1],
                block_cell_counts[1],
            )
            output_upper[1] = _lower_bound_output_owner(
                coordinate[1] + 1,
                uniform_grid.shape[2],
                sample_lower[1],
                output_spacing[1],
                domain_lower[1],
                domain_upper[1],
                native_spacing[1],
                domain_cell_counts[1],
                block_cell_counts[1],
            )
            output_lower[2] = _lower_bound_output_owner(
                coordinate[2],
                uniform_grid.shape[3],
                sample_lower[2],
                output_spacing[2],
                domain_lower[2],
                domain_upper[2],
                native_spacing[2],
                domain_cell_counts[2],
                block_cell_counts[2],
            )
            output_upper[2] = _lower_bound_output_owner(
                coordinate[2] + 1,
                uniform_grid.shape[3],
                sample_lower[2],
                output_spacing[2],
                domain_lower[2],
                domain_upper[2],
                native_spacing[2],
                domain_cell_counts[2],
                block_cell_counts[2],
            )

            for i in range(output_lower[0], output_upper[0]):
                global_i = _source_cell_index(
                    _sample_center(sample_lower[0], output_spacing[0], i),
                    domain_lower[0],
                    domain_upper[0],
                    native_spacing[0],
                    domain_cell_counts[0],
                )
                local_i = global_i - coordinate[0] * block_cell_counts[0]
                for j in range(output_lower[1], output_upper[1]):
                    global_j = _source_cell_index(
                        _sample_center(sample_lower[1], output_spacing[1], j),
                        domain_lower[1],
                        domain_upper[1],
                        native_spacing[1],
                        domain_cell_counts[1],
                    )
                    local_j = global_j - coordinate[1] * block_cell_counts[1]
                    for k in range(output_lower[2], output_upper[2]):
                        global_k = _source_cell_index(
                            _sample_center(sample_lower[2], output_spacing[2], k),
                            domain_lower[2],
                            domain_upper[2],
                            native_spacing[2],
                            domain_cell_counts[2],
                        )
                        local_k = (
                            global_k - coordinate[2] * block_cell_counts[2]
                        )
                        for field in range(payload.shape[1]):
                            uniform_grid[field, i, j, k] = payload[
                                slot,
                                field,
                                payload_valid_lower[0] + local_i,
                                payload_valid_lower[1] + local_j,
                                payload_valid_lower[2] + local_k,
                            ]


cdef inline void _trilinear_axis_stencil(
    int64_t output_index,
    int64_t block_coordinate,
    int64_t block_cells,
    double sample_lower,
    double output_spacing,
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t domain_cells,
    int64_t* left_index,
    double* weight,
) noexcept nogil:
    cdef double center = _sample_center(
        sample_lower,
        output_spacing,
        output_index,
    )
    cdef double block_lower = _canonical_face(
        domain_lower,
        domain_upper,
        native_spacing,
        block_coordinate * block_cells,
        domain_cells,
    )
    cdef volatile double delta = center - block_lower
    cdef volatile double ratio = delta / native_spacing
    cdef volatile double normalized = ratio - 0.5
    left_index[0] = <int64_t>floor(normalized)
    weight[0] = normalized - <double>left_index[0]


cdef inline bint _trilinear_axis_stencil_is_valid(
    int64_t output_index,
    int64_t block_coordinate,
    int64_t block_cells,
    double sample_lower,
    double output_spacing,
    double domain_lower,
    double domain_upper,
    double native_spacing,
    int64_t domain_cells,
) noexcept nogil:
    cdef int64_t left_index
    cdef double weight
    _trilinear_axis_stencil(
        output_index,
        block_coordinate,
        block_cells,
        sample_lower,
        output_spacing,
        domain_lower,
        domain_upper,
        native_spacing,
        domain_cells,
        &left_index,
        &weight,
    )
    return (
        -1 <= left_index <= block_cells - 1
        and isfinite(weight)
        and 0.0 <= weight < 1.0
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int64_t validate_trilinear_stencils_unchecked(
    const int64_t[::1] block_ids,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, ::1] rank_to_coord,
    const double[::1] native_spacing,
    const double[::1] sample_lower,
    const double[::1] output_spacing,
    const double[:, :, :, ::1] uniform_grid,
):
    cdef int64_t slot, axis, block_id, coordinate
    cdef int64_t output_lower[3]
    cdef int64_t output_upper[3]
    for slot in range(block_ids.shape[0]):
        block_id = block_ids[slot]
        for axis in range(3):
            coordinate = rank_to_coord[block_id, axis]
            output_lower[axis] = _lower_bound_output_owner(
                coordinate,
                uniform_grid.shape[axis + 1],
                sample_lower[axis],
                output_spacing[axis],
                domain_lower[axis],
                domain_upper[axis],
                native_spacing[axis],
                domain_cell_counts[axis],
                block_cell_counts[axis],
            )
            output_upper[axis] = _lower_bound_output_owner(
                coordinate + 1,
                uniform_grid.shape[axis + 1],
                sample_lower[axis],
                output_spacing[axis],
                domain_lower[axis],
                domain_upper[axis],
                native_spacing[axis],
                domain_cell_counts[axis],
                block_cell_counts[axis],
            )
        if (
            output_lower[0] == output_upper[0]
            or output_lower[1] == output_upper[1]
            or output_lower[2] == output_upper[2]
        ):
            continue
        for axis in range(3):
            coordinate = rank_to_coord[block_id, axis]
            if not _trilinear_axis_stencil_is_valid(
                output_lower[axis],
                coordinate,
                block_cell_counts[axis],
                sample_lower[axis],
                output_spacing[axis],
                domain_lower[axis],
                domain_upper[axis],
                native_spacing[axis],
                domain_cell_counts[axis],
            ) or not _trilinear_axis_stencil_is_valid(
                output_upper[axis] - 1,
                coordinate,
                block_cell_counts[axis],
                sample_lower[axis],
                output_spacing[axis],
                domain_lower[axis],
                domain_upper[axis],
                native_spacing[axis],
                domain_cell_counts[axis],
            ):
                return slot
    return -1


cdef inline double _lerp(
    double left,
    double right,
    double weight,
) noexcept nogil:
    cdef volatile double one_minus = 1.0 - weight
    cdef volatile double left_term = left * one_minus
    cdef volatile double right_term = right * weight
    cdef volatile double result = left_term + right_term
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void sample_level1_trilinear_unchecked(
    const double[:, :, :, :, ::1] payload,
    const int64_t[::1] interior_lower,
    const int64_t[::1] block_ids,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, ::1] rank_to_coord,
    const double[::1] native_spacing,
    const double[::1] sample_lower,
    const double[::1] output_spacing,
    double[:, :, :, ::1] uniform_grid,
):
    cdef int64_t slot, field, axis, i, j, k, block_id
    cdef int64_t coordinate[3]
    cdef int64_t output_lower[3]
    cdef int64_t output_upper[3]
    cdef int64_t left_i, left_j, left_k
    cdef int64_t i0, i1, j0, j1, k0, k1
    cdef double wx, wy, wz
    cdef double c00, c01, c10, c11, c0, c1

    with nogil:
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            for axis in range(3):
                coordinate[axis] = rank_to_coord[block_id, axis]
                output_lower[axis] = _lower_bound_output_owner(
                    coordinate[axis],
                    uniform_grid.shape[axis + 1],
                    sample_lower[axis],
                    output_spacing[axis],
                    domain_lower[axis],
                    domain_upper[axis],
                    native_spacing[axis],
                    domain_cell_counts[axis],
                    block_cell_counts[axis],
                )
                output_upper[axis] = _lower_bound_output_owner(
                    coordinate[axis] + 1,
                    uniform_grid.shape[axis + 1],
                    sample_lower[axis],
                    output_spacing[axis],
                    domain_lower[axis],
                    domain_upper[axis],
                    native_spacing[axis],
                    domain_cell_counts[axis],
                    block_cell_counts[axis],
                )

            for i in range(output_lower[0], output_upper[0]):
                _trilinear_axis_stencil(
                    i,
                    coordinate[0],
                    block_cell_counts[0],
                    sample_lower[0],
                    output_spacing[0],
                    domain_lower[0],
                    domain_upper[0],
                    native_spacing[0],
                    domain_cell_counts[0],
                    &left_i,
                    &wx,
                )
                i0 = interior_lower[0] + left_i
                i1 = i0 + 1
                for j in range(output_lower[1], output_upper[1]):
                    _trilinear_axis_stencil(
                        j,
                        coordinate[1],
                        block_cell_counts[1],
                        sample_lower[1],
                        output_spacing[1],
                        domain_lower[1],
                        domain_upper[1],
                        native_spacing[1],
                        domain_cell_counts[1],
                        &left_j,
                        &wy,
                    )
                    j0 = interior_lower[1] + left_j
                    j1 = j0 + 1
                    for k in range(output_lower[2], output_upper[2]):
                        _trilinear_axis_stencil(
                            k,
                            coordinate[2],
                            block_cell_counts[2],
                            sample_lower[2],
                            output_spacing[2],
                            domain_lower[2],
                            domain_upper[2],
                            native_spacing[2],
                            domain_cell_counts[2],
                            &left_k,
                            &wz,
                        )
                        k0 = interior_lower[2] + left_k
                        k1 = k0 + 1
                        for field in range(payload.shape[1]):
                            c00 = _lerp(
                                payload[slot, field, i0, j0, k0],
                                payload[slot, field, i0, j0, k1],
                                wz,
                            )
                            c01 = _lerp(
                                payload[slot, field, i0, j1, k0],
                                payload[slot, field, i0, j1, k1],
                                wz,
                            )
                            c10 = _lerp(
                                payload[slot, field, i1, j0, k0],
                                payload[slot, field, i1, j0, k1],
                                wz,
                            )
                            c11 = _lerp(
                                payload[slot, field, i1, j1, k0],
                                payload[slot, field, i1, j1, k1],
                                wz,
                            )
                            c0 = _lerp(c00, c01, wy)
                            c1 = _lerp(c10, c11, wy)
                            uniform_grid[field, i, j, k] = _lerp(c0, c1, wx)
