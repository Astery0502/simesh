# cython: boundscheck=False, wraparound=False

"""Exact Cartesian 3D refined point ownership for LOC-001."""

import cython

from libc.float cimport DBL_MIN
from libc.math cimport isfinite, ldexp
from libc.stdint cimport INT64_MAX, int64_t

from ._sampling_core cimport canonical_face, source_cell_index


cdef inline int64_t _locate_refined_interior_point(
    const double* point,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const double[::1] base_spacing,
) noexcept nogil:
    """Return the LOC-001 owner for one already-proved interior point."""
    cdef int64_t axis, native_cell
    cdef int64_t root_coordinate[3]
    cdef int64_t coordinate[3]
    cdef int64_t child_bits[3]
    cdef int64_t rank, node_id, leaf_id, level, child_column
    cdef int64_t scale, level_domain_cells, midpoint_index
    cdef double midpoint_spacing, midpoint

    for axis in range(3):
        native_cell = source_cell_index(
            point[axis],
            domain_lower[axis],
            domain_upper[axis],
            base_spacing[axis],
            domain_cell_counts[axis],
        )
        root_coordinate[axis] = native_cell // block_cell_counts[axis]
        coordinate[axis] = root_coordinate[axis]

    rank = coord_to_rank[
        root_coordinate[0],
        root_coordinate[1],
        root_coordinate[2],
    ]
    node_id = root_node_ids[rank]
    leaf_id = node_leaf_ids[node_id]
    level = 1

    while leaf_id < 0:
        scale = (<int64_t>1) << level
        child_column = 0
        for axis in range(3):
            level_domain_cells = domain_cell_counts[axis] * scale
            midpoint_spacing = ldexp(base_spacing[axis], -<int>level)
            midpoint_index = (
                (2 * coordinate[axis] + 1) * block_cell_counts[axis]
            )
            midpoint = canonical_face(
                domain_lower[axis],
                domain_upper[axis],
                midpoint_spacing,
                midpoint_index,
                level_domain_cells,
            )
            child_bits[axis] = 1 if midpoint <= point[axis] else 0
        child_column = (
            child_bits[0] + 2 * child_bits[1] + 4 * child_bits[2]
        )
        node_id = child_node_ids[node_id, child_column]
        coordinate[0] = 2 * coordinate[0] + child_bits[0]
        coordinate[1] = 2 * coordinate[1] + child_bits[1]
        coordinate[2] = 2 * coordinate[2] + child_bits[2]
        level += 1
        leaf_id = node_leaf_ids[node_id]

    return leaf_id


cdef inline bint _hint_contains_point(
    const double* point,
    int64_t hint_leaf_id,
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const double[::1] base_spacing,
) noexcept nogil:
    """Apply the exact GEO-002 half-open containment rule to one valid hint."""
    cdef int64_t node_id = leaf_node_ids[hint_leaf_id]
    cdef int64_t shift = node_levels[node_id] - 1
    cdef int64_t scale = (<int64_t>1) << shift
    cdef int64_t axis, total_cells, lower_index, upper_index
    cdef double spacing, lower_face, upper_face

    for axis in range(3):
        total_cells = domain_cell_counts[axis] * scale
        lower_index = node_coords[node_id, axis] * block_cell_counts[axis]
        upper_index = lower_index + block_cell_counts[axis]
        spacing = ldexp(base_spacing[axis], -<int>shift)
        lower_face = canonical_face(
            domain_lower[axis],
            domain_upper[axis],
            spacing,
            lower_index,
            total_cells,
        )
        upper_face = canonical_face(
            domain_lower[axis],
            domain_upper[axis],
            spacing,
            upper_index,
            total_cells,
        )
        if point[axis] < lower_face or point[axis] >= upper_face:
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple validate_refined_points_finite_unchecked(
    const double[:, ::1] points,
):
    """Return the first nonfinite point coordinate, or ``(-1, -1)``."""
    cdef int64_t point_index, axis
    cdef int64_t bad_point = -1
    cdef int64_t bad_axis = -1
    with nogil:
        for point_index in range(points.shape[0]):
            for axis in range(3):
                if not isfinite(points[point_index, axis]):
                    bad_point = point_index
                    bad_axis = axis
                    break
            if bad_point >= 0:
                break
    return bad_point, bad_axis


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fill_refined_point_leaf_ids_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const double[::1] base_spacing,
    const double[:, ::1] points,
    int64_t[::1] point_leaf_ids,
):
    """Fill owners after LOC-001's complete Python-boundary preflight."""
    cdef int64_t point_index, axis
    cdef bint exterior

    with nogil:
        for point_index in range(points.shape[0]):
            exterior = False
            for axis in range(3):
                if (
                    points[point_index, axis] < domain_lower[axis]
                    or points[point_index, axis] >= domain_upper[axis]
                ):
                    exterior = True
                    break
            if exterior:
                point_leaf_ids[point_index] = -1
                continue

            point_leaf_ids[point_index] = _locate_refined_interior_point(
                &points[point_index, 0],
                domain_lower,
                domain_upper,
                domain_cell_counts,
                block_cell_counts,
                coord_to_rank,
                root_node_ids,
                child_node_ids,
                node_leaf_ids,
                base_spacing,
            )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple validate_refined_point_hints_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] root_shape,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    int64_t max_level,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
    const double[::1] base_spacing,
    const double[:, ::1] points,
    const int64_t[::1] hint_leaf_ids,
):
    """Validate every candidate hint before HLO output mutation.

    Status one is an invalid hint value, two is invalid hinted FST/GEO state,
    three is signed integer overflow, four is invalid floating geometry, and
    five is a nonfinite point coordinate.
    """
    cdef int status = 0
    cdef int64_t point_index, hint, node_id, level, shift, scale, axis
    cdef int64_t coordinate, total_cells, lower_index, upper_index
    cdef int64_t bad_point = -1
    cdef int64_t bad_axis = -1
    cdef int64_t interior_point_count = 0
    cdef int64_t exterior_point_count = 0
    cdef int64_t hint_candidate_count = 0
    cdef double spacing, lower_face, upper_face
    cdef bint exterior

    with nogil:
        for point_index in range(points.shape[0]):
            exterior = False
            for axis in range(3):
                if not isfinite(points[point_index, axis]):
                    status = 5
                    bad_point = point_index
                    bad_axis = axis
                    break
                if (
                    points[point_index, axis] < domain_lower[axis]
                    or points[point_index, axis] >= domain_upper[axis]
                ):
                    exterior = True
            if status != 0:
                break
            if exterior:
                exterior_point_count += 1
            else:
                interior_point_count += 1

        if status == 0:
            for point_index in range(hint_leaf_ids.shape[0]):
                hint = hint_leaf_ids[point_index]
                if hint < -1 or hint >= leaf_node_ids.shape[0]:
                    status = 1
                    bad_point = point_index
                    break
                if hint < 0:
                    continue
                hint_candidate_count += 1
                node_id = leaf_node_ids[hint]
                if (
                    node_id < 0
                    or node_id >= node_levels.shape[0]
                    or node_leaf_ids[node_id] != hint
                ):
                    status = 2
                    bad_point = point_index
                    break
                level = node_levels[node_id]
                if level < 1 or level > max_level:
                    status = 2
                    bad_point = point_index
                    break
                if level > 63:
                    status = 3
                    bad_point = point_index
                    break
                shift = level - 1
                scale = (<int64_t>1) << shift
                for axis in range(3):
                    coordinate = node_coords[node_id, axis]
                    if coordinate < 0 or coordinate // scale >= root_shape[axis]:
                        status = 2
                        bad_point = point_index
                        bad_axis = axis
                        break
                    if domain_cell_counts[axis] > INT64_MAX // scale:
                        status = 3
                        bad_point = point_index
                        bad_axis = axis
                        break
                    total_cells = domain_cell_counts[axis] * scale
                    if coordinate > INT64_MAX // block_cell_counts[axis]:
                        status = 3
                        bad_point = point_index
                        bad_axis = axis
                        break
                    lower_index = coordinate * block_cell_counts[axis]
                    if lower_index > INT64_MAX - block_cell_counts[axis]:
                        status = 3
                        bad_point = point_index
                        bad_axis = axis
                        break
                    upper_index = lower_index + block_cell_counts[axis]
                    if upper_index > total_cells:
                        status = 2
                        bad_point = point_index
                        bad_axis = axis
                        break
                    spacing = ldexp(base_spacing[axis], -<int>shift)
                    if not isfinite(spacing) or spacing < DBL_MIN:
                        status = 4
                        bad_point = point_index
                        bad_axis = axis
                        break
                    lower_face = canonical_face(
                        domain_lower[axis],
                        domain_upper[axis],
                        spacing,
                        lower_index,
                        total_cells,
                    )
                    upper_face = canonical_face(
                        domain_lower[axis],
                        domain_upper[axis],
                        spacing,
                        upper_index,
                        total_cells,
                    )
                    if (
                        not isfinite(lower_face)
                        or not isfinite(upper_face)
                        or lower_face >= upper_face
                    ):
                        status = 4
                        bad_point = point_index
                        bad_axis = axis
                        break
                if status != 0:
                    break

    return (
        status,
        bad_point,
        bad_axis,
        interior_point_count,
        exterior_point_count,
        hint_candidate_count,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int64_t fill_refined_point_leaf_ids_with_hints_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
    const double[::1] base_spacing,
    const double[:, ::1] points,
    const int64_t[::1] hint_leaf_ids,
    int64_t[::1] point_leaf_ids,
):
    """Fill exact hint-or-LOC owners and return the hint-hit count."""
    cdef int64_t point_index, axis, hint, owner
    cdef int64_t hint_hit_count = 0
    cdef bint exterior

    with nogil:
        for point_index in range(points.shape[0]):
            exterior = False
            for axis in range(3):
                if (
                    points[point_index, axis] < domain_lower[axis]
                    or points[point_index, axis] >= domain_upper[axis]
                ):
                    exterior = True
                    break
            if exterior:
                point_leaf_ids[point_index] = -1
                continue

            hint = hint_leaf_ids[point_index]
            if hint >= 0 and _hint_contains_point(
                &points[point_index, 0],
                hint,
                domain_lower,
                domain_upper,
                domain_cell_counts,
                block_cell_counts,
                node_levels,
                node_coords,
                leaf_node_ids,
                base_spacing,
            ):
                owner = hint
                hint_hit_count += 1
            else:
                owner = _locate_refined_interior_point(
                    &points[point_index, 0],
                    domain_lower,
                    domain_upper,
                    domain_cell_counts,
                    block_cell_counts,
                    coord_to_rank,
                    root_node_ids,
                    child_node_ids,
                    node_leaf_ids,
                    base_spacing,
                )
            point_leaf_ids[point_index] = owner

    return hint_hit_count
