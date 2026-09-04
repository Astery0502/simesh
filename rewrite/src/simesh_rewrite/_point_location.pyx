# cython: boundscheck=False, wraparound=False

"""Exact Cartesian 3D refined point ownership for LOC-001."""

import cython

from libc.math cimport isfinite, ldexp
from libc.stdint cimport int64_t

from ._sampling_core cimport canonical_face, source_cell_index


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
    cdef int64_t point_index, axis, native_cell
    cdef int64_t root_coordinate[3]
    cdef int64_t coordinate[3]
    cdef int64_t child_bits[3]
    cdef int64_t rank, node_id, leaf_id, level, child_column
    cdef int64_t scale, level_domain_cells, midpoint_index
    cdef double midpoint_spacing, midpoint
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

            for axis in range(3):
                native_cell = source_cell_index(
                    points[point_index, axis],
                    domain_lower[axis],
                    domain_upper[axis],
                    base_spacing[axis],
                    domain_cell_counts[axis],
                )
                root_coordinate[axis] = (
                    native_cell // block_cell_counts[axis]
                )
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
                        (2 * coordinate[axis] + 1)
                        * block_cell_counts[axis]
                    )
                    midpoint = canonical_face(
                        domain_lower[axis],
                        domain_upper[axis],
                        midpoint_spacing,
                        midpoint_index,
                        level_domain_cells,
                    )
                    child_bits[axis] = (
                        1 if midpoint <= points[point_index, axis] else 0
                    )
                child_column = (
                    child_bits[0]
                    + 2 * child_bits[1]
                    + 4 * child_bits[2]
                )
                node_id = child_node_ids[node_id, child_column]
                coordinate[0] = 2 * coordinate[0] + child_bits[0]
                coordinate[1] = 2 * coordinate[1] + child_bits[1]
                coordinate[2] = 2 * coordinate[2] + child_bits[2]
                level += 1
                leaf_id = node_leaf_ids[node_id]

            point_leaf_ids[point_index] = leaf_id
