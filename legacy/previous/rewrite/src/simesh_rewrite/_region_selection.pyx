# cython: boundscheck=False, wraparound=False

"""Exact refined cell-center physical-region windows for ROI-001."""

import cython

from libc.float cimport DBL_MIN
from libc.math cimport isfinite, ldexp
from libc.stdint cimport INT64_MAX, int64_t

from ._sampling_core cimport canonical_face


cdef inline double _canonical_center(
    double domain_lower,
    double spacing,
    int64_t global_cell,
) noexcept nogil:
    cdef volatile double factor = <double>global_cell + 0.5
    cdef volatile double offset = factor * spacing
    return domain_lower + offset


cdef inline int64_t _center_lower_bound(
    double target,
    double domain_lower,
    double spacing,
    int64_t first_global_cell,
    int64_t cell_count,
) noexcept nogil:
    """Return the first local index whose canonical center is >= target."""
    cdef int64_t lower = 0
    cdef int64_t upper = cell_count
    cdef int64_t middle
    cdef double center
    while lower < upper:
        middle = lower + (upper - lower) // 2
        center = _canonical_center(
            domain_lower,
            spacing,
            first_global_cell + middle,
        )
        if center < target:
            lower = middle + 1
        else:
            upper = middle
    return lower


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple validate_and_count_refined_region_windows_unchecked(
    const double[::1] domain_lower,
    const double[::1] domain_upper,
    const int64_t[::1] root_shape,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const double[::1] region_lower,
    const double[::1] region_upper,
    const double[::1] base_spacing,
):
    """Validate every leaf geometry and count nonempty ROI windows.

    Python owns outer type/layout/domain validation.  Status one denotes an
    invalid lifecycle row, two integer overflow, and three invalid refined
    floating geometry.
    """
    cdef int status = 0
    cdef int64_t leaf_id, node_id, level, shift, scale, axis
    cdef int64_t coordinate, total_cells, first_global, upper_global
    cdef int64_t first_local, upper_local, selected_count = 0
    cdef int64_t bad_leaf = -1
    cdef int64_t bad_axis = -1
    cdef double spacing, leaf_lower, leaf_upper
    cdef double first_center, last_center
    cdef bint nonempty

    with nogil:
        for leaf_id in range(leaf_node_ids.shape[0]):
            node_id = leaf_node_ids[leaf_id]
            if node_id < 0 or node_id >= node_levels.shape[0]:
                status = 1
                bad_leaf = leaf_id
                break
            level = node_levels[node_id]
            if level < 1:
                status = 1
                bad_leaf = leaf_id
                break
            if level > 63:
                status = 2
                bad_leaf = leaf_id
                break
            shift = level - 1
            scale = (<int64_t>1) << shift
            nonempty = True

            for axis in range(3):
                coordinate = node_coords[node_id, axis]
                if coordinate < 0 or coordinate // scale >= root_shape[axis]:
                    status = 1
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break
                if domain_cell_counts[axis] > INT64_MAX // scale:
                    status = 2
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break
                total_cells = domain_cell_counts[axis] * scale
                if coordinate > INT64_MAX // block_cell_counts[axis]:
                    status = 2
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break
                first_global = coordinate * block_cell_counts[axis]
                if first_global > INT64_MAX - block_cell_counts[axis]:
                    status = 2
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break
                upper_global = first_global + block_cell_counts[axis]
                if upper_global > total_cells:
                    status = 1
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break

                spacing = ldexp(base_spacing[axis], -<int>shift)
                if not isfinite(spacing) or spacing < DBL_MIN:
                    status = 3
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break
                leaf_lower = canonical_face(
                    domain_lower[axis],
                    domain_upper[axis],
                    spacing,
                    first_global,
                    total_cells,
                )
                leaf_upper = canonical_face(
                    domain_lower[axis],
                    domain_upper[axis],
                    spacing,
                    upper_global,
                    total_cells,
                )
                first_center = _canonical_center(
                    domain_lower[axis], spacing, first_global
                )
                last_center = _canonical_center(
                    domain_lower[axis], spacing, upper_global - 1
                )
                if (
                    not isfinite(leaf_lower)
                    or not isfinite(leaf_upper)
                    or leaf_lower >= leaf_upper
                    or not isfinite(first_center)
                    or not isfinite(last_center)
                    or first_center <= leaf_lower
                    or first_center > last_center
                    or last_center >= leaf_upper
                ):
                    status = 3
                    bad_leaf = leaf_id
                    bad_axis = axis
                    break

                first_local = _center_lower_bound(
                    region_lower[axis],
                    domain_lower[axis],
                    spacing,
                    first_global,
                    block_cell_counts[axis],
                )
                upper_local = _center_lower_bound(
                    region_upper[axis],
                    domain_lower[axis],
                    spacing,
                    first_global,
                    block_cell_counts[axis],
                )
                if first_local >= upper_local:
                    nonempty = False

            if status != 0:
                break
            if nonempty:
                selected_count += 1

    return status, bad_leaf, bad_axis, selected_count


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fill_refined_region_windows_unchecked(
    const double[::1] domain_lower,
    const int64_t[::1] domain_cell_counts,
    const int64_t[::1] block_cell_counts,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const double[::1] region_lower,
    const double[::1] region_upper,
    const double[::1] base_spacing,
    int64_t[::1] leaf_ids,
    int64_t[:, ::1] cell_lower,
    int64_t[:, ::1] cell_upper,
):
    """Fill exact outputs after the complete validation/count pass."""
    cdef int64_t leaf_id, node_id, shift, scale, axis
    cdef int64_t coordinate, first_global
    cdef int64_t first_local[3]
    cdef int64_t upper_local[3]
    cdef int64_t output_slot = 0
    cdef double spacing
    cdef bint nonempty

    with nogil:
        for leaf_id in range(leaf_node_ids.shape[0]):
            node_id = leaf_node_ids[leaf_id]
            shift = node_levels[node_id] - 1
            scale = (<int64_t>1) << shift
            nonempty = True
            for axis in range(3):
                coordinate = node_coords[node_id, axis]
                first_global = coordinate * block_cell_counts[axis]
                spacing = ldexp(base_spacing[axis], -<int>shift)
                first_local[axis] = _center_lower_bound(
                    region_lower[axis],
                    domain_lower[axis],
                    spacing,
                    first_global,
                    block_cell_counts[axis],
                )
                upper_local[axis] = _center_lower_bound(
                    region_upper[axis],
                    domain_lower[axis],
                    spacing,
                    first_global,
                    block_cell_counts[axis],
                )
                if first_local[axis] >= upper_local[axis]:
                    nonempty = False
            if nonempty:
                leaf_ids[output_slot] = leaf_id
                for axis in range(3):
                    cell_lower[output_slot, axis] = first_local[axis]
                    cell_upper[output_slot, axis] = upper_local[axis]
                output_slot += 1
