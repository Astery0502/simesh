# cython: boundscheck=False, wraparound=False

"""Typed TOP-002 raw refined contact target lookup."""

import cython

from libc.stdint cimport int64_t


cpdef int64_t invalid_contact_direction_unchecked(
    const int64_t[:, ::1] directions,
):
    cdef int64_t query
    cdef int64_t dx, dy, dz
    for query in range(directions.shape[0]):
        dx = directions[query, 0]
        dy = directions[query, 1]
        dz = directions[query, 2]
        if (
            dx < -1 or dx > 1
            or dy < -1 or dy > 1
            or dz < -1 or dz > 1
            or (dx == 0 and dy == 0 and dz == 0)
        ):
            return query
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fill_refined_contact_targets_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
    const int64_t[::1] source_leaf_ids,
    const int64_t[:, ::1] directions,
    int64_t[::1] target_node_ids,
):
    cdef int64_t query

    for query in range(source_leaf_ids.shape[0]):
        target_node_ids[query] = refined_contact_target_c(
            root_shape,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            source_leaf_ids[query],
            <int>directions[query, 0],
            <int>directions[query, 1],
            <int>directions[query, 2],
        )
