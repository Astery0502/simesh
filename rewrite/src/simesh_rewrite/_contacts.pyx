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
    cdef int64_t query, source_node, level, scale
    cdef int64_t tx, ty, tz, extent_x, extent_y, extent_z
    cdef int64_t root_x, root_y, root_z, root_rank
    cdef int64_t node, depth, shift, child

    for query in range(source_leaf_ids.shape[0]):
        source_node = leaf_node_ids[source_leaf_ids[query]]
        level = node_levels[source_node]
        scale = (<int64_t>1) << (level - 1)
        extent_x = root_shape[0] * scale
        extent_y = root_shape[1] * scale
        extent_z = root_shape[2] * scale
        tx = node_coords[source_node, 0] + directions[query, 0]
        ty = node_coords[source_node, 1] + directions[query, 1]
        tz = node_coords[source_node, 2] + directions[query, 2]
        if (
            tx < 0 or tx >= extent_x
            or ty < 0 or ty >= extent_y
            or tz < 0 or tz >= extent_z
        ):
            target_node_ids[query] = -1
            continue

        root_x = tx // scale
        root_y = ty // scale
        root_z = tz // scale
        root_rank = coord_to_rank[root_x, root_y, root_z]
        node = root_node_ids[root_rank]
        depth = 1
        while depth < level:
            if node_leaf_ids[node] >= 0:
                break
            shift = level - depth - 1
            child = (
                ((tx >> shift) & 1)
                + 2 * ((ty >> shift) & 1)
                + 4 * ((tz >> shift) & 1)
            )
            node = child_node_ids[node, child]
            depth += 1
        target_node_ids[query] = node
