# cython: boundscheck=False, wraparound=False

import cython

from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int64_t refined_contact_target_c(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
    int64_t source_leaf_id,
    int dx,
    int dy,
    int dz,
    unsigned char periodic_mask=0,
) noexcept nogil:
    cdef int64_t source_node = leaf_node_ids[source_leaf_id]
    cdef int64_t level = node_levels[source_node]
    cdef int64_t scale = (<int64_t>1) << (level - 1)
    cdef int64_t extent_x = root_shape[0] * scale
    cdef int64_t extent_y = root_shape[1] * scale
    cdef int64_t extent_z = root_shape[2] * scale
    cdef int64_t tx = node_coords[source_node, 0] + dx
    cdef int64_t ty = node_coords[source_node, 1] + dy
    cdef int64_t tz = node_coords[source_node, 2] + dz
    cdef int64_t root_rank, node, depth, shift, child
    # Only topology queries wrap. Published coordinates stay in the original domain.
    # The caller supplies adjacent offsets, so one translation suffices (including
    # a one-block periodic axis). Avoid C remainder semantics for negative indices.
    if periodic_mask & 1:
        if tx < 0:
            tx += extent_x
        elif tx >= extent_x:
            tx -= extent_x
    if periodic_mask & 2:
        if ty < 0:
            ty += extent_y
        elif ty >= extent_y:
            ty -= extent_y
    if periodic_mask & 4:
        if tz < 0:
            tz += extent_z
        elif tz >= extent_z:
            tz -= extent_z
    if (
        tx < 0 or tx >= extent_x
        or ty < 0 or ty >= extent_y
        or tz < 0 or tz >= extent_z
    ):
        return -1
    root_rank = coord_to_rank[tx // scale, ty // scale, tz // scale]
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
    return node
