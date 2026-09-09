# cython: boundscheck=False, wraparound=False

"""BAL-001 all-touch validation without size-dependent scratch."""

from libc.stdint cimport int64_t

from ._contacts cimport refined_contact_target_c


cpdef tuple first_refined_balance_violation_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
):
    cdef int64_t source_leaf, source_node, source_level
    cdef int64_t column, target_node, target_leaf, child_node
    cdef int dx, dy, dz, child, xbit, ybit, zbit
    for source_leaf in range(leaf_node_ids.shape[0]):
        source_node = leaf_node_ids[source_leaf]
        source_level = node_levels[source_node]
        for column in range(27):
            if column == 13:
                continue
            dx = column % 3 - 1
            dy = (column // 3) % 3 - 1
            dz = column // 9 - 1
            target_node = refined_contact_target_c(
                root_shape,
                coord_to_rank,
                root_node_ids,
                node_levels,
                node_coords,
                child_node_ids,
                node_leaf_ids,
                leaf_node_ids,
                source_leaf,
                dx,
                dy,
                dz,
            )
            if target_node < 0:
                continue
            target_leaf = node_leaf_ids[target_node]
            if target_leaf >= 0:
                if source_level - node_levels[target_node] > 1:
                    return source_leaf, column, target_node, target_node
                continue

            for child in range(8):
                xbit = child & 1
                ybit = (child >> 1) & 1
                zbit = (child >> 2) & 1
                if dx < 0 and xbit != 1:
                    continue
                if dx > 0 and xbit != 0:
                    continue
                if dy < 0 and ybit != 1:
                    continue
                if dy > 0 and ybit != 0:
                    continue
                if dz < 0 and zbit != 1:
                    continue
                if dz > 0 and zbit != 0:
                    continue
                child_node = child_node_ids[target_node, child]
                if node_leaf_ids[child_node] < 0:
                    return source_leaf, column, target_node, child_node
    return -1, -1, -1, -1
