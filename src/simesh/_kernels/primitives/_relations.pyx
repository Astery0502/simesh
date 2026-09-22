# cython: boundscheck=False, wraparound=False

"""Scratch-free REL-001 balanced refined direction records."""

from libc.stdint cimport int64_t, uint8_t

from ._contacts cimport refined_contact_target_c


cdef uint8_t PHYSICAL = 1
cdef uint8_t COARSER = 2
cdef uint8_t SAME = 3
cdef uint8_t FINER = 4


cpdef void fill_balanced_refined_relations_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
    const int64_t[::1] leaf_ids,
    const int64_t[:, ::1] directions,
    uint8_t[:, ::1] relation_kinds,
    uint8_t[:, ::1] physical_masks,
    uint8_t[:, ::1] source_counts,
    int64_t[:, :, ::1] source_leaf_ids,
    uint8_t periodic_mask=0,
):
    if leaf_ids.shape[0] == 0 or directions.shape[0] == 0:
        return
    with nogil:
        _fill_balanced_refined_relations(
            root_shape,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            leaf_ids,
            directions,
            relation_kinds,
            physical_masks,
            source_counts,
            source_leaf_ids,
            periodic_mask,
        )


cdef void _fill_balanced_refined_relations(
    const int64_t[::1] root_shape,
    const int64_t[:, :, ::1] coord_to_rank,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
    const int64_t[::1] leaf_ids,
    const int64_t[:, ::1] directions,
    uint8_t[:, ::1] relation_kinds,
    uint8_t[:, ::1] physical_masks,
    uint8_t[:, ::1] source_counts,
    int64_t[:, :, ::1] source_leaf_ids,
    uint8_t periodic_mask,
) noexcept nogil:
    cdef int64_t primary, direction_index, source_leaf, source_node
    cdef int64_t source_level, scale, extent_x, extent_y, extent_z
    cdef int64_t target_node, target_leaf, child_node
    cdef int dx, dy, dz, rx, ry, rz, child, count, column
    cdef int xbit, ybit, zbit
    cdef uint8_t mask
    for primary in range(leaf_ids.shape[0]):
        source_leaf = leaf_ids[primary]
        source_node = leaf_node_ids[source_leaf]
        source_level = node_levels[source_node]
        scale = (<int64_t>1) << (source_level - 1)
        extent_x = root_shape[0] * scale
        extent_y = root_shape[1] * scale
        extent_z = root_shape[2] * scale
        for direction_index in range(directions.shape[0]):
            dx = <int>directions[direction_index, 0]
            dy = <int>directions[direction_index, 1]
            dz = <int>directions[direction_index, 2]
            rx = dx
            ry = dy
            rz = dz
            mask = 0
            if not (periodic_mask & 1) and (
                (dx < 0 and node_coords[source_node, 0] == 0)
                or (
                    dx > 0
                    and node_coords[source_node, 0] + 1 == extent_x
                )
            ):
                mask |= 1
                rx = 0
            if not (periodic_mask & 2) and (
                (dy < 0 and node_coords[source_node, 1] == 0)
                or (
                    dy > 0
                    and node_coords[source_node, 1] + 1 == extent_y
                )
            ):
                mask |= 2
                ry = 0
            if not (periodic_mask & 4) and (
                (dz < 0 and node_coords[source_node, 2] == 0)
                or (
                    dz > 0
                    and node_coords[source_node, 2] + 1 == extent_z
                )
            ):
                mask |= 4
                rz = 0
            physical_masks[primary, direction_index] = mask
            source_counts[primary, direction_index] = 0
            for column in range(4):
                source_leaf_ids[primary, direction_index, column] = -1

            if rx == 0 and ry == 0 and rz == 0:
                relation_kinds[primary, direction_index] = PHYSICAL
                continue

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
                rx,
                ry,
                rz,
                periodic_mask,
            )
            target_leaf = node_leaf_ids[target_node]
            if target_leaf >= 0:
                if node_levels[target_node] < source_level:
                    relation_kinds[primary, direction_index] = COARSER
                else:
                    relation_kinds[primary, direction_index] = SAME
                source_counts[primary, direction_index] = 1
                source_leaf_ids[primary, direction_index, 0] = target_leaf
                continue

            relation_kinds[primary, direction_index] = FINER
            count = 0
            for child in range(8):
                xbit = child & 1
                ybit = (child >> 1) & 1
                zbit = (child >> 2) & 1
                if rx < 0 and xbit != 1:
                    continue
                if rx > 0 and xbit != 0:
                    continue
                if ry < 0 and ybit != 1:
                    continue
                if ry > 0 and ybit != 0:
                    continue
                if rz < 0 and zbit != 1:
                    continue
                if rz > 0 and zbit != 0:
                    continue
                child_node = child_node_ids[target_node, child]
                source_leaf_ids[primary, direction_index, count] = (
                    node_leaf_ids[child_node]
                )
                count += 1
            source_counts[primary, direction_index] = <uint8_t>count
