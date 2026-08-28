# cython: boundscheck=False, wraparound=False

"""RPH-001 refined relation child-phase validation and fill."""

from libc.stdint cimport int64_t, uint8_t


cdef inline int64_t _floor_divide_two(int64_t value) noexcept nogil:
    if value >= 0:
        return value >> 1
    return -1 - ((-1 - value) >> 1)


cdef inline uint8_t _node_phase(
    const int64_t[:, ::1] node_coords,
    int64_t node,
) noexcept nogil:
    return <uint8_t>(
        (node_coords[node, 0] & 1)
        | ((node_coords[node, 1] & 1) << 1)
        | ((node_coords[node, 2] & 1) << 2)
    )


cpdef tuple validate_selected_phase_nodes_unchecked(
    const int64_t[::1] selected_leaf_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
):
    cdef int64_t slot, leaf_id, node, axis
    for slot in range(selected_leaf_ids.shape[0]):
        leaf_id = selected_leaf_ids[slot]
        node = leaf_node_ids[leaf_id]
        if node < 0 or node >= node_levels.shape[0]:
            return 1, slot, -1
        if node_levels[node] < 1:
            return 2, slot, -1
        for axis in range(3):
            if node_coords[node, axis] < 0:
                return 3, slot, axis
    return 0, -1, -1


cpdef tuple validate_refined_relation_phases_unchecked(
    const int64_t[::1] selected_leaf_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const int64_t[:, ::1] directions,
    const uint8_t[:, ::1] relation_kinds,
    const uint8_t[:, ::1] physical_masks,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_slots,
):
    cdef int64_t row, direction, source, axis, slot, node
    cdef int64_t primary_leaf, primary_node, primary_level
    cdef int64_t source_leaf, source_node, source_level
    cdef int64_t shifted, expected_coord, reduced
    cdef int64_t neutral_axes, expected_count
    cdef uint8_t kind, mask, count, phase, prior_phase
    cdef bint reduced_noncenter

    for direction in range(directions.shape[0]):
        reduced_noncenter = False
        for axis in range(3):
            if directions[direction, axis] < -1 or directions[direction, axis] > 1:
                return 1, -1, direction, -1, axis
            if directions[direction, axis] != 0:
                reduced_noncenter = True
        if not reduced_noncenter:
            return 2, -1, direction, -1, -1

    for row in range(relation_kinds.shape[0]):
        primary_leaf = selected_leaf_ids[row]
        primary_node = leaf_node_ids[primary_leaf]
        primary_level = node_levels[primary_node]
        for direction in range(relation_kinds.shape[1]):
            kind = relation_kinds[row, direction]
            mask = physical_masks[row, direction]
            count = source_counts[row, direction]
            if mask & <uint8_t>248:
                return 3, row, direction, -1, -1
            reduced_noncenter = False
            neutral_axes = 0
            for axis in range(3):
                if (mask & (<uint8_t>1 << axis)) and directions[direction, axis] == 0:
                    return 4, row, direction, -1, axis
                if (mask & (<uint8_t>1 << axis)) or directions[direction, axis] == 0:
                    neutral_axes += 1
                else:
                    reduced_noncenter = True
            if kind < 1 or kind > 4:
                return 5, row, direction, -1, -1
            if kind == 1:
                if reduced_noncenter or count != 0:
                    return 6, row, direction, -1, -1
            else:
                if not reduced_noncenter:
                    return 6, row, direction, -1, -1
                if kind == 4:
                    expected_count = 1 << neutral_axes
                    if count != expected_count or count > 4:
                        return 7, row, direction, -1, -1
                elif count != 1:
                    return 7, row, direction, -1, -1
            if count > 4:
                return 7, row, direction, -1, -1

            for source in range(4):
                slot = source_slots[row, direction, source]
                if source < count:
                    if slot < 0 or slot >= selected_leaf_ids.shape[0]:
                        return 8, row, direction, source, -1
                elif slot != -1:
                    return 9, row, direction, source, -1

            if kind == 1:
                continue
            slot = source_slots[row, direction, 0]
            source_leaf = selected_leaf_ids[slot]
            source_node = leaf_node_ids[source_leaf]
            source_level = node_levels[source_node]
            if kind == 3:
                if source_level != primary_level:
                    return 10, row, direction, 0, -1
                continue
            if kind == 2:
                if primary_level <= source_level or primary_level - source_level != 1:
                    return 10, row, direction, 0, -1
                for axis in range(3):
                    reduced = 0 if (mask & (<uint8_t>1 << axis)) else directions[direction, axis]
                    if reduced > 0 and node_coords[primary_node, axis] == 9223372036854775807:
                        return 11, row, direction, 0, axis
                    shifted = node_coords[primary_node, axis] + reduced
                    expected_coord = _floor_divide_two(shifted)
                    if node_coords[source_node, axis] != expected_coord:
                        return 11, row, direction, 0, axis
                continue

            prior_phase = 0
            for source in range(count):
                slot = source_slots[row, direction, source]
                source_leaf = selected_leaf_ids[slot]
                source_node = leaf_node_ids[source_leaf]
                source_level = node_levels[source_node]
                if source_level <= primary_level or source_level - primary_level != 1:
                    return 10, row, direction, source, -1
                phase = _node_phase(node_coords, source_node)
                for axis in range(3):
                    if mask & (<uint8_t>1 << axis):
                        continue
                    if directions[direction, axis] < 0 and not (phase & (<uint8_t>1 << axis)):
                        return 12, row, direction, source, axis
                    if directions[direction, axis] > 0 and (phase & (<uint8_t>1 << axis)):
                        return 12, row, direction, source, axis
                if source > 0 and phase <= prior_phase:
                    return 13, row, direction, source, -1
                prior_phase = phase
    return 0, -1, -1, -1, -1


cpdef void fill_refined_relation_phase_codes_unchecked(
    const int64_t[::1] selected_leaf_ids,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] leaf_node_ids,
    const uint8_t[:, ::1] relation_kinds,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_slots,
    uint8_t[:, :, ::1] fine_phase_codes,
):
    cdef int64_t row, direction, source, slot, leaf_id, node
    cdef uint8_t kind, count
    for row in range(relation_kinds.shape[0]):
        for direction in range(relation_kinds.shape[1]):
            for source in range(4):
                fine_phase_codes[row, direction, source] = 255
            kind = relation_kinds[row, direction]
            count = source_counts[row, direction]
            if kind == 2:
                leaf_id = selected_leaf_ids[row]
                node = leaf_node_ids[leaf_id]
                fine_phase_codes[row, direction, 0] = _node_phase(
                    node_coords, node
                )
            elif kind == 4:
                for source in range(count):
                    slot = source_slots[row, direction, source]
                    leaf_id = selected_leaf_ids[slot]
                    node = leaf_node_ids[leaf_id]
                    fine_phase_codes[row, direction, source] = _node_phase(
                        node_coords, node
                    )
