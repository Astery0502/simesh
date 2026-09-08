# cython: boundscheck=False, wraparound=False

"""RSL-001 accepted refined relation source-slot resolution."""

from libc.stdint cimport int64_t, uint8_t


cdef inline int64_t _find_selected_slot(
    const int64_t[::1] selected_leaf_ids,
    int64_t leaf_id,
) noexcept nogil:
    cdef int64_t slot
    for slot in range(selected_leaf_ids.shape[0]):
        if selected_leaf_ids[slot] == leaf_id:
            return slot
    return -1


cpdef int64_t duplicate_selected_leaf_id_unchecked(
    const int64_t[::1] selected_leaf_ids,
):
    cdef int64_t first, second
    for second in range(selected_leaf_ids.shape[0]):
        for first in range(second):
            if selected_leaf_ids[first] == selected_leaf_ids[second]:
                return second
    return -1


cpdef int64_t missing_refined_relation_source_unchecked(
    const int64_t[::1] selected_leaf_ids,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
):
    cdef int64_t primary, direction, source, leaf_id
    for primary in range(source_counts.shape[0]):
        for direction in range(source_counts.shape[1]):
            for source in range(source_counts[primary, direction]):
                leaf_id = source_leaf_ids[primary, direction, source]
                if _find_selected_slot(selected_leaf_ids, leaf_id) < 0:
                    return (
                        primary * source_counts.shape[1] * 4
                        + direction * 4
                        + source
                    )
    return -1


cpdef int64_t count_relation_slot_comparisons_unchecked(
    const int64_t[::1] selected_leaf_ids,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
):
    cdef int64_t primary, direction, source, slot, leaf_id
    cdef int64_t comparisons = 0
    for primary in range(source_counts.shape[0]):
        for direction in range(source_counts.shape[1]):
            for source in range(source_counts[primary, direction]):
                leaf_id = source_leaf_ids[primary, direction, source]
                for slot in range(selected_leaf_ids.shape[0]):
                    comparisons += 1
                    if selected_leaf_ids[slot] == leaf_id:
                        break
    return comparisons


cpdef void resolve_refined_relation_source_slots_unchecked(
    const int64_t[::1] selected_leaf_ids,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
    int64_t[:, :, ::1] source_slots,
):
    cdef int64_t primary, direction, source, count, leaf_id
    for primary in range(source_counts.shape[0]):
        for direction in range(source_counts.shape[1]):
            count = source_counts[primary, direction]
            for source in range(count):
                leaf_id = source_leaf_ids[primary, direction, source]
                source_slots[primary, direction, source] = _find_selected_slot(
                    selected_leaf_ids,
                    leaf_id,
                )
            for source in range(count, 4):
                source_slots[primary, direction, source] = -1
