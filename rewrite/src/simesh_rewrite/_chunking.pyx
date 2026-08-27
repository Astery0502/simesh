"""Deterministic bounded chunk planning for STO-002."""

from libc.stdint cimport int64_t


cdef inline int64_t _find_support_id(
    const int64_t[::1] chunk_ids,
    int64_t primary_count,
    int64_t selected_count,
    int64_t value,
):
    cdef int64_t position
    for position in range(primary_count, selected_count):
        if chunk_ids[position] == value:
            return position
    return -1


cdef inline bint _is_selected(
    int64_t first_primary_id,
    int64_t primary_count,
    const int64_t[::1] chunk_ids,
    int64_t selected_count,
    int64_t value,
):
    if first_primary_id <= value < first_primary_id + primary_count:
        return True
    return _find_support_id(
        chunk_ids,
        primary_count,
        selected_count,
        value,
    ) >= 0


cpdef int64_t minimum_face_closed_slots_unchecked(
    const int64_t[:, ::1] face_neighbor_ids,
):
    cdef int64_t block_id, face, count, maximum = 0
    for block_id in range(face_neighbor_ids.shape[0]):
        count = 1
        for face in range(6):
            if face_neighbor_ids[block_id, face] >= 0:
                count += 1
        if count > maximum:
            maximum = count
    return maximum


cpdef tuple plan_level1_chunk_unchecked(
    int64_t first_primary_id,
    const int64_t[:, ::1] face_neighbor_ids,
    bint include_face_closure,
    int64_t[::1] chunk_block_ids,
):
    cdef int64_t block_count = face_neighbor_ids.shape[0]
    cdef int64_t capacity = chunk_block_ids.shape[0]
    cdef int64_t primary_count = 0
    cdef int64_t selected_count = 0
    cdef int64_t candidate, face, neighbor, missing, position, index

    if first_primary_id == block_count:
        return 0, 0

    if not include_face_closure:
        primary_count = min(capacity, block_count - first_primary_id)
        for index in range(primary_count):
            chunk_block_ids[index] = first_primary_id + index
        return primary_count, primary_count

    while first_primary_id + primary_count < block_count:
        candidate = first_primary_id + primary_count
        position = _find_support_id(
            chunk_block_ids,
            primary_count,
            selected_count,
            candidate,
        )
        missing = 1 if position < 0 else 0
        for face in range(6):
            neighbor = face_neighbor_ids[candidate, face]
            if neighbor >= 0 and not _is_selected(
                first_primary_id,
                primary_count,
                chunk_block_ids,
                selected_count,
                neighbor,
            ):
                missing += 1

        if selected_count + missing > capacity:
            if primary_count == 0:
                raise ValueError("chunk capacity cannot fit the first primary closure")
            break

        if position >= primary_count:
            for index in range(position, selected_count - 1):
                chunk_block_ids[index] = chunk_block_ids[index + 1]
            selected_count -= 1

        for index in range(selected_count, primary_count, -1):
            chunk_block_ids[index] = chunk_block_ids[index - 1]
        chunk_block_ids[primary_count] = candidate
        primary_count += 1
        selected_count += 1

        for face in range(6):
            neighbor = face_neighbor_ids[candidate, face]
            if neighbor >= 0 and not _is_selected(
                first_primary_id,
                primary_count,
                chunk_block_ids,
                selected_count,
                neighbor,
            ):
                chunk_block_ids[selected_count] = neighbor
                selected_count += 1

    return primary_count, selected_count
