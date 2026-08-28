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


cdef inline int64_t _walk_direction(
    int64_t block_id,
    int dx,
    int dy,
    int dz,
    const int64_t[:, ::1] face_neighbor_ids,
):
    if dx != 0:
        block_id = face_neighbor_ids[block_id, 0 if dx < 0 else 1]
        if block_id < 0:
            return -1
    if dy != 0:
        block_id = face_neighbor_ids[block_id, 2 if dy < 0 else 3]
        if block_id < 0:
            return -1
    if dz != 0:
        block_id = face_neighbor_ids[block_id, 4 if dz < 0 else 5]
    return block_id


cpdef int64_t minimum_halo_closed_slots_unchecked(
    const int64_t[:, ::1] face_neighbor_ids,
):
    cdef int64_t block_id, neighbor, count, maximum = 0
    cdef int dx, dy, dz
    for block_id in range(face_neighbor_ids.shape[0]):
        count = 0
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor = _walk_direction(
                        block_id,
                        dx,
                        dy,
                        dz,
                        face_neighbor_ids,
                    )
                    if neighbor >= 0:
                        count += 1
        if count > maximum:
            maximum = count
    return maximum


cpdef int64_t minimum_level1_halo_closed_slots_unchecked(
    const int64_t[:, ::1] face_neighbor_ids,
):
    return minimum_halo_closed_slots_unchecked(face_neighbor_ids)


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

    return plan_direct_face_closed_prefix_unchecked(
        first_primary_id,
        face_neighbor_ids,
        chunk_block_ids,
    )


cpdef tuple plan_direct_face_closed_prefix_unchecked(
    int64_t first_primary_id,
    const int64_t[:, ::1] face_neighbor_ids,
    int64_t[::1] chunk_block_ids,
):
    cdef int64_t block_count = face_neighbor_ids.shape[0]
    cdef int64_t capacity = chunk_block_ids.shape[0]
    cdef int64_t primary_count = 0
    cdef int64_t selected_count = 0
    cdef int64_t candidate, face, neighbor, missing, position, index

    if first_primary_id == block_count:
        return 0, 0

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


cpdef tuple plan_level1_halo_chunk_unchecked(
    int64_t first_primary_id,
    const int64_t[:, ::1] face_neighbor_ids,
    int64_t[::1] chunk_block_ids,
):
    return plan_level1_halo_closed_prefix_unchecked(
        first_primary_id,
        face_neighbor_ids,
        chunk_block_ids,
    )


cpdef tuple plan_level1_halo_closed_prefix_unchecked(
    int64_t first_primary_id,
    const int64_t[:, ::1] face_neighbor_ids,
    int64_t[::1] chunk_block_ids,
):
    cdef int64_t block_count = face_neighbor_ids.shape[0]
    cdef int64_t capacity = chunk_block_ids.shape[0]
    cdef int64_t primary_count = 0
    cdef int64_t selected_count = 0
    cdef int64_t candidate, neighbor, missing, position, index, new_count
    cdef int64_t new_support[26]
    cdef int dx, dy, dz

    if first_primary_id == block_count:
        return 0, 0

    while first_primary_id + primary_count < block_count:
        candidate = first_primary_id + primary_count
        position = _find_support_id(
            chunk_block_ids,
            primary_count,
            selected_count,
            candidate,
        )
        missing = 1 if position < 0 else 0
        new_count = 0
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = _walk_direction(
                        candidate,
                        dx,
                        dy,
                        dz,
                        face_neighbor_ids,
                    )
                    if neighbor >= 0 and not _is_selected(
                        first_primary_id,
                        primary_count,
                        chunk_block_ids,
                        selected_count,
                        neighbor,
                    ):
                        missing += 1
                        new_support[new_count] = neighbor
                        new_count += 1

        if selected_count + missing > capacity:
            if primary_count == 0:
                raise ValueError("chunk capacity cannot fit the first halo closure")
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

        for index in range(new_count):
            chunk_block_ids[selected_count] = new_support[index]
            selected_count += 1

    return primary_count, selected_count
