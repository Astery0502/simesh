"""Direct level-1 halo fills for HAL-001 and HAL-002."""

import cython

from libc.stdint cimport int64_t, uint8_t


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


cdef inline int64_t _find_slot(
    const int64_t[::1] block_ids,
    int64_t block_id,
):
    cdef int64_t slot
    for slot in range(block_ids.shape[0]):
        if block_ids[slot] == block_id:
            return slot
    return -1


cpdef int64_t duplicate_block_id_index_unchecked(
    const int64_t[::1] block_ids,
):
    cdef int64_t first, second
    for second in range(block_ids.shape[0]):
        for first in range(second):
            if block_ids[first] == block_ids[second]:
                return second
    return -1


cpdef int64_t missing_halo_closure_primary_unchecked(
    int64_t primary_count,
    const int64_t[::1] block_ids,
    const int64_t[:, ::1] face_neighbor_ids,
):
    cdef int64_t primary, block_id, source_block
    cdef int dx, dy, dz
    for primary in range(primary_count):
        block_id = block_ids[primary]
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    source_block = _walk_direction(
                        block_id,
                        dx,
                        dy,
                        dz,
                        face_neighbor_ids,
                    )
                    if source_block >= 0 and _find_slot(
                        block_ids,
                        source_block,
                    ) < 0:
                        return primary
    return -1


cpdef void common_physical_valid_region_unchecked(
    const int64_t[::1] spatial_shape,
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[::1] block_ids,
    const int64_t[:, ::1] face_neighbor_ids,
    int64_t[::1] common_lower,
    int64_t[::1] common_upper,
):
    cdef int64_t axis, slot, block_id
    cdef bint all_lower_physical, all_upper_physical
    for axis in range(3):
        common_lower[axis] = interior_lower[axis]
        common_upper[axis] = interior_upper[axis]
        all_lower_physical = True
        all_upper_physical = True
        for slot in range(block_ids.shape[0]):
            block_id = block_ids[slot]
            if face_neighbor_ids[block_id, 2 * axis] != -1:
                all_lower_physical = False
            if face_neighbor_ids[block_id, 2 * axis + 1] != -1:
                all_upper_physical = False
            if not all_lower_physical and not all_upper_physical:
                break
        if block_ids.shape[0] > 0 and all_lower_physical:
            common_lower[axis] = 0
        if block_ids.shape[0] > 0 and all_upper_physical:
            common_upper[axis] = spatial_shape[axis]


cpdef void fill_physical_halos_unchecked(
    double[:, :, :, :, ::1] payload,
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[::1] block_ids,
    const int64_t[:, ::1] face_neighbor_ids,
    const uint8_t[:, ::1] boundary_modes,
    const int64_t[::1] normal_field_slots,
):
    cdef int64_t slot, field, i, j, k, axis, block_id
    cdef int64_t target[3]
    cdef int64_t source[3]
    cdef int64_t face[3]
    cdef uint8_t mode[3]
    cdef bint outside[3]
    cdef bint physical
    cdef double value

    for slot in range(payload.shape[0]):
        block_id = block_ids[slot]
        for field in range(payload.shape[1]):
            for i in range(payload.shape[2]):
                target[0] = i
                for j in range(payload.shape[3]):
                    target[1] = j
                    for k in range(payload.shape[4]):
                        target[2] = k
                        physical = True
                        for axis in range(3):
                            outside[axis] = False
                            source[axis] = target[axis]
                            if target[axis] < interior_lower[axis]:
                                outside[axis] = True
                                face[axis] = 2 * axis
                            elif target[axis] >= interior_upper[axis]:
                                outside[axis] = True
                                face[axis] = 2 * axis + 1
                            if not outside[axis]:
                                continue
                            if face_neighbor_ids[block_id, face[axis]] != -1:
                                physical = False
                                break
                            mode[axis] = boundary_modes[field, face[axis]]
                            if mode[axis] == 1 or mode[axis] == 2:
                                if face[axis] % 2 == 0:
                                    source[axis] = (
                                        2 * interior_lower[axis]
                                        - target[axis]
                                        - 1
                                    )
                                else:
                                    source[axis] = (
                                        2 * interior_upper[axis]
                                        - target[axis]
                                        - 1
                                    )
                            elif face[axis] % 2 == 0:
                                source[axis] = interior_lower[axis]
                            else:
                                source[axis] = interior_upper[axis] - 1

                        if not physical or not (outside[0] or outside[1] or outside[2]):
                            continue

                        value = payload[
                            slot,
                            field,
                            source[0],
                            source[1],
                            source[2],
                        ]
                        for axis in range(3):
                            if not outside[axis]:
                                continue
                            if mode[axis] == 2:
                                value = -value
                            elif (
                                mode[axis] == 3
                                and field == normal_field_slots[axis]
                            ):
                                if face[axis] % 2 == 0 and value > 0.0:
                                    value = 0.0
                                elif face[axis] % 2 == 1 and value < 0.0:
                                    value = 0.0
                        payload[slot, field, i, j, k] = value


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fill_same_level_halos_unchecked(
    double[:, :, :, :, ::1] payload,
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[::1] block_ids,
    int64_t primary_count,
    const int64_t[:, ::1] face_neighbor_ids,
    const uint8_t[:, ::1] boundary_modes,
    const int64_t[::1] normal_field_slots,
):
    cdef int64_t primary, field, i, j, k, axis, block_id
    cdef int64_t source_block, source_slot, direction_index
    cdef int64_t target[3]
    cdef int64_t source[3]
    cdef int64_t displacement[3]
    cdef int64_t face[3]
    cdef int64_t source_slots[27]
    cdef uint8_t mode[3]
    cdef bint physical[3]
    cdef bint has_sibling
    cdef double value
    cdef int dx, dy, dz

    for primary in range(primary_count):
        block_id = block_ids[primary]
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    direction_index = (dz + 1) * 9 + (dy + 1) * 3 + dx + 1
                    source_block = _walk_direction(
                        block_id,
                        dx,
                        dy,
                        dz,
                        face_neighbor_ids,
                    )
                    if source_block < 0:
                        source_slots[direction_index] = -1
                    else:
                        source_slots[direction_index] = _find_slot(
                            block_ids,
                            source_block,
                        )

        for field in range(payload.shape[1]):
            for i in range(payload.shape[2]):
                target[0] = i
                for j in range(payload.shape[3]):
                    target[1] = j
                    for k in range(payload.shape[4]):
                        target[2] = k
                        has_sibling = False
                        for axis in range(3):
                            source[axis] = target[axis]
                            displacement[axis] = 0
                            physical[axis] = False
                            if target[axis] < interior_lower[axis]:
                                face[axis] = 2 * axis
                                if face_neighbor_ids[block_id, face[axis]] >= 0:
                                    displacement[axis] = -1
                                    source[axis] = (
                                        interior_upper[axis]
                                        - interior_lower[axis]
                                        + target[axis]
                                    )
                                    has_sibling = True
                                else:
                                    physical[axis] = True
                            elif target[axis] >= interior_upper[axis]:
                                face[axis] = 2 * axis + 1
                                if face_neighbor_ids[block_id, face[axis]] >= 0:
                                    displacement[axis] = 1
                                    source[axis] = (
                                        interior_lower[axis]
                                        + target[axis]
                                        - interior_upper[axis]
                                    )
                                    has_sibling = True
                                else:
                                    physical[axis] = True

                            if not physical[axis]:
                                continue
                            mode[axis] = boundary_modes[field, face[axis]]
                            if mode[axis] == 1 or mode[axis] == 2:
                                if face[axis] % 2 == 0:
                                    source[axis] = (
                                        2 * interior_lower[axis]
                                        - target[axis]
                                        - 1
                                    )
                                else:
                                    source[axis] = (
                                        2 * interior_upper[axis]
                                        - target[axis]
                                        - 1
                                    )
                            elif face[axis] % 2 == 0:
                                source[axis] = interior_lower[axis]
                            else:
                                source[axis] = interior_upper[axis] - 1

                        if not has_sibling:
                            continue

                        direction_index = (
                            (displacement[2] + 1) * 9
                            + (displacement[1] + 1) * 3
                            + displacement[0]
                            + 1
                        )
                        source_slot = source_slots[direction_index]
                        value = payload[
                            source_slot,
                            field,
                            source[0],
                            source[1],
                            source[2],
                        ]
                        for axis in range(3):
                            if not physical[axis]:
                                continue
                            if mode[axis] == 2:
                                value = -value
                            elif (
                                mode[axis] == 3
                                and field == normal_field_slots[axis]
                            ):
                                if face[axis] % 2 == 0 and value > 0.0:
                                    value = 0.0
                                elif face[axis] % 2 == 1 and value < 0.0:
                                    value = 0.0
                        payload[primary, field, i, j, k] = value
