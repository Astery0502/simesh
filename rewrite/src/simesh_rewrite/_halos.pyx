"""Direct physical-envelope fill for HAL-001."""

from libc.stdint cimport int64_t, uint8_t


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
