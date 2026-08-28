"""Typed HAX-001 application of explicit level-1 halo plans."""

import cython

from libc.stdint cimport int64_t, uint8_t

from ._boundary_rules cimport (
    physical_halo_source_index_c,
    transform_physical_halo_value_c,
)


cpdef int64_t invalid_level1_halo_plan_entry_unchecked(
    const int64_t[:, ::1] source_slots,
    const uint8_t[:, ::1] physical_masks,
    int64_t selected_count,
):
    cdef int64_t primary, column, source_slot
    cdef int dx, dy, dz
    cdef uint8_t mask, direction_mask
    for primary in range(source_slots.shape[0]):
        for column in range(27):
            dx = column % 3 - 1
            dy = (column // 3) % 3 - 1
            dz = column // 9 - 1
            direction_mask = 0
            if dx != 0:
                direction_mask |= 1
            if dy != 0:
                direction_mask |= 2
            if dz != 0:
                direction_mask |= 4
            source_slot = source_slots[primary, column]
            mask = physical_masks[primary, column]
            if source_slot < -1 or source_slot >= selected_count:
                return primary * 27 + column
            if mask > 7 or (mask & (~direction_mask)) != 0:
                return primary * 27 + column
            if source_slot < 0:
                if mask != direction_mask:
                    return primary * 27 + column
            elif (direction_mask & (~mask)) == 0:
                return primary * 27 + column
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void apply_level1_same_level_halo_plan_unchecked(
    double[:, :, :, :, ::1] payload,
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    const int64_t[:, ::1] source_slots,
    const uint8_t[:, ::1] physical_masks,
    const uint8_t[:, ::1] boundary_modes,
    const int64_t[::1] normal_field_slots,
):
    cdef int64_t primary, field, i, j, k, axis
    cdef int64_t source_slot, column, layer_or_offset
    cdef int64_t target[3]
    cdef int64_t source[3]
    cdef int64_t face[3]
    cdef int direction[3]
    cdef uint8_t mode[3]
    cdef uint8_t mask
    cdef double value

    for primary in range(source_slots.shape[0]):
        for field in range(payload.shape[1]):
            for i in range(payload.shape[2]):
                target[0] = i
                for j in range(payload.shape[3]):
                    target[1] = j
                    for k in range(payload.shape[4]):
                        target[2] = k
                        for axis in range(3):
                            source[axis] = target[axis]
                            direction[axis] = 0
                            if target[axis] < interior_lower[axis]:
                                direction[axis] = -1
                            elif target[axis] >= interior_upper[axis]:
                                direction[axis] = 1

                        column = (
                            (direction[2] + 1) * 9
                            + (direction[1] + 1) * 3
                            + direction[0]
                            + 1
                        )
                        source_slot = source_slots[primary, column]
                        if source_slot < 0:
                            continue
                        mask = physical_masks[primary, column]

                        for axis in range(3):
                            if direction[axis] == 0:
                                continue
                            face[axis] = (
                                2 * axis + (1 if direction[axis] > 0 else 0)
                            )
                            if (mask & (1 << axis)) != 0:
                                mode[axis] = boundary_modes[field, face[axis]]
                                source[axis] = physical_halo_source_index_c(
                                    target[axis],
                                    interior_lower[axis],
                                    interior_upper[axis],
                                    face[axis],
                                    mode[axis],
                                )
                            elif direction[axis] < 0:
                                layer_or_offset = (
                                    interior_lower[axis] - target[axis]
                                )
                                source[axis] = (
                                    interior_upper[axis] - layer_or_offset
                                )
                            else:
                                layer_or_offset = (
                                    target[axis] - interior_upper[axis]
                                )
                                source[axis] = (
                                    interior_lower[axis] + layer_or_offset
                                )

                        value = payload[
                            source_slot,
                            field,
                            source[0],
                            source[1],
                            source[2],
                        ]
                        for axis in range(3):
                            if (mask & (1 << axis)) == 0:
                                continue
                            value = transform_physical_halo_value_c(
                                value,
                                field,
                                normal_field_slots[axis],
                                face[axis],
                                mode[axis],
                            )
                        payload[primary, field, i, j, k] = value
