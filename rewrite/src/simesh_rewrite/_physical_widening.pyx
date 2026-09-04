# cython: boundscheck=False, wraparound=False

"""Allocation-free PWA-001 Cartesian physical widening kernel."""

from libc.stdint cimport int64_t, uint8_t

from ._boundary_rules cimport (
    physical_halo_source_index_c,
    transform_physical_halo_value_c,
)


cpdef void apply_cartesian_physical_widening_unchecked(
    double[:, :, :, :, ::1] payload,
    int64_t target_slot,
    const int64_t[:, ::1] logical_interior_lower,
    const int64_t[:, ::1] logical_interior_upper,
    const int64_t[:, ::1] storage_logical_offsets,
    const int64_t[:, ::1] directions,
    const uint8_t[::1] physical_masks,
    const int64_t[:, ::1] base_lower,
    const int64_t[:, ::1] base_upper,
    const int64_t[:, ::1] target_lower,
    const int64_t[:, ::1] target_upper,
    const uint8_t[:, ::1] boundary_modes,
    const int64_t[::1] normal_field_slots,
):
    cdef int64_t row, field, i, j, k, axis
    cdef int64_t target[3]
    cdef int64_t source[3]
    cdef int64_t face[3]
    cdef uint8_t mode[3]
    cdef uint8_t mask
    cdef double value

    # Base boxes are a checked-boundary validity promise.  The narrowed
    # complete-base contract therefore needs no ownership branch here.
    for row in range(directions.shape[0]):
        mask = physical_masks[row]
        for axis in range(3):
            if (mask & (1 << axis)) != 0:
                face[axis] = 2 * axis + (
                    1 if directions[row, axis] > 0 else 0
                )
        for field in range(payload.shape[1]):
            for axis in range(3):
                if (mask & (1 << axis)) != 0:
                    mode[axis] = boundary_modes[field, face[axis]]
            for i in range(target_lower[row, 0], target_upper[row, 0]):
                target[0] = i
                for j in range(target_lower[row, 1], target_upper[row, 1]):
                    target[1] = j
                    for k in range(target_lower[row, 2], target_upper[row, 2]):
                        target[2] = k
                        for axis in range(3):
                            source[axis] = target[axis]
                            if (mask & (1 << axis)) == 0:
                                continue
                            source[axis] = physical_halo_source_index_c(
                                target[axis] + storage_logical_offsets[row, axis],
                                logical_interior_lower[row, axis],
                                logical_interior_upper[row, axis],
                                face[axis],
                                mode[axis],
                            ) - storage_logical_offsets[row, axis]

                        value = payload[
                            target_slot,
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
                        payload[target_slot, field, i, j, k] = value
