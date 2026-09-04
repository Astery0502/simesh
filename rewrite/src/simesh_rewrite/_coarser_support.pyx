# cython: boundscheck=False, wraparound=False

"""CSP-001 fixed-capacity COARSER slope-support plan fill."""

from libc.stdint cimport int64_t, uint8_t


cdef inline int64_t _maximum(int64_t left, int64_t right) noexcept nogil:
    return left if left >= right else right


cdef inline int64_t _minimum(int64_t left, int64_t right) noexcept nogil:
    return left if left <= right else right


cdef inline int64_t _floor_divide_two(int64_t value) noexcept nogil:
    if value >= 0:
        return value >> 1
    return -1 - ((-1 - value) >> 1)


cdef inline int64_t _direction_row(int64_t column) noexcept nogil:
    return column if column < 13 else column - 1


cpdef tuple fill_coarser_slope_support_plan_unchecked(
    const int64_t[::1] interior_lower,
    const int64_t[::1] interior_upper,
    int64_t primary_slot,
    uint8_t primary_phase_code,
    const int64_t[::1] reduced_direction,
    const uint8_t[::1] relation_kinds,
    const uint8_t[::1] physical_masks,
    const int64_t[:, ::1] source_slots,
    const int64_t[::1] coarse_source_lower,
    const int64_t[::1] coarse_source_upper,
    const int64_t[::1] workspace_source_lower,
    const int64_t[::1] workspace_source_upper,
    const int64_t[::1] workspace_required_lower,
    const int64_t[::1] workspace_required_upper,
    const int64_t[::1] workspace_coarse_origin,
    int64_t[::1] plan_source_slots,
    uint8_t[::1] plan_source_is_fine,
    uint8_t[::1] plan_physical_masks,
    int64_t[:, ::1] plan_directions,
    int64_t[:, ::1] plan_source_lower,
    int64_t[:, ::1] plan_source_upper,
    int64_t[:, ::1] plan_base_lower,
    int64_t[:, ::1] plan_base_upper,
    int64_t[:, ::1] plan_target_lower,
    int64_t[:, ::1] plan_target_upper,
    int64_t[:, ::1] plan_logical_interior_lower,
    int64_t[:, ::1] plan_logical_interior_upper,
    int64_t[:, ::1] plan_storage_logical_offsets,
):
    cdef int64_t raw_lower[27][3]
    cdef int64_t raw_upper[27][3]
    cdef uint8_t raw_active[27]
    cdef int64_t block[3]
    cdef int64_t half[3]
    cdef int64_t phase_bits[3]
    cdef int64_t column, axis, component, direction_row
    cdef int64_t start, stop, interval_lower, interval_upper
    cdef int64_t record, owner, value, source_origin
    cdef int64_t transfer_count, record_count, reduced_column
    cdef uint8_t mask, kind, flag
    cdef bint nonempty, inside_workspace, physical

    for record in range(18):
        plan_source_slots[record] = -1
        plan_source_is_fine[record] = 255
        plan_physical_masks[record] = 0
        for axis in range(3):
            plan_directions[record, axis] = 0
            plan_source_lower[record, axis] = 0
            plan_source_upper[record, axis] = 0
            plan_base_lower[record, axis] = 0
            plan_base_upper[record, axis] = 0
            plan_target_lower[record, axis] = 0
            plan_target_upper[record, axis] = 0
            plan_logical_interior_lower[record, axis] = 0
            plan_logical_interior_upper[record, axis] = 0
            plan_storage_logical_offsets[record, axis] = 0

    for axis in range(3):
        block[axis] = interior_upper[axis] - interior_lower[axis]
        half[axis] = block[axis] // 2
        phase_bits[axis] = (primary_phase_code >> axis) & 1

    for column in range(27):
        raw_active[column] = 0
        nonempty = True
        inside_workspace = True
        value = column
        for axis in range(3):
            component = value % 3 - 1
            value //= 3
            interval_lower = (
                workspace_coarse_origin[axis] + component * half[axis]
            )
            interval_upper = interval_lower + half[axis]
            start = _maximum(workspace_required_lower[axis], interval_lower)
            stop = _minimum(workspace_required_upper[axis], interval_upper)
            raw_lower[column][axis] = start
            raw_upper[column][axis] = stop
            if start >= stop:
                nonempty = False
            if (
                start < workspace_source_lower[axis]
                or stop > workspace_source_upper[axis]
            ):
                inside_workspace = False
        if nonempty and not inside_workspace:
            raw_active[column] = 1

    record = 0
    reduced_column = (
        (reduced_direction[2] + 1) * 9
        + (reduced_direction[1] + 1) * 3
        + reduced_direction[0]
        + 1
    )
    direction_row = _direction_row(reduced_column)
    plan_source_slots[record] = source_slots[direction_row, 0]
    plan_source_is_fine[record] = 0
    for axis in range(3):
        plan_directions[record, axis] = reduced_direction[axis]
        plan_source_lower[record, axis] = coarse_source_lower[axis]
        plan_source_upper[record, axis] = coarse_source_upper[axis]
        plan_base_lower[record, axis] = workspace_source_lower[axis]
        plan_base_upper[record, axis] = workspace_source_upper[axis]
        plan_target_lower[record, axis] = workspace_source_lower[axis]
        plan_target_upper[record, axis] = workspace_source_upper[axis]
    record += 1

    for column in range(27):
        if not raw_active[column]:
            continue
        if column == 13:
            mask = 0
            kind = 0
            direction_row = -1
        else:
            direction_row = _direction_row(column)
            mask = physical_masks[direction_row]
            kind = relation_kinds[direction_row]
        if mask != 0:
            continue

        plan_physical_masks[record] = 0
        value = column
        for axis in range(3):
            component = value % 3 - 1
            value //= 3
            plan_directions[record, axis] = component
            plan_target_lower[record, axis] = raw_lower[column][axis]
            plan_target_upper[record, axis] = raw_upper[column][axis]
            plan_base_lower[record, axis] = raw_lower[column][axis]
            plan_base_upper[record, axis] = raw_upper[column][axis]

        if kind == 0:
            plan_source_slots[record] = primary_slot
            plan_source_is_fine[record] = 1
        elif kind == 3:
            plan_source_slots[record] = source_slots[direction_row, 0]
            plan_source_is_fine[record] = 1
        else:
            plan_source_slots[record] = source_slots[direction_row, 0]
            plan_source_is_fine[record] = 0

        flag = plan_source_is_fine[record]
        for axis in range(3):
            component = plan_directions[record, axis]
            if flag == 1:
                source_origin = (
                    workspace_coarse_origin[axis]
                    + component * half[axis]
                )
                plan_source_lower[record, axis] = (
                    interior_lower[axis]
                    + 2 * (
                        plan_target_lower[record, axis]
                        - source_origin
                    )
                )
                plan_source_upper[record, axis] = (
                    interior_lower[axis]
                    + 2 * (
                        plan_target_upper[record, axis]
                        - source_origin
                    )
                )
            else:
                value = phase_bits[axis] + component
                source_origin = (
                    interior_lower[axis]
                    + (
                        value - 2 * _floor_divide_two(value)
                    ) * half[axis]
                )
                start = (
                    workspace_coarse_origin[axis]
                    + component * half[axis]
                )
                plan_source_lower[record, axis] = source_origin + (
                    plan_target_lower[record, axis]
                    - start
                )
                plan_source_upper[record, axis] = source_origin + (
                    plan_target_upper[record, axis]
                    - start
                )
        record += 1

    transfer_count = record

    for column in range(27):
        if not raw_active[column] or column == 13:
            continue
        direction_row = _direction_row(column)
        mask = physical_masks[direction_row]
        if mask == 0:
            continue
        plan_physical_masks[record] = mask
        value = column
        for axis in range(3):
            component = value % 3 - 1
            value //= 3
            plan_directions[record, axis] = component
            plan_target_lower[record, axis] = raw_lower[column][axis]
            plan_target_upper[record, axis] = raw_upper[column][axis]
            if mask & (<uint8_t>1 << axis):
                plan_base_lower[record, axis] = (
                    raw_lower[column][axis] - component
                )
                plan_base_upper[record, axis] = (
                    raw_upper[column][axis] - component
                )
            else:
                plan_base_lower[record, axis] = raw_lower[column][axis]
                plan_base_upper[record, axis] = raw_upper[column][axis]

        owner = -1
        for value in range(transfer_count):
            inside_workspace = True
            for axis in range(3):
                if (
                    plan_base_lower[record, axis]
                    < plan_target_lower[value, axis]
                    or plan_base_upper[record, axis]
                    > plan_target_upper[value, axis]
                ):
                    inside_workspace = False
                    break
            if inside_workspace:
                owner = value
                break

        if plan_source_is_fine[owner] == 1:
            for axis in range(3):
                plan_logical_interior_lower[record, axis] = 0
                plan_logical_interior_upper[record, axis] = half[axis]
                plan_storage_logical_offsets[record, axis] = (
                    (
                        plan_source_lower[owner, axis]
                        - interior_lower[axis]
                    )
                    // 2
                    - plan_target_lower[owner, axis]
                )
        else:
            for axis in range(3):
                plan_logical_interior_lower[record, axis] = interior_lower[axis]
                plan_logical_interior_upper[record, axis] = interior_upper[axis]
                plan_storage_logical_offsets[record, axis] = (
                    plan_source_lower[owner, axis]
                    - plan_target_lower[owner, axis]
                )
        record += 1

    record_count = record
    return transfer_count, record_count
