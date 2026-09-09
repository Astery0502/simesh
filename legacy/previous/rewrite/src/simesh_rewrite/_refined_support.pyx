# cython: boundscheck=False, wraparound=False

"""STO-004 refined relation-source union and prefix planning."""

from libc.stdint cimport int64_t, uint8_t


cdef inline int64_t _find_support_id(
    const int64_t[::1] selected_ids,
    int64_t primary_count,
    int64_t selected_count,
    int64_t value,
) noexcept nogil:
    cdef int64_t position
    for position in range(primary_count, selected_count):
        if selected_ids[position] == value:
            return position
    return -1


cdef inline bint _is_selected(
    int64_t first_primary_id,
    int64_t primary_count,
    const int64_t[::1] selected_ids,
    int64_t selected_count,
    int64_t value,
) noexcept nogil:
    if first_primary_id <= value < first_primary_id + primary_count:
        return True
    return _find_support_id(
        selected_ids,
        primary_count,
        selected_count,
        value,
    ) >= 0


cdef inline bint _is_explicitly_selected(
    const int64_t[::1] selected_ids,
    int64_t selected_count,
    int64_t value,
) noexcept nogil:
    cdef int64_t position
    for position in range(selected_count):
        if selected_ids[position] == value:
            return True
    return False


cdef inline bint _source_appeared_before(
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
    int64_t row,
    int64_t direction,
    int64_t source_column,
    int64_t value,
) noexcept nogil:
    cdef int64_t prior_direction, prior_source, limit
    for prior_direction in range(direction + 1):
        limit = source_counts[row, prior_direction]
        if prior_direction == direction:
            limit = source_column
        for prior_source in range(limit):
            if source_leaf_ids[row, prior_direction, prior_source] == value:
                return True
    return False


cdef inline bint _fixed_list_contains(
    const int64_t* values,
    int64_t count,
    int64_t value,
) noexcept nogil:
    cdef int64_t index
    for index in range(count):
        if values[index] == value:
            return True
    return False


cpdef tuple validate_refined_support_rows_unchecked(
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
    int64_t leaf_count,
):
    cdef int64_t row, direction, source, value, count
    for row in range(source_counts.shape[0]):
        for direction in range(source_counts.shape[1]):
            count = source_counts[row, direction]
            if count > 4:
                return 1, row, direction, 0
            for source in range(4):
                value = source_leaf_ids[row, direction, source]
                if source < count:
                    if value < 0 or value >= leaf_count:
                        return 2, row, direction, source
                elif value != -1:
                    return 3, row, direction, source
    return 0, -1, -1, -1


cpdef int64_t invalid_selected_primary_order_unchecked(
    const int64_t[::1] primary_leaf_ids,
    int64_t leaf_count,
):
    cdef int64_t index, value, previous = -1
    for index in range(primary_leaf_ids.shape[0]):
        value = primary_leaf_ids[index]
        if value < 0 or value >= leaf_count:
            return index
        if index > 0 and value <= previous:
            return index
        previous = value
    return -1


cpdef int64_t maximum_balanced_refined_support_slots_unchecked(
    int64_t first_primary_id,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
):
    cdef int64_t row, direction, source, value, candidate, count, maximum = 0
    cdef int64_t unique_count
    cdef int64_t unique_sources[56]
    cdef bint use_fixed
    for row in range(source_counts.shape[0]):
        candidate = first_primary_id + row
        unique_count = 0
        use_fixed = True
        for direction in range(source_counts.shape[1]):
            for source in range(source_counts[row, direction]):
                value = source_leaf_ids[row, direction, source]
                if (
                    value != candidate
                    and not _fixed_list_contains(
                        &unique_sources[0], unique_count, value
                    )
                ):
                    if unique_count == 56:
                        use_fixed = False
                    else:
                        unique_sources[unique_count] = value
                        unique_count += 1
        if use_fixed:
            count = 1 + unique_count
        else:
            count = 1
            for direction in range(source_counts.shape[1]):
                for source in range(source_counts[row, direction]):
                    value = source_leaf_ids[row, direction, source]
                    if value != candidate and not _source_appeared_before(
                        source_counts,
                        source_leaf_ids,
                        row,
                        direction,
                        source,
                        value,
                    ):
                        count += 1
        if count > maximum:
            maximum = count
    return maximum


cpdef tuple plan_balanced_refined_support_prefix_unchecked(
    int64_t first_primary_id,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
    int64_t[::1] selected_leaf_ids,
):
    cdef int64_t capacity = selected_leaf_ids.shape[0]
    cdef int64_t primary_count = 0
    cdef int64_t selected_count = 0
    cdef int64_t row, candidate, position, missing, direction, source, value
    cdef int64_t index, new_count
    cdef int64_t new_support[56]
    cdef bint use_fixed

    while primary_count < source_counts.shape[0]:
        row = primary_count
        candidate = first_primary_id + primary_count
        position = _find_support_id(
            selected_leaf_ids,
            primary_count,
            selected_count,
            candidate,
        )
        missing = 1 if position < 0 else 0
        new_count = 0
        use_fixed = True
        for direction in range(source_counts.shape[1]):
            for source in range(source_counts[row, direction]):
                value = source_leaf_ids[row, direction, source]
                if (
                    value != candidate
                    and not _is_selected(
                        first_primary_id,
                        primary_count,
                        selected_leaf_ids,
                        selected_count,
                        value,
                    )
                    and not _fixed_list_contains(
                        &new_support[0], new_count, value
                    )
                ):
                    if new_count == 56:
                        use_fixed = False
                    else:
                        new_support[new_count] = value
                        new_count += 1

        if use_fixed:
            missing += new_count
        else:
            missing = 1 if position < 0 else 0
            for direction in range(source_counts.shape[1]):
                for source in range(source_counts[row, direction]):
                    value = source_leaf_ids[row, direction, source]
                    if (
                        value != candidate
                        and not _source_appeared_before(
                            source_counts,
                            source_leaf_ids,
                            row,
                            direction,
                            source,
                            value,
                        )
                        and not _is_selected(
                            first_primary_id,
                            primary_count,
                            selected_leaf_ids,
                            selected_count,
                            value,
                        )
                    ):
                        missing += 1

        if selected_count + missing > capacity:
            if primary_count == 0:
                raise ValueError(
                    "selected capacity cannot fit the first refined support closure"
                )
            break

        if position >= primary_count:
            for index in range(position, selected_count - 1):
                selected_leaf_ids[index] = selected_leaf_ids[index + 1]
            selected_count -= 1

        for index in range(selected_count, primary_count, -1):
            selected_leaf_ids[index] = selected_leaf_ids[index - 1]
        selected_leaf_ids[primary_count] = candidate
        primary_count += 1
        selected_count += 1

        if use_fixed:
            for index in range(new_count):
                selected_leaf_ids[selected_count] = new_support[index]
                selected_count += 1
        else:
            for direction in range(source_counts.shape[1]):
                for source in range(source_counts[row, direction]):
                    value = source_leaf_ids[row, direction, source]
                    if (
                        value != candidate
                        and not _source_appeared_before(
                            source_counts,
                            source_leaf_ids,
                            row,
                            direction,
                            source,
                            value,
                        )
                        and not _is_selected(
                            first_primary_id,
                            primary_count,
                            selected_leaf_ids,
                            selected_count,
                            value,
                        )
                    ):
                        selected_leaf_ids[selected_count] = value
                        selected_count += 1

    return primary_count, selected_count


cpdef int64_t maximum_selected_refined_support_slots_unchecked(
    const int64_t[::1] candidate_leaf_ids,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
):
    cdef int64_t row, direction, source, value, candidate, count, maximum = 0
    cdef int64_t unique_count
    cdef int64_t unique_sources[56]
    cdef bint use_fixed
    for row in range(source_counts.shape[0]):
        candidate = candidate_leaf_ids[row]
        unique_count = 0
        use_fixed = True
        for direction in range(source_counts.shape[1]):
            for source in range(source_counts[row, direction]):
                value = source_leaf_ids[row, direction, source]
                if (
                    value != candidate
                    and not _fixed_list_contains(
                        &unique_sources[0], unique_count, value
                    )
                ):
                    if unique_count == 56:
                        use_fixed = False
                    else:
                        unique_sources[unique_count] = value
                        unique_count += 1
        if use_fixed:
            count = 1 + unique_count
        else:
            count = 1
            for direction in range(source_counts.shape[1]):
                for source in range(source_counts[row, direction]):
                    value = source_leaf_ids[row, direction, source]
                    if value != candidate and not _source_appeared_before(
                        source_counts,
                        source_leaf_ids,
                        row,
                        direction,
                        source,
                        value,
                    ):
                        count += 1
        if count > maximum:
            maximum = count
    return maximum


cpdef tuple plan_selected_refined_support_prefix_unchecked(
    const int64_t[::1] candidate_leaf_ids,
    const uint8_t[:, ::1] source_counts,
    const int64_t[:, :, ::1] source_leaf_ids,
    int64_t[::1] selected_leaf_ids,
):
    cdef int64_t capacity = selected_leaf_ids.shape[0]
    cdef int64_t primary_count = 0
    cdef int64_t selected_count = 0
    cdef int64_t row, candidate, position, missing, direction, source, value
    cdef int64_t index, new_count
    cdef int64_t new_support[56]
    cdef bint use_fixed

    while primary_count < source_counts.shape[0]:
        row = primary_count
        candidate = candidate_leaf_ids[row]
        position = _find_support_id(
            selected_leaf_ids,
            primary_count,
            selected_count,
            candidate,
        )
        missing = 1 if position < 0 else 0
        new_count = 0
        use_fixed = True
        for direction in range(source_counts.shape[1]):
            for source in range(source_counts[row, direction]):
                value = source_leaf_ids[row, direction, source]
                if (
                    value != candidate
                    and not _is_explicitly_selected(
                        selected_leaf_ids,
                        selected_count,
                        value,
                    )
                    and not _fixed_list_contains(
                        &new_support[0], new_count, value
                    )
                ):
                    if new_count == 56:
                        use_fixed = False
                    else:
                        new_support[new_count] = value
                        new_count += 1

        if use_fixed:
            missing += new_count
        else:
            missing = 1 if position < 0 else 0
            for direction in range(source_counts.shape[1]):
                for source in range(source_counts[row, direction]):
                    value = source_leaf_ids[row, direction, source]
                    if (
                        value != candidate
                        and not _source_appeared_before(
                            source_counts,
                            source_leaf_ids,
                            row,
                            direction,
                            source,
                            value,
                        )
                        and not _is_explicitly_selected(
                            selected_leaf_ids,
                            selected_count,
                            value,
                        )
                    ):
                        missing += 1

        if selected_count + missing > capacity:
            if primary_count == 0:
                raise ValueError(
                    "selected capacity cannot fit the first refined support closure"
                )
            break

        if position >= primary_count:
            for index in range(position, selected_count - 1):
                selected_leaf_ids[index] = selected_leaf_ids[index + 1]
            selected_count -= 1

        for index in range(selected_count, primary_count, -1):
            selected_leaf_ids[index] = selected_leaf_ids[index - 1]
        selected_leaf_ids[primary_count] = candidate
        primary_count += 1
        selected_count += 1

        if use_fixed:
            for index in range(new_count):
                selected_leaf_ids[selected_count] = new_support[index]
                selected_count += 1
        else:
            for direction in range(source_counts.shape[1]):
                for source in range(source_counts[row, direction]):
                    value = source_leaf_ids[row, direction, source]
                    if (
                        value != candidate
                        and not _source_appeared_before(
                            source_counts,
                            source_leaf_ids,
                            row,
                            direction,
                            source,
                            value,
                        )
                        and not _is_explicitly_selected(
                            selected_leaf_ids,
                            selected_count,
                            value,
                        )
                    ):
                        selected_leaf_ids[selected_count] = value
                        selected_count += 1

    return primary_count, selected_count
