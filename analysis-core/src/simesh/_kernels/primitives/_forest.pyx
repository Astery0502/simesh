"""Typed Cartesian 3D refined-forest reconstruction for FST-001."""

import cython

from libc.stdint cimport int64_t, uint8_t


cdef int64_t _INDEX_MAX = 0x7fffffffffffffff


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _validate_node(
    const uint8_t* is_leaf,
    int64_t node_count,
    int64_t* next_node,
    int64_t level,
    int64_t extent_x,
    int64_t extent_y,
    int64_t extent_z,
    int64_t* max_level,
    int64_t* bad_node,
) noexcept nogil:
    cdef int64_t node_id, child
    cdef int status

    if next_node[0] >= node_count:
        bad_node[0] = next_node[0]
        return 1

    node_id = next_node[0]
    next_node[0] += 1
    if level > max_level[0]:
        max_level[0] = level
    if is_leaf[node_id] != 0:
        return 0

    # A child level doubles the complete logical grid on every axis.  This
    # bounds recursion by the signed-int64 index universe even for a deep
    # child-zero chain whose occupied coordinate itself remains zero.
    if (
        extent_x > _INDEX_MAX // 2
        or extent_y > _INDEX_MAX // 2
        or extent_z > _INDEX_MAX // 2
    ):
        bad_node[0] = node_id
        return 2

    for child in range(8):
        status = _validate_node(
            is_leaf,
            node_count,
            next_node,
            level + 1,
            extent_x * 2,
            extent_y * 2,
            extent_z * 2,
            max_level,
            bad_node,
        )
        if status != 0:
            return status
    return 0


cpdef tuple validate_refined_forest_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, ::1] rank_to_coord,
    const uint8_t[::1] is_leaf,
):
    cdef int64_t node_count = is_leaf.shape[0]
    cdef int64_t root_count = rank_to_coord.shape[0]
    cdef int64_t next_node = 0
    cdef int64_t max_level = 0
    cdef int64_t bad_node = 0
    cdef int64_t root
    cdef int status
    cdef const uint8_t* leaf_ptr

    if node_count == 0:
        return 1, 0, 0
    leaf_ptr = &is_leaf[0]
    with nogil:
        for root in range(root_count):
            status = _validate_node(
                leaf_ptr,
                node_count,
                &next_node,
                1,
                root_shape[0],
                root_shape[1],
                root_shape[2],
                &max_level,
                &bad_node,
            )
            if status != 0:
                break
    if status != 0:
        return status, bad_node, max_level
    if next_node != node_count:
        return 3, next_node, max_level
    return 0, -1, max_level


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_node(
    const uint8_t* is_leaf,
    int64_t* next_node,
    int64_t* next_leaf,
    int64_t level,
    int64_t x,
    int64_t y,
    int64_t z,
    int64_t parent,
    int64_t* node_levels,
    int64_t* node_coords,
    int64_t* parent_node_ids,
    int64_t* child_node_ids,
    int64_t* node_leaf_ids,
    int64_t* leaf_node_ids,
) noexcept nogil:
    cdef int64_t node_id = next_node[0]
    cdef int64_t child, child_node, leaf
    cdef int64_t xbit, ybit, zbit

    next_node[0] += 1
    node_levels[node_id] = level
    node_coords[node_id * 3] = x
    node_coords[node_id * 3 + 1] = y
    node_coords[node_id * 3 + 2] = z
    parent_node_ids[node_id] = parent
    node_leaf_ids[node_id] = -1
    for child in range(8):
        child_node_ids[node_id * 8 + child] = -1

    if is_leaf[node_id] != 0:
        leaf = next_leaf[0]
        next_leaf[0] += 1
        node_leaf_ids[node_id] = leaf
        leaf_node_ids[leaf] = node_id
        return

    for child in range(8):
        child_node = next_node[0]
        child_node_ids[node_id * 8 + child] = child_node
        xbit = child & 1
        ybit = (child >> 1) & 1
        zbit = (child >> 2) & 1
        _fill_node(
            is_leaf,
            next_node,
            next_leaf,
            level + 1,
            x * 2 + xbit,
            y * 2 + ybit,
            z * 2 + zbit,
            node_id,
            node_levels,
            node_coords,
            parent_node_ids,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
        )


cpdef void fill_refined_forest_unchecked(
    const int64_t[:, ::1] rank_to_coord,
    const uint8_t[::1] is_leaf,
    int64_t[::1] node_levels,
    int64_t[:, ::1] node_coords,
    int64_t[::1] parent_node_ids,
    int64_t[:, ::1] child_node_ids,
    int64_t[::1] node_leaf_ids,
    int64_t[::1] leaf_node_ids,
    int64_t[::1] root_node_ids,
):
    cdef int64_t root_count = rank_to_coord.shape[0]
    cdef int64_t next_node = 0
    cdef int64_t next_leaf = 0
    cdef int64_t root
    cdef const uint8_t* leaf_ptr = &is_leaf[0]
    cdef const int64_t* root_coord_ptr = &rank_to_coord[0, 0]
    cdef int64_t* level_ptr = &node_levels[0]
    cdef int64_t* coord_ptr = &node_coords[0, 0]
    cdef int64_t* parent_ptr = &parent_node_ids[0]
    cdef int64_t* child_ptr = &child_node_ids[0, 0]
    cdef int64_t* node_leaf_ptr = &node_leaf_ids[0]
    cdef int64_t* leaf_node_ptr = &leaf_node_ids[0]
    cdef int64_t* root_node_ptr = &root_node_ids[0]

    with nogil:
        for root in range(root_count):
            root_node_ptr[root] = next_node
            _fill_node(
                leaf_ptr,
                &next_node,
                &next_leaf,
                1,
                root_coord_ptr[root * 3],
                root_coord_ptr[root * 3 + 1],
                root_coord_ptr[root * 3 + 2],
                -1,
                level_ptr,
                coord_ptr,
                parent_ptr,
                child_ptr,
                node_leaf_ptr,
                leaf_node_ptr,
            )
