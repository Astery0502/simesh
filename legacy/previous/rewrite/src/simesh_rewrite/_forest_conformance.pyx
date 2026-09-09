"""Allocation-free FST-002 flat-forest artifact conformance."""

import cython

from libc.stdint cimport int64_t


cdef int64_t _INDEX_MAX = 0x7fffffffffffffff


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _validate_node(
    int64_t node_count,
    int64_t leaf_count,
    int64_t* next_node,
    int64_t* next_leaf,
    int64_t expected_level,
    int64_t expected_x,
    int64_t expected_y,
    int64_t expected_z,
    int64_t expected_parent,
    int64_t extent_x,
    int64_t extent_y,
    int64_t extent_z,
    const int64_t* node_levels,
    const int64_t* node_coords,
    const int64_t* parent_node_ids,
    const int64_t* child_node_ids,
    const int64_t* node_leaf_ids,
    const int64_t* leaf_node_ids,
    int64_t* max_level,
    int64_t* bad_node,
    int64_t* bad_leaf,
) noexcept nogil:
    cdef int64_t node, leaf, child, xbit, ybit, zbit
    cdef int status
    if next_node[0] >= node_count:
        bad_node[0] = next_node[0]
        return 1
    node = next_node[0]
    next_node[0] += 1
    if (
        node_levels[node] != expected_level
        or node_coords[node * 3] != expected_x
        or node_coords[node * 3 + 1] != expected_y
        or node_coords[node * 3 + 2] != expected_z
        or parent_node_ids[node] != expected_parent
    ):
        bad_node[0] = node
        return 2
    if expected_level > max_level[0]:
        max_level[0] = expected_level

    leaf = node_leaf_ids[node]
    if leaf >= 0:
        if next_leaf[0] >= leaf_count:
            bad_node[0] = node
            bad_leaf[0] = next_leaf[0]
            return 3
        if leaf != next_leaf[0] or leaf_node_ids[next_leaf[0]] != node:
            bad_node[0] = node
            bad_leaf[0] = next_leaf[0]
            return 4
        for child in range(8):
            if child_node_ids[node * 8 + child] != -1:
                bad_node[0] = node
                return 5
        next_leaf[0] += 1
        return 0

    if leaf != -1:
        bad_node[0] = node
        return 6
    if (
        extent_x > _INDEX_MAX // 2
        or extent_y > _INDEX_MAX // 2
        or extent_z > _INDEX_MAX // 2
    ):
        bad_node[0] = node
        return 7

    for child in range(8):
        if child_node_ids[node * 8 + child] != next_node[0]:
            bad_node[0] = node
            return 8
        xbit = child & 1
        ybit = (child >> 1) & 1
        zbit = (child >> 2) & 1
        status = _validate_node(
            node_count,
            leaf_count,
            next_node,
            next_leaf,
            expected_level + 1,
            expected_x * 2 + xbit,
            expected_y * 2 + ybit,
            expected_z * 2 + zbit,
            node,
            extent_x * 2,
            extent_y * 2,
            extent_z * 2,
            node_levels,
            node_coords,
            parent_node_ids,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            max_level,
            bad_node,
            bad_leaf,
        )
        if status != 0:
            return status
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple validate_refined_forest_arrays_unchecked(
    const int64_t[::1] root_shape,
    const int64_t[:, ::1] rank_to_coord,
    const int64_t[::1] root_node_ids,
    const int64_t[::1] node_levels,
    const int64_t[:, ::1] node_coords,
    const int64_t[::1] parent_node_ids,
    const int64_t[:, ::1] child_node_ids,
    const int64_t[::1] node_leaf_ids,
    const int64_t[::1] leaf_node_ids,
):
    cdef int64_t root_count = root_node_ids.shape[0]
    cdef int64_t node_count = node_levels.shape[0]
    cdef int64_t leaf_count = leaf_node_ids.shape[0]
    cdef int64_t next_node = 0
    cdef int64_t next_leaf = 0
    cdef int64_t max_level = 0
    cdef int64_t bad_node = -1
    cdef int64_t bad_leaf = -1
    cdef int64_t root
    cdef int status = 0

    with nogil:
        for root in range(root_count):
            if next_node >= node_count or root_node_ids[root] != next_node:
                bad_node = next_node
                bad_leaf = root
                status = 9
                break
            status = _validate_node(
                node_count,
                leaf_count,
                &next_node,
                &next_leaf,
                1,
                rank_to_coord[root, 0],
                rank_to_coord[root, 1],
                rank_to_coord[root, 2],
                -1,
                root_shape[0],
                root_shape[1],
                root_shape[2],
                &node_levels[0],
                &node_coords[0, 0],
                &parent_node_ids[0],
                &child_node_ids[0, 0],
                &node_leaf_ids[0],
                &leaf_node_ids[0],
                &max_level,
                &bad_node,
                &bad_leaf,
            )
            if status != 0:
                break
    if status != 0:
        return status, bad_node, bad_leaf, max_level
    if next_node != node_count:
        return 10, next_node, next_leaf, max_level
    if next_leaf != leaf_count:
        return 11, next_node, next_leaf, max_level
    return 0, -1, -1, max_level
