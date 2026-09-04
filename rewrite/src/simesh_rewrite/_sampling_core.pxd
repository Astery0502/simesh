"""Shared exact binary64 primitives for Cartesian sampling kernels."""

from libc.math cimport floor
from libc.stdint cimport int64_t


cdef inline double canonical_face(
    double domain_lower,
    double domain_upper,
    double spacing,
    int64_t face_index,
    int64_t domain_cells,
) noexcept nogil:
    cdef volatile double offset
    if face_index == 0:
        return domain_lower
    if face_index == domain_cells:
        return domain_upper
    offset = <double>face_index * spacing
    return domain_lower + offset


cdef inline int64_t source_cell_index_range_binary(
    double point,
    double domain_lower,
    double domain_upper,
    double spacing,
    int64_t first_cell,
    int64_t cell_count,
    int64_t domain_cells,
) noexcept nogil:
    cdef int64_t lower = first_cell
    cdef int64_t upper = first_cell + cell_count - 1
    cdef int64_t middle
    while lower < upper:
        middle = lower + (upper - lower + 1) // 2
        if canonical_face(
            domain_lower,
            domain_upper,
            spacing,
            middle,
            domain_cells,
        ) <= point:
            lower = middle
        else:
            upper = middle - 1
    return lower


cdef inline int64_t source_cell_index_range(
    double point,
    double domain_lower,
    double domain_upper,
    double spacing,
    int64_t first_cell,
    int64_t cell_count,
    int64_t domain_cells,
) noexcept nogil:
    cdef volatile double delta = point - domain_lower
    cdef volatile double ratio = delta / spacing
    cdef int64_t last_cell = first_cell + cell_count - 1
    cdef int64_t candidate
    if ratio <= <double>first_cell:
        candidate = first_cell
    elif ratio >= <double>last_cell:
        candidate = last_cell
    else:
        candidate = <int64_t>ratio
    if canonical_face(
        domain_lower,
        domain_upper,
        spacing,
        candidate,
        domain_cells,
    ) > point or (
        candidate < last_cell
        and canonical_face(
            domain_lower,
            domain_upper,
            spacing,
            candidate + 1,
            domain_cells,
        ) <= point
    ):
        return source_cell_index_range_binary(
            point,
            domain_lower,
            domain_upper,
            spacing,
            first_cell,
            cell_count,
            domain_cells,
        )
    return candidate


cdef inline int64_t source_cell_index(
    double point,
    double domain_lower,
    double domain_upper,
    double spacing,
    int64_t domain_cells,
) noexcept nogil:
    return source_cell_index_range(
        point,
        domain_lower,
        domain_upper,
        spacing,
        0,
        domain_cells,
        domain_cells,
    )


cdef inline void trilinear_axis_stencil_from_point(
    double point,
    double block_lower,
    double spacing,
    int64_t* left_index,
    double* weight,
) noexcept nogil:
    cdef volatile double delta = point - block_lower
    cdef volatile double ratio = delta / spacing
    cdef volatile double normalized = ratio - 0.5
    left_index[0] = <int64_t>floor(normalized)
    weight[0] = normalized - <double>left_index[0]


cdef inline double fixed_lerp(
    double left,
    double right,
    double weight,
) noexcept nogil:
    cdef volatile double one_minus = 1.0 - weight
    cdef volatile double left_term = left * one_minus
    cdef volatile double right_term = right * weight
    cdef volatile double result = left_term + right_term
    return result
