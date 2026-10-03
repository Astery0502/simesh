"""Cartesian box and lattice geometry, independent of topology and field storage."""



cdef inline bint contains_point(
    const double* p, const double* lo, const double* hi,
) noexcept nogil:
    # Ordered comparisons reject NaN and keep the exact half-open ownership.
    return (lo[0] <= p[0] < hi[0] and lo[1] <= p[1] < hi[1] and
            lo[2] <= p[2] < hi[2])
