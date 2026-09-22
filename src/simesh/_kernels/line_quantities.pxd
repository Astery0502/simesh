"""Compiled magnetic quantities shared by field-line consumers."""

cdef inline double twist_density(const double* b, const double* curl,
                                  double norm) noexcept nogil:
    # Avoid squaring a potentially large norm; retain the established ordering.
    cdef double value = (curl[0]/norm)*(b[0]/norm)
    value = value+(curl[1]/norm)*(b[1]/norm)
    value = value+(curl[2]/norm)*(b[2]/norm)
    return value/(4.*3.141592653589793)
