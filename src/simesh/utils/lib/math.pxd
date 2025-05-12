# Functions for basic math operations

# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t

ctypedef fused int_t:
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t

cdef extern from * nogil:
    """
    // Helper macros for MAX and MIN
    #define MAX2(a, b) ((a) > (b) ? (a) : (b))
    #define MAX3(a, b, c) MAX2(MAX2(a, b), c)
    """

cdef inline uint32_t max2(uint32_t a, uint32_t b) nogil:
    return a if a > b else b

cdef inline uint32_t max3(uint32_t a, uint32_t b, uint32_t c) nogil:
    return max2(max2(a, b), c)
