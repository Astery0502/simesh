# Functions for basic math operations

# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t

ctypedef fused int_t:
    int
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t

cdef extern from *:
    """
    #define GET_MACRO(_1, _2, _3, NAME, ...) NAME
    #define MAX(...) GET_MACRO(__VA_ARGS__, MAX3, MAX2)(__VA_ARGS__)
    #define MAX2(a, b) ((a) > (b) ? (a) : (b))
    #define MAX3(a, b, c) MAX2(a, MAX2(b, c))
    """
