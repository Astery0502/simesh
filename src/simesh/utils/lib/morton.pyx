# Functions for morton encoding and order calculation

import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t

cdef uint32_t extract_10bits(uint32_t n) nogil:
    """
    Extract the first 10 bits from a 32-bit integer (1/3)
    """
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n

cdef uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) nogil:
    """
    Encode 3D coordinates into a morton code
    the input is 32-bit integers but the value is limited to 10 bits (at most 16 bits)
    """

    x, y, z = extract_10bits(x), extract_10bits(y), extract_10bits(z)
    return x | (y << 1) | (z << 2)

def level1_Morton_order(uint32_t n1, uint32_t n2, uint32_t n3):
