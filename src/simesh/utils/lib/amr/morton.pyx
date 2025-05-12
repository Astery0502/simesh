# Functions for morton encoding and order calculation

# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport uint32_t
from libc.stdlib cimport malloc, free
from libc.math cimport ceil, log2

from ..math cimport max3

cdef uint32_t extract_10bits(uint32_t n):
    """
    Extract the first 10 bits from a 32-bit integer (1/3)

    Input:
    n: uint32_t, the integer to be extracted
    Output:
    n: uint32_t, the extracted integer of the first 10 bits
    """
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n

cdef uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z):
    """
    Encode 3D coordinates into a morton code
    the input is 32-bit integers but the value is limited to 10 bits (at most 16 bits)

    Input:
    x, y, z: uint32_t, the index coordinates to be encoded
    Output:
    morton: uint32_t, the morton code of the input coordinates
    """

    x, y, z = extract_10bits(x), extract_10bits(y), extract_10bits(z)
    return x | (y << 1) | (z << 2)

cpdef void fill_morton_mapping3D(
    uint32_t[:,:,:] ig2morton,
    uint32_t[:,:] morton2ig,
    uint32_t n1, n2, n3
):
    """
    Fill the mapping from the index to the morton code and vice versa

    Input:
    ig2morton: uint32_t[:,:,:], the mapping from the index to the morton code
    morton2ig: uint32_t[:,:], the mapping from the morton code to the index
    n1, n2, n3: uint32_t, the number of points in each dimension
    """
    cdef:
        uint32_t* idx2morton
        uint32_t** morton2ipow2

        uint32_t i, j, k
        uint32_t i1, i2, i3
        uint32_t idx, ipow2
        uint32_t npow2, npows
    
    # the smallest power of 2 that fits the largest dimension
    npow2 = 1 << (<uint32_t>ceil(log2(max3(n1, n2, n3)))) 
    npows = npow2 * npow2 * npow2

    # create a mapping from the index (1D) in the power of 2 grid to the morton code
    idx2morton = <uint32_t*>malloc(npows*sizeof(uint32_t))
    # create a mapping from morton code to the index in the power of 2 grid
    morton2ipow2 = <uint32_t**>malloc(npows * sizeof(uint32_t*))
    for i in range(npows):
        morton2ipow2[i] = <uint32_t*>malloc(3 * sizeof(uint32_t))

    # fill the full mapping 
    for i in range(npow2):
        for j in range(npow2):
            for k in range(npow2):
                idx = morton3D(i, j, k)
                idx2morton[idx] = idx
                morton2ipow2[idx][0] = i
                morton2ipow2[idx][1] = j
                morton2ipow2[idx][2] = k

    # fill the mapping from the morton code to the index
    for i in range(npows):
        i1 = morton2ipow2[i][0]
        i2 = morton2ipow2[i][1]
        i3 = morton2ipow2[i][2]
        if ((i1 < n1) and (i2 < n2) and (i3 < n3)):
            idx = idx2morton[i]
            morton2ig[idx][0] = i1
            morton2ig[idx][1] = i2
            morton2ig[idx][2] = i3
            ig2morton[i1, i2, i3] = idx
        else:
            for j in range(i, npows):
                idx2morton[j] -= 1
        
    # free the memory
    free(idx2morton)
    for i in range(npows):
        free(morton2ipow2[i])
    free(morton2ipow2)

def pmorton3D(uint32_t x, uint32_t y, uint32_t z):
    """
    Python wrapper for morton3D function - interleaves bits from three coordinates
    to create a Morton code. Useful for testing.
    
    Parameters:
    -----------
    x, y, z : int
        The 3D coordinates to encode (values should be within 10-bit range)
        
    Returns:
    --------
    int
        The Morton code created by interleaving bits of x, y, and z
    """
    return morton3D(x, y, z)
