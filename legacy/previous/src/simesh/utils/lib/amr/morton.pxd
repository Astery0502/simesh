# Declaration file for morton encoding functions

from libc.stdint cimport uint32_t

# Private function (not exposed)
cdef uint32_t extract_10bits(uint32_t n)

# Public functions
cdef uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) 

cpdef void fill_morton_mapping3D(
    uint32_t[:,:,:] ig2morton,
    uint32_t[:,:] morton2ig,
    uint32_t n1, n2, n3
) 
