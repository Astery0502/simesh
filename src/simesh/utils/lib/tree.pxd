# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True


from libc.stdint cimport uint32_t

# forward declarations
cdef struct OctreeNode
cdef struct QuadtreeNode

# wrapper of octreenode pointer acting as a 2nd level pointer but more memory managment friendly
cdef packed struct octptr:
    OctreeNode* node

# class for octree node
cdef struct OctreeNode:

    uint32_t ig[3]
    uint32_t level, ileaf # note that ileaf begins from 1, 0 means not a leaf
    bint isleaf
    octptr parent
    # note that children at 1st direction are stored at 3rd index to be compatiable with fortran
    octptr children[2][2][2] 
    octptr neighbors[3][2]


# use ig3 = 0 for 2D 
# # wrapper of quadtree node pointer acting as a 2nd level pointer but more memory managment friendly
# cdef packed struct quadptr:
#     QuadtreeNode* node

# # class for quadtree node
# cdef struct QuadtreeNode:

#     uint32_t ig1, ig2, level, ileaf
#     bint is_leaf
#     quadptr parent
#     quadptr children[2][2]       # 4 children in 2D (2×2)
#     quadptr neighbors[2][2]      # 4 edge neighbors (2 per dimension)

