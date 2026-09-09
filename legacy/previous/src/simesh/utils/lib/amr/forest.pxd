# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

# class for forest

from libc.stdint cimport uint32_t

from ..tree cimport treeptr, TreeNode

cdef class AMRForest:
    cdef:
        uint32_t ng[3]
        bint periodB[3]
        bint poleB[3]
        public uint32_t ndim
        public uint32_t nleafs
        public uint32_t nparents
        public uint32_t max_level
    
        # [ig1, ig2, ig3] -> morton number (level 1)
        public uint32_t[:,:,:] ig2morton

        # morton number (level 1) -> [ig1, ig2, ig3]
        public uint32_t[:,:] morton2ig

        # of nleafs length
        public bint[:] is_leaf

        # [nleaf, 3^ndim], [:,:] is the neighbor type, [:,:] is the neighbor index
        public uint32_t[:,:] neighbor_index
        public uint32_t[:,:] neighbor_type

        # [nleaf, 4^ndim], containing the corner and edge neighbors of the next level
        public uint32_t[:,:] neighbor_children

        # index of the level 1 tree starts in the forest leaf list
        public uint32_t[:] idx1 

        treeptr* forest
        treeptr* sfc2node
        
        # Store original pointers for deallocation
        uint32_t* _idx1_ptr
        uint32_t* _ig2morton_ptr
        uint32_t* _morton2ig_ptr
        bint* _is_leaf_ptr
        uint32_t* _neighbor_children_ptr
        uint32_t* _neighbor_index_ptr
        uint32_t* _neighbor_type_ptr

    # [child_dim3, child_dim2, child_dim1] (reversed indices in python to be compatiable with fortran, 
    # to reserve the morton order) -> child_dim1*2**(ndim-1)+child_dim2*2**(ndim-2)+child_dim3*2**(ndim-3)
    # for 2d, child_dim3=0
    cdef inline uint32_t cindex(self, uint32_t* ic)

    # morton index for self.forest, [ng1, ng2, ng3] format 1d array, column major here since i is lowest in mortonEncode
    cdef inline uint32_t mindex(self, uint32_t* ig)

    # neighbor index for self.neighbor_index/type, 
    cdef inline uint32_t nindex(self, uint32_t* n)

    # neighbor children index for self.neighbor_children, [4, 4(, 4)] format 1d array, column major
    cdef inline uint32_t ncindex(self, uint32_t* nc)

    cdef void read_forest(self, bint[:] is_leaf)

    cdef void read_node(self, treeptr tree, uint32_t* ig, uint32_t level, 
                        uint32_t* inode_ptr, uint32_t* ileaf_ptr)

    cpdef bint[:] write_forest(self)

    cdef void write_node(self, treeptr tree, bint[:] forest, uint32_t* inode_ptr)

    cdef void asign_tree_neighbor(self, treeptr tree)

    cdef void find_root_neighbor(self, treeptr* neighbor, treeptr tree, int* ii)

    # find the neighbor type and index of tree at ii1, ii2[, ii3] direction
    cdef uint32_t find_neighbor(self, treeptr* neighbor, treeptr tree, int* ii, bint* pole)

    # after read_forest, the non-corner/edge neighbors are allocated to each node
    cdef void build_connectivity(self)

    # helper function for build_connectivity to avoid too much nested loops
    cdef void build_neighbor_children(self, uint32_t ileaf, uint32_t* ii)

    cpdef void test_neighbors(self)
