# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

# class for forest

from libc.stdint cimport uint32_t

from ..tree cimport octptr

cdef class OctreeForest:
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

        octptr* forest
        octptr* sfc2node

    # [ig1, ig2, ig3] -> ig1 * ng[0] + ig2 * ng[1] + ig3
    cdef inline uint32_t findex(self, uint32_t* ig)

    # [n1, n2, n3] -> n1 * 3 * 3 + n2 * 3 + n3
    cdef inline uint32_t nindex(self, uint32_t* n)

    # [nc1, nc2, nc3] -> nc1 * 4 * 4 + nc2 * 4 + nc3
    cdef inline uint32_t ncindex(self, uint32_t* nc)

    cdef void read_forest(self, bint[:] is_leaf)

    cdef void read_node(self, octptr tree, uint32_t* ig, uint32_t level, 
                        uint32_t* inode_ptr, uint32_t* ileaf_ptr)

    cpdef bint[:] write_forest(self)

    cdef void write_node(self, octptr tree, bint[:] forest, uint32_t* inode_ptr)

    cdef void asign_tree_neighbor(self, octptr tree)

    cdef void find_root_neighbor(self, octptr* neighbor, octptr tree, int* ii)

    # find the neighbor type and index of tree at ii1, ii2[, ii3] direction
    cdef uint32_t find_neighbor(self, octptr* neighbor, octptr tree, int* ii, bint* pole)

    # after read_forest, the non-corner/edge neighbors are allocated to each node
    cdef void build_connectivity(self)

    # helper function for build_connectivity to avoid too much nested loops
    cdef void build_neighbor_children(self, uint32_t ileaf, uint32_t* ii)

    cpdef void test_neighbors(self)
