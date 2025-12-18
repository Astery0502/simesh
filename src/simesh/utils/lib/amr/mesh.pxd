# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport uint32_t

from .forest cimport AMRForest

cdef class AMRMesh:

    cdef:

        uint32_t ndim

        # physical domain
        double xmin[3]
        double xmax[3]

        # attached forest object, add a count
        AMRForest forest

        # number of leaf blocks
        uint32_t nleafs

        # block size
        uint32_t bsize[3]

        # domain size
        uint32_t dsize[3]

        # number of lev1 blocks
        uint32_t nb[3]

        # number of ghost cells
        uint32_t ng

        # number of fields
        uint32_t nfields

        # coordinates of the block
        # Exposed as a public attribute so Python tests can access mesh.rnode
        public double[:,:] rnode

        # data array (very big)
        double[:,:,:,:,:] data

        # coarse data array (very big, too)
        double[:,:,:,:,:] datac

        # physical boundary indices to help identify physical boundaries
        int[:,:] idphyb

        # Store original pointers for deallocation
        double* _data_ptr
        double* _datac_ptr
        double* _rnode_ptr
        int* _idphyb_ptr

        # grid indices
        int ixGmin[3]
        int ixGmax[3]
        int ixMmin[3]
        int ixMmax[3]

        # coarse grid indices
        int ixCoGmin[3]
        int ixCoGmax[3]
        int ixCoMmin[3]
        int ixCoMmax[3]

        # block size for the coarse grid
        uint32_t bCosize[3]

        # number of ghost cells for the coarse grid
        uint32_t ngCo

        # interpolation order: default to 2
        uint32_t interpolation_order

        # send and receive min and max indices for the sibling neighbor blocks
        # first dim is 3 directions
        # second dim is 4 types of boundary neighbors:0 for near the lower, 2 for near the upper, and 1 for awary form, 3 for both boundary
        # third dim is three sides of neighbor condition: 0 is min, 2 is max, and 1 stays the same
        int ixS_srl_min[3][4][3]
        int ixS_srl_max[3][4][3]
        int ixR_srl_min[3][4][3]
        int ixR_srl_max[3][4][3]
        
        # send restricted (r) from finer (already coarsened)
        # first dim is 3 directions
        # second dim is 3 types of neighbor blocks
        # third dim is 3 sides of neighbor condition: 0 is min, 2 is max, and 1 stays the same
        # for the receive, additional 1 for finer block range (contains 2 coarser one)
        int ixS_r_min[3][3][3]
        int ixS_r_max[3][3][3]
        int ixR_r_min[3][3][4]
        int ixR_r_max[3][3][4]

        # send prolonged (p) to finer blocks
        # first dim is 3 directions
        # second dim is similar
        # third dim is 3 sides of neighbor condition: 0 is min, 2 is max, and 1 stays the same
        int ixS_p_min[3][3][4]
        int ixS_p_max[3][3][4]
        int ixR_p_min[3][3][4]
        int ixR_p_max[3][3][4]

    cdef void _init_block_gridindex(self)

    cdef void _init_block_coordinates(self)

    cdef inline uint32_t nindex(self, uint32_t n1, uint32_t n2, uint32_t n3) noexcept nogil

    cdef inline uint32_t ncindex(self, uint32_t nc1, uint32_t nc2, uint32_t nc3) noexcept nogil

    cpdef void uniform_grid_zero_order(self, double[:,:,:,:,:] data, double[:,:,:,:] uniform_grid, 
        uint32_t[:] nx, double[:] xmin_new, double[:] xmax_new)

    cpdef void uniform_to_sfc(self, double[:,:,:,:] uniform_data, double[:,:,:,:,:] sfc_data)

    cdef bint is_boundary(self, uint32_t ileaf, uint32_t[:,:] neighbor_type)# noexcept nogil

    cdef void getbc(self)

    cdef void fill_boundary_before_gc(self, uint32_t ileaf, uint32_t[:,:] neighbor_type)# noexcept nogil

    cdef void bc_phys(self, int iside, uint32_t idim, uint32_t ileaf, int ixBmin[3], int ixBmax[3], bint is_coarse)# noexcept nogil

    cdef void coarsen_grid(self, uint32_t ileaf)# noexcept nogil
    
    cdef void fill_coarse_boundary(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t[:,:] neighbor_type)# noexcept nogil

