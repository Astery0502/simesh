# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from libc.math cimport ceil, floor

from cython.parallel import prange

from .forest cimport AMRForest
from ..tree cimport treeptr
from ..math cimport max2, min2

cdef class AMRMesh:

    def __cinit__(self, uint32_t ndim, uint32_t[:] bsize, uint32_t[:] dsize, 
                  double[:] xmin, double[:] xmax, uint32_t nghostcells, uint32_t nfields, 
                  AMRForest forest):

        cdef uint32_t i

        assert bsize.shape[0] == ndim
        assert dsize.shape[0] == ndim
        # reserve the pointer to the forest and read the nleafs from the forest
        self.forest = forest
        self.nleafs = forest.nleafs

        # check the dimension consistency
        self.ndim = ndim
        assert self.ndim == forest.ndim

        # initialize the rnode array (coordinates of blocks)
        rnode_data = <double*>malloc(self.nleafs * 9 * sizeof(double))
        for i in range(self.nleafs * 9):
            rnode_data[i] = 0
        self.rnode = <double[:self.nleafs, :9]>rnode_data
        self._rnode_ptr = rnode_data  # Store the original pointer for deallocation

        # calculate the block size and the coarse block size
        cdef uint32_t bgsize[3]
        cdef uint32_t bCosize[3]

        for i in range(self.ndim):
            self.xmin[i] = xmin[i]
            self.xmax[i] = xmax[i]

            self.bsize[i] = bsize[i]
            self.dsize[i] = dsize[i]
            bgsize[i] = self.bsize[i] + 2*self.ng

            assert dsize[i] % bsize[i] == 0
            assert bsize[i] % 2 == 0

            bCosize[i] = self.bsize[i]//2 + 2*self.ng
            self.bCosize[i] = bCosize[i]
            self.nb[i] = dsize[i] // bsize[i]

        # initialize the number of ghost cells and the number of fields
        self.ng = nghostcells
        self.nfields = nfields

        # initialize the data array
        if self.ng > 0:
            data_data = <double*>malloc(self.nleafs*bgsize[0]*bgsize[1]*bgsize[2]*self.nfields*sizeof(double))
            for i in range(self.nleafs*bgsize[0]*bgsize[1]*bgsize[2]*self.nfields):
                data_data[i] = 0
            self.data = <double[:self.nleafs, :bgsize[0], :bgsize[1], :bgsize[2], :self.nfields]>data_data
            self._data_ptr = data_data  # Store the original pointer for deallocation

            datac_data = <double*>malloc(self.nleafs*bCosize[0]*bCosize[1]*bCosize[2]*self.nfields*sizeof(double))
            for i in range(self.nleafs*bCosize[0]*bCosize[1]*bCosize[2]*self.nfields):
                datac_data[i] = 0
            self.datac = <double[:self.nleafs, :bCosize[0], :bCosize[1], :bCosize[2], :self.nfields]>datac_data
            self._datac_ptr = datac_data  # Store the original pointer for deallocation

            idphyb_data = <int*>malloc(3*self.nleafs*sizeof(int))
            for i in range(3*self.nleafs):
                idphyb_data[i] = 0
            self.idphyb = <int[:self.nleafs, :3]>idphyb_data
            self._idphyb_ptr = idphyb_data  # Store the original pointer for deallocation

            # initialize the grid indices
            self._init_block_gridindex()
        
        self._init_block_coordinates()

    def __dealloc__(self):
        """Free all allocated memory when the object is destroyed"""
        # Free the data arrays using stored pointers
        if hasattr(self, '_data_ptr') and self._data_ptr is not NULL:
            free(self._data_ptr)
        
        if hasattr(self, '_datac_ptr') and self._datac_ptr is not NULL:
            free(self._datac_ptr)
        
        if hasattr(self, '_rnode_ptr') and self._rnode_ptr is not NULL:
            free(self._rnode_ptr)
        
        if hasattr(self, '_idphyb_ptr') and self._idphyb_ptr is not NULL:
            free(self._idphyb_ptr)

    cdef void _init_block_coordinates(self):
        """Initialize the coordinates for each block."""

        cdef int ileaf
        cdef treeptr leaf_node_ptr
        cdef int idim,
        cdef int ig[3]

        for ileaf in range(self.nleafs):
            leaf_node_ptr = self.forest.sfc2node[ileaf]
            level = leaf_node_ptr.node.level
            for idim in range(self.ndim):
                ig[idim] = leaf_node_ptr.node.ig[idim]

                # 0: xmin, 1: ymin, 2: zmin
                self.rnode[ileaf, idim] = ig[idim] * (self.xmax[idim]-self.xmin[idim])/2**(level-1)/self.nb[idim] + self.xmin[idim]
                
                # 3: xmax, 4: ymax, 5: zmax
                self.rnode[ileaf, self.ndim+idim] = self.rnode[ileaf, idim] + (self.xmax[idim]-self.xmin[idim])/2**(level-1)/self.nb[idim]

                # 6: dx, 7: dy, 8: dz
                self.rnode[ileaf, 2*self.ndim+idim] = (self.xmax[idim]-self.xmin[idim])/2**(level-1)/self.dsize[idim]


    cdef void _init_block_gridindex(self):
        """Initialize the grid indices for each block."""

        cdef uint32_t i, j, k

        self.ngCo = (self.ng+1) // 2
        self.interpolation_order = 2 # default here temporally

        for i in range(self.ndim):

            self.ixGmin[i] = 0
            self.ixGmax[i] = self.bsize[i] + 2*self.ng - 1
            self.ixMmin[i] = self.ixGmin[i] + self.ng
            self.ixMmax[i] = self.ixGmax[i] - self.ng

            self.ixCoGmin[i] = 0
            self.ixCoGmax[i] = self.bsize[i]//2+2*self.ng-1
            self.ixCoMmin[i] = self.ixCoGmin[i] + self.ng
            self.ixCoMmax[i] = self.ixCoGmax[i] - self.ng

        for i in range(self.ndim):
            for j in range(4):
                self.ixS_srl_min[i][j][0] = self.ixMmin[i]
                self.ixS_srl_min[i][j][1] = self.ixMmin[i]
                self.ixS_srl_min[i][j][2] = self.ixMmax[i]+1-self.ng
                self.ixS_srl_max[i][j][0] = self.ixMmin[i]-1+self.ng
                self.ixS_srl_max[i][j][1] = self.ixMmax[i]
                self.ixS_srl_max[i][j][2] = self.ixMmax[i]

                self.ixR_srl_min[i][j][0] = self.ixGmin[i]
                self.ixR_srl_min[i][j][1] = self.ixMmin[i]
                self.ixR_srl_min[i][j][2] = self.ixMmax[i]+1
                self.ixR_srl_max[i][j][0] = self.ng-1
                self.ixR_srl_max[i][j][1] = self.ixMmax[i]
                self.ixR_srl_max[i][j][2] = self.ixGmax[i]


        for i in range(self.ndim):
            for j in range(3):
                self.ixS_r_min[i][j][0] = self.ixCoMmin[i]
                self.ixS_r_min[i][j][1] = self.ixCoMmin[i]
                self.ixS_r_min[i][j][2] = self.ixCoMmax[i]+1-self.ng
                self.ixS_r_max[i][j][0] = self.ixCoMmin[i]-1+self.ng
                self.ixS_r_max[i][j][1] = self.ixCoMmax[i]
                self.ixS_r_max[i][j][2] = self.ixCoMmax[i]

                self.ixR_r_min[i][j][0] = self.ixGmin[i]
                self.ixR_r_min[i][j][1] = self.ixMmin[i]
                self.ixR_r_min[i][j][2] = self.ixMmin[i]+self.bCosize[i]
                self.ixR_r_min[i][j][3] = self.ixMmax[i]+1
                self.ixR_r_max[i][j][0] = self.ng-1
                self.ixR_r_max[i][j][1] = self.ixMmin[i]-1+self.bCosize[i]
                self.ixR_r_max[i][j][2] = self.ixMmax[i]
                self.ixR_r_max[i][j][3] = self.ixGmax[i]

        for i in range(self.ndim):
            for j in range(3):
                self.ixS_p_min[i][j][0] = self.ixMmin[i]-(self.interpolation_order-1)
                self.ixS_p_min[i][j][1] = self.ixMmin[i]-(self.interpolation_order-1)
                self.ixS_p_min[i][j][2] = self.ixMmin[i]+self.bCosize[i]-self.ngCo-(self.interpolation_order-1)
                self.ixS_p_min[i][j][3] = self.ixMmax[i]+1-self.ngCo-(self.interpolation_order-1)
                self.ixS_p_max[i][j][0] = self.ixMmin[i]-1+self.ngCo+(self.interpolation_order-1)
                self.ixS_p_max[i][j][1] = self.ixMmin[i]-1+self.bCosize[i]+self.ngCo+(self.interpolation_order-1)
                self.ixS_p_max[i][j][2] = self.ixMmax[i]+(self.interpolation_order-1)
                self.ixS_p_max[i][j][3] = self.ixMmax[i]+(self.interpolation_order-1)

                self.ixR_p_min[i][j][0] = self.ixCoMmin[i]-self.ngCo-(self.interpolation_order-1)
                self.ixR_p_min[i][j][1] = self.ixCoMmin[i]-(self.interpolation_order-1)
                self.ixR_p_min[i][j][2] = self.ixCoMmin[i]-self.ngCo-(self.interpolation_order-1)
                self.ixR_p_min[i][j][3] = self.ixCoMmax[i]+1-(self.interpolation_order-1)
                self.ixR_p_max[i][j][0] = self.ng-1+(self.interpolation_order-1)
                self.ixR_p_max[i][j][1] = self.ixCoMmax[i]+self.ngCo+(self.interpolation_order-1)
                self.ixR_p_max[i][j][2] = self.ixCoMmax[i]+(self.interpolation_order-1)
                self.ixR_p_max[i][j][3] = self.ixCoMmax[i]+self.ngCo+(self.interpolation_order-1)
                
        for i in range(self.ndim):
            self.ixS_srl_min[i][0][1] = self.ixGmin[i]
            self.ixS_srl_min[i][2][1] = self.ixMmin[i]
            self.ixS_srl_min[i][3][1] = self.ixGmin[i]
            self.ixS_srl_max[i][0][1] = self.ixMmax[i]
            self.ixS_srl_max[i][2][1] = self.ixGmax[i]
            self.ixS_srl_max[i][3][1] = self.ixGmax[i]

            self.ixR_srl_min[i][0][1] = self.ixGmin[i]
            self.ixR_srl_min[i][2][1] = self.ixMmin[i]
            self.ixR_srl_min[i][3][1] = self.ixGmin[i]
            self.ixR_srl_max[i][0][1] = self.ixMmax[i]
            self.ixR_srl_max[i][2][1] = self.ixGmax[i]
            self.ixR_srl_max[i][3][1] = self.ixGmax[i]

            self.ixS_r_min[i][0][1] = self.ixGmin[i]
            self.ixS_r_min[i][2][1] = self.ixCoMmin[i]
            self.ixS_r_max[i][0][1] = self.ixCoMmax[i]
            self.ixS_r_max[i][2][1] = self.ixCoGmax[i]

            self.ixR_r_min[i][0][1] = self.ixGmin[i]
            self.ixR_r_min[i][2][1] = self.ixMmin[i]+self.bCosize[i]
            self.ixR_r_max[i][0][1] = self.ixMmin[i]-1+self.bCosize[i]
            self.ixR_r_max[i][2][1] = self.ixGmax[i]

            self.ixS_p_min[i][0][1] = self.ixGmin[i]
            self.ixS_p_max[i][2][1] = self.ixGmax[i]
            self.ixR_p_min[i][0][1] = self.ixGmin[i]
            self.ixR_p_max[i][2][1] = self.ixCoGmax[i]

    cpdef void uniform_grid_zero_order(self, double[:,:,:,:,:] data, double[:,:,:,:] uniform_grid, uint32_t[:] nx,
                                      double[:] xmin_new, double[:] xmax_new):
        """
        Interpolate the data from amr grid to uniform grid, zero order interpolation (nearest neighbor)
        """
        cdef uint32_t ileaf, idim
        cdef double dx_uniform[3]
        cdef int igmin[3]
        cdef int igmax[3]
        cdef uint32_t ix_uniform_in_block[3]
        cdef int i, j, k
        cdef bint flag

        assert data.shape[0] == self.nleafs, \
            f"data must have the same number of leafs as the mesh, {data.shape[0]} != {self.nleafs}"
        assert data.shape[1] == uniform_grid.shape[0], \
            f"data must have the same number of fields as the uniform grid, {data.shape[1]} != {uniform_grid.shape[0]}"
        # Compare spatial shape of uniform_grid with nx component-wise
        for idim in range(self.ndim):
            assert uniform_grid.shape[idim + 1] == nx[idim], \
                f"uniform_grid shape mismatch on dim {idim}: " \
                f"{uniform_grid.shape[idim + 1]} != {nx[idim]}"

        for idim in range(self.ndim):
            dx_uniform[idim] = (xmax_new[idim]-xmin_new[idim])/nx[idim]

        for ileaf in range(self.nleafs):
            flag = False
            for idim in range(self.ndim):

                # calculate the cell center of uniform grid included in the block: 0->nx
                igmin[idim] = <int>ceil((self.rnode[ileaf, idim]-xmin_new[idim])/dx_uniform[idim]-0.5)
                igmax[idim] = <int>floor((self.rnode[ileaf, idim+self.ndim]-xmin_new[idim])/dx_uniform[idim]+0.5)
                # Clamp to valid range (don't use max2 - it takes uint32_t and breaks for negative ints)
                if igmin[idim] < 0:
                    igmin[idim] = 0
                if igmax[idim] > <int>nx[idim]:
                    igmax[idim] = <int>nx[idim]

                if igmin[idim] > igmax[idim]:
                    flag = True
                    break
            
            if flag:
                continue

            # calculate the index of block cell that includes the uniform grid cell
            for i in range(igmin[0], igmax[0]):
                for j in range(igmin[1], igmax[1]):
                    for k in range(igmin[2], igmax[2]):
                        ix_uniform_in_block[0] = <uint32_t>floor(((i+0.5)*dx_uniform[0]+xmin_new[0]-self.rnode[ileaf, 0])/self.rnode[ileaf, 6])
                        ix_uniform_in_block[1] = <uint32_t>floor(((j+0.5)*dx_uniform[1]+xmin_new[1]-self.rnode[ileaf, 1])/self.rnode[ileaf, 7])
                        ix_uniform_in_block[2] = <uint32_t>floor(((k+0.5)*dx_uniform[2]+xmin_new[2]-self.rnode[ileaf, 2])/self.rnode[ileaf, 8])

                        uniform_grid[:,i,j,k] = data[ileaf, :, ix_uniform_in_block[0], ix_uniform_in_block[1], ix_uniform_in_block[2]]

    cpdef void uniform_to_sfc(self, double[:,:,:,:] uniform_data, double[:,:,:,:,:] sfc_data):
        """
        reallocate uniform data to block based sfc sequence
        """
        cdef uint32_t ileaf, idim
        cdef uint32_t nxg1[3]
        cdef uint32_t nxg2[3]

        assert uniform_data.shape[0] == self.nfields, \
         f"uniform_data must have the same number of fields as the mesh, {uniform_data.shape[0]} != {self.nfields}"
        assert sfc_data.shape[0] == self.nleafs, \
         f"sfc_data must have the same number of leafs as the mesh, {sfc_data.shape[0]} != {self.nleafs}"

        for ileaf in range(self.nleafs):
            for idim in range(self.ndim):
                nxg1[idim] = self.forest.sfc2node[ileaf].node.ig[idim]*self.bsize[idim]
                nxg2[idim] = (self.forest.sfc2node[ileaf].node.ig[idim]+1)*self.bsize[idim]
            if self.ndim == 2:
                nxg1[2] = 0
                nxg2[2] = 1
            sfc_data[ileaf,0:self.nfields,:self.bsize[0],:self.bsize[1],:self.bsize[2]] = \
                uniform_data[0:self.nfields,nxg1[0]:nxg2[0],nxg1[1]:nxg2[1],nxg1[2]:nxg2[2]]


    # two helper funcnctions to get the index same as forest to avoid python object usage (forest)
    cdef inline uint32_t nindex(self, uint32_t n1, uint32_t n2, uint32_t n3) noexcept nogil:
        # 3D case only
        return n1*3*3 + n2*3 + n3

    cdef inline uint32_t ncindex(self, uint32_t nc1, uint32_t nc2, uint32_t nc3) noexcept nogil:
        # 3D case only
        return nc1*4*4 + nc2*4 + nc3

    cdef bint is_boundary(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):# noexcept nogil:

        cdef uint32_t i

        for i in range(3**self.ndim):
            if neighbor_type[ileaf, i] == 1:
                return True
        return False

    cdef void getbc(self):
        """Get the boundary cells for each block."""

        # get some memoryviews for the objects in the forest and the index functions
        cdef uint32_t ileaf, i,j,k
        cdef uint32_t[:,:] neighbor_type = self.forest.neighbor_type
        cdef bint isboundary

        # for ileaf in prange(self.nleafs, nogil=True, schedule='static'):
        for ileaf in range(self.nleafs):

            self.fill_boundary_before_gc(ileaf, neighbor_type)

        for ileaf in range(self.nleafs):
            for i in range(3**self.ndim):
                if neighbor_type[ileaf,i] == 2:
                    self.coarsen_grid(ileaf)
                    break
            isboundary = self.is_boundary(ileaf, neighbor_type)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if neighbor_type[ileaf, self.nindex(i,j,k)] == 2 and isboundary:
                            self.fill_coarse_boundary(ileaf, i,j,k, neighbor_type)

    cdef void fill_boundary_before_gc(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):# noexcept nogil:
        """Fill the boundary cells for each block."""

        cdef uint32_t idim, idir
        cdef int ixBmin[3]
        cdef int ixBmax[3]
        cdef uint32_t i
        cdef int i1, i2, i3, iside

        # iterate over 3 directions
        for idim in range(self.ndim):

            for i in range(self.ndim):
                ixBmin[i] = self.ixGmin[i] + (self.ng if i != idim else 0)
                ixBmax[i] = self.ixGmax[i] - (self.ng if i != idim else 0)

            # if left boundary
            if ((idim > 0) and neighbor_type[ileaf, self.nindex(0,1,1)]==1):
                ixBmin[0] = self.ixGmin[0]
            # if right boundary
            if ((idim > 0) and neighbor_type[ileaf, self.nindex(2,1,1)]==1):
                ixBmax[0] = self.ixGmax[0]
            # if back boundary
            if ((idim > 1) and neighbor_type[ileaf, self.nindex(1,0,1)]==1):
                ixBmin[1] = self.ixGmin[1]
            # if front boundary
            if ((idim > 1) and neighbor_type[ileaf, self.nindex(1,2,1)]==1):
                ixBmax[1] = self.ixGmax[1]
            
            for iside in range(2):
                i1 = 1 + (2*iside - 1) * (idim == 0)
                i2 = 1 + (2*iside - 1) * (idim == 1)
                i3 = 1 + (2*iside - 1) * (idim == 2)
                if (neighbor_type[ileaf, self.nindex(i1,i2,i3)] != 1):
                    continue
                self.bc_phys(iside,idim,ileaf,ixBmin,ixBmax,False)
        
    cdef void bc_phys(self, int iside, uint32_t idim, uint32_t ileaf, int ixBmin[3], int ixBmax[3], bint is_coarse):# noexcept nogil:

        cdef int ixOmin[3]
        cdef int ixOmax[3]
        cdef int ixImin[3]
        cdef int ixImax[3]
        cdef uint32_t idir
        cdef int i1, i2, i3, o1, o2, o3, ifield
        cdef double[:,:,:,:] data_array

        if is_coarse:
            data_array = self.datac[ileaf]
        else:
            data_array = self.data[ileaf]

        # right side boundary
        if iside == 1:
            for idir in range(self.ndim):
                ixOmin[idir] = ixBmin[idir] if idir != idim else ixBmax[idir]+1-self.ng
                ixOmax[idir] = ixBmax[idir] 

                ixImin[idir] = ixOmin[idir] if idir != idim else ixOmin[idir]-1
                ixImax[idir] = ixOmax[idir] if idir != idim else ixOmax[idir]

        elif iside == 0:
            for idir in range(self.ndim):
                ixOmin[idir] = ixBmin[idir]
                ixOmax[idir] = ixBmax[idir] if idir != idim else ixBmin[idir]-1+self.ng

                ixImin[idir] = ixOmin[idir] if idir != idim else ixOmax[idir]+1
                ixImax[idir] = ixOmax[idir] if idir != idim else ixOmax[idir]+2
        
        # constant physical boundary
        for o1 in range(ixOmin[0], ixOmax[0]+1):
            i1 = o1 if idim !=0 else ixImin[0]
            for o2 in range(ixOmin[1], ixOmax[1]+1):
                i2 = o2 if idim !=1 else ixImin[1]
                for o3 in range(ixOmin[2], ixOmax[2]+1):
                    i3 = o3 if idim !=2 else ixImin[2]
                    for ifield in range(self.nfields):
                        data_array[o1,o2,o3,ifield] = data_array[i1,i2,i3,ifield]

    cdef void coarsen_grid(self, uint32_t ileaf):# noexcept nogil:

        cdef uint32_t ixCo1, ixCo2, ixCo3, ixFi1, ixFi2, ixFi3
        cdef uint32_t i, j, k
        cdef double sum_value
        cdef double CoFiratio = 1/2**3

        for ixCo1 in range(self.ixCoMmin[0], self.ixCoMmax[0]+1):
            ixFi1 = (ixCo1-self.ixCoMin[0])*2+self.ixMmin[0]
            for ixCo2 in range(self.ixCoMmin[1], self.ixCoMmax[1]+1):
                ixFi2 = (ixCo2-self.ixCoMin[1])*2+self.ixMmin[1]
                for ixCo3 in range(self.ixCoMmin[2], self.ixCoMmax[2]+1):
                    ixFi3 = (ixCo3-self.ixCoMin[2])*2+self.ixMmin[2]
                    for ifield in range(self.nfields):
                        sum_value = (
                            self.data[ileaf, ixFi1,   ixFi2,   ixFi3,   ifield] +
                            self.data[ileaf, ixFi1+1, ixFi2,   ixFi3,   ifield] +
                            self.data[ileaf, ixFi1,   ixFi2+1, ixFi3,   ifield] +
                            self.data[ileaf, ixFi1+1, ixFi2+1, ixFi3,   ifield] +
                            self.data[ileaf, ixFi1,   ixFi2,   ixFi3+1, ifield] +
                            self.data[ileaf, ixFi1+1, ixFi2,   ixFi3+1, ifield] +
                            self.data[ileaf, ixFi1,   ixFi2+1, ixFi3+1, ifield] +
                            self.data[ileaf, ixFi1+1, ixFi2+1, ixFi3+1, ifield]
                        ) * CoFiratio
                        
                        self.datac[ileaf, ixCo1, ixCo2, ixCo3, ifield] = sum_value

    cdef void fill_coarse_boundary(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t[:,:] neighbor_type):# noexcept nogil:

        cdef uint32_t idim
        cdef uint32_t ins[3]
        cdef int ixBmin[3]
        cdef int ixBmax[3]

        cdef uint32_t i
        cdef int iside

        cdef int iis[3]
        cdef bint should_continue

        ins[0] = i1
        ins[1] = i2
        ins[2] = i3

        # iterate over 3 directions
        for idim in range(self.ndim):

            for i in range(self.ndim):
                ixBmin[i] = self.ixCoGmin[i] + (self.ng if i != idim else 0)
                ixBmax[i] = self.ixCoGmax[i] - (self.ng if i != idim else 0)

            # if left boundary
            if ((idim > 0) and neighbor_type[ileaf, self.nindex(0,1,1)]==1):
                ixBmin[0] = self.ixCoGmin[0]
            # if right boundary
            if ((idim > 0) and neighbor_type[ileaf, self.nindex(2,1,1)]==1):
                ixBmax[0] = self.ixCoGmax[0]
            # if back boundary
            if ((idim > 1) and neighbor_type[ileaf, self.nindex(1,0,1)]==1):
                ixBmin[1] = self.ixCoGmin[1]
            # if front boundary
            if ((idim > 1) and neighbor_type[ileaf, self.nindex(1,2,1)]==1):
                ixBmax[1] = self.ixCoGmax[1]

            for i in range(self.ndim):
                if ins[i] == 0:
                    ixBmin[i] = self.ixCoGmin[i] + self.ng
                    ixBmax[i] = self.ixCoGmin[i] + 2*self.ng-1
                elif ins[i] == 2:
                    ixBmin[i] = self.ixCoGmax[i] - 2*self.ng + 1
                    ixBmax[i] = self.ixCoGmax[i] - self.ng
                
            for iside in range(2):
                iis[0] = 1 + (2*iside - 1) * (idim == 0)
                iis[1] = 1 + (2*iside - 1) * (idim == 1)
                iis[2] = 1 + (2*iside - 1) * (idim == 2)

                should_continue = False
