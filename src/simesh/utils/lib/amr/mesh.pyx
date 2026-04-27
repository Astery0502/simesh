# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t
from libc.stddef cimport size_t
from libc.math cimport ceil, floor

import numpy as np

from .forest cimport AMRForest
from ..tree cimport treeptr


cdef inline double _abs_double(double value) noexcept nogil:
    if value < 0.0:
        return -value
    return value


cdef inline double _min_double(double a, double b) noexcept nogil:
    if a < b:
        return a
    return b


cdef inline double _limited_slope(double left_value, double center_value, double right_value) noexcept nogil:
    cdef double slope_l = center_value - left_value
    cdef double slope_r = right_value - center_value
    cdef double slope_c = 0.5 * (slope_l + slope_r)
    cdef double sign_c
    cdef double limited

    if slope_c > 0.0:
        sign_c = 1.0
        limited = _min_double(_abs_double(slope_c), _min_double(slope_l, slope_r))
    elif slope_c < 0.0:
        sign_c = -1.0
        limited = _min_double(_abs_double(slope_c), _min_double(-slope_l, -slope_r))
    else:
        return 0.0

    if limited <= 0.0:
        return 0.0
    return sign_c * limited


cdef class AMRMesh:

    def __cinit__(
        self,
        uint32_t ndim,
        uint32_t[:] bsize,
        uint32_t[:] dsize,
        double[:] xmin,
        double[:] xmax,
        uint32_t nghostcells,
        uint32_t nfields,
        AMRForest forest,
    ):

        cdef uint32_t i
        cdef uint32_t bgsize[3]
        cdef uint32_t bCosize[3]
        cdef double* rnode_data

        assert bsize.shape[0] == ndim
        assert dsize.shape[0] == ndim

        self.forest = forest
        self.nleafs = forest.nleafs
        self.ndim = ndim
        self.ng = nghostcells
        self.nfields = nfields

        assert self.ndim == forest.ndim

        self._data_ptr = NULL
        self._datac_ptr = NULL
        self._rnode_ptr = NULL
        self._idphyb_ptr = NULL

        rnode_data = <double*>malloc(self.nleafs * 9 * sizeof(double))
        for i in range(self.nleafs * 9):
            rnode_data[i] = 0.0
        self.rnode = <double[:self.nleafs, :9]>rnode_data
        self._rnode_ptr = rnode_data

        for i in range(self.ndim):
            self.xmin[i] = xmin[i]
            self.xmax[i] = xmax[i]
            self.bsize[i] = bsize[i]
            self.dsize[i] = dsize[i]

            assert dsize[i] % bsize[i] == 0
            assert bsize[i] % 2 == 0

            bgsize[i] = self.bsize[i] + 2 * self.ng
            bCosize[i] = self.bsize[i] // 2 + 2 * self.ng
            self.bCosize[i] = bCosize[i]
            self.nb[i] = dsize[i] // bsize[i]

        if self.ndim == 2:
            self.bsize[2] = 1
            self.dsize[2] = 1
            bgsize[2] = 1 + 2 * self.ng
            bCosize[2] = 1 + 2 * self.ng
            self.bCosize[2] = bCosize[2]
            self.nb[2] = 1

        self._init_block_gridindex()
        self._init_block_coordinates()

        if self.ng > 0:
            self._allocate_padded_storage()

    def __dealloc__(self):
        if self._data_ptr is not NULL:
            free(self._data_ptr)

        if self._datac_ptr is not NULL:
            free(self._datac_ptr)

        if self._rnode_ptr is not NULL:
            free(self._rnode_ptr)

        if self._idphyb_ptr is not NULL:
            free(self._idphyb_ptr)

    def __getattr__(self, name):
        if name == "data":
            if self._data_ptr is NULL:
                return None
            return np.asarray(self.data)
        if name == "datac":
            if self._datac_ptr is NULL:
                return None
            return np.asarray(self.datac)
        if name == "idphyb":
            if self._idphyb_ptr is NULL:
                return None
            return np.asarray(self.idphyb)
        raise AttributeError(name)

    cdef void _allocate_padded_storage(self):
        cdef uint32_t bgsize0 = self.bsize[0] + 2 * self.ng
        cdef uint32_t bgsize1 = self.bsize[1] + 2 * self.ng
        cdef uint32_t bgsize2 = self.bsize[2] + 2 * self.ng
        cdef size_t nvalues = <size_t>self.nleafs * bgsize0 * bgsize1 * bgsize2 * self.nfields
        cdef size_t i
        cdef double* data_data

        if self._data_ptr is not NULL:
            return

        data_data = <double*>malloc(nvalues * sizeof(double))
        if data_data is NULL:
            raise MemoryError()
        for i in range(nvalues):
            data_data[i] = 0.0

        self.data = <double[:self.nleafs, :bgsize0, :bgsize1, :bgsize2, :self.nfields]>data_data
        self._data_ptr = data_data

    cdef void _ensure_idphyb_storage(self):
        cdef size_t i
        cdef int* idphyb_data

        if self.ng == 0:
            return

        if self._idphyb_ptr is NULL:
            idphyb_data = <int*>malloc(3 * self.nleafs * sizeof(int))
            if idphyb_data is NULL:
                raise MemoryError()
            for i in range(3 * self.nleafs):
                idphyb_data[i] = 0
            self.idphyb = <int[:self.nleafs, :3]>idphyb_data
            self._idphyb_ptr = idphyb_data

    cdef void _ensure_coarse_storage(self):
        cdef size_t nvalues
        cdef size_t i
        cdef double* datac_data

        if self.ng == 0 or self._datac_ptr is not NULL:
            return

        nvalues = <size_t>self.nleafs * self.bCosize[0] * self.bCosize[1] * self.bCosize[2] * self.nfields
        datac_data = <double*>malloc(nvalues * sizeof(double))
        if datac_data is NULL:
            raise MemoryError()
        for i in range(nvalues):
            datac_data[i] = 0.0
        self.datac = <double[:self.nleafs, :self.bCosize[0], :self.bCosize[1], :self.bCosize[2], :self.nfields]>datac_data
        self._datac_ptr = datac_data

    cdef void _zero_idphyb_storage(self):
        cdef uint32_t ileaf, idim

        if self._idphyb_ptr is NULL:
            return

        for ileaf in range(self.nleafs):
            for idim in range(3):
                self.idphyb[ileaf, idim] = 0

    cdef void _zero_coarse_storage(self):
        cdef uint32_t ileaf, i, j, k, ifield

        if self._datac_ptr is NULL:
            return

        for ileaf in range(self.nleafs):
            for i in range(self.bCosize[0]):
                for j in range(self.bCosize[1]):
                    for k in range(self.bCosize[2]):
                        for ifield in range(self.nfields):
                            self.datac[ileaf, i, j, k, ifield] = 0.0

    cdef bint _has_coarse_or_fine_neighbors(self, uint32_t[:,:] neighbor_type):
        cdef uint32_t ileaf, i

        for ileaf in range(self.nleafs):
            for i in range(3 ** self.ndim):
                if neighbor_type[ileaf, i] == 2 or neighbor_type[ileaf, i] == 4:
                    return True
        return False

    cpdef bint has_padded_data(self):
        return self._data_ptr is not NULL

    cpdef object padded_view(self):
        if self._data_ptr is NULL:
            return None
        return np.asarray(self.data)

    cpdef object interior_view(self):
        cdef object padded

        if self._data_ptr is NULL:
            return None

        padded = np.asarray(self.data)
        return np.transpose(
            padded[
                :,
                self.ixMmin[0]:self.ixMmax[0] + 1,
                self.ixMmin[1]:self.ixMmax[1] + 1,
                self.ixMmin[2]:self.ixMmax[2] + 1,
                :,
            ],
            (0, 4, 1, 2, 3),
        )

    cpdef void load_interior_data(self, double[:,:,:,:,:] interior):
        cdef object padded
        cdef object interior_array

        if self._data_ptr is NULL:
            raise ValueError("Interior loading requires padded mesh storage; construct the mesh with nghostcells > 0.")

        assert interior.shape[0] == self.nleafs
        assert interior.shape[1] == self.nfields
        assert interior.shape[2] == self.bsize[0]
        assert interior.shape[3] == self.bsize[1]
        assert interior.shape[4] == self.bsize[2]

        padded = np.asarray(self.data)
        padded[...] = 0.0
        interior_array = np.transpose(np.asarray(interior), (0, 2, 3, 4, 1))
        padded[
            :,
            self.ixMmin[0]:self.ixMmax[0] + 1,
            self.ixMmin[1]:self.ixMmax[1] + 1,
            self.ixMmin[2]:self.ixMmax[2] + 1,
            :,
        ] = interior_array

    cpdef void apply_ghost_cells(self):
        if self.ng == 0:
            return
        if self.ndim != 3:
            raise NotImplementedError("Ghost-cell exchange is currently implemented only for 3D meshes.")
        self.getbc()

    cdef void _init_block_coordinates(self):
        cdef uint32_t ileaf, idim, level
        cdef treeptr leaf_node_ptr
        cdef int ig[3]

        for ileaf in range(self.nleafs):
            leaf_node_ptr = self.forest.sfc2node[ileaf]
            level = leaf_node_ptr.node.level
            for idim in range(self.ndim):
                ig[idim] = leaf_node_ptr.node.ig[idim]

                self.rnode[ileaf, idim] = (
                    ig[idim] * (self.xmax[idim] - self.xmin[idim]) / 2 ** (level - 1) / self.nb[idim]
                    + self.xmin[idim]
                )
                self.rnode[ileaf, self.ndim + idim] = (
                    self.rnode[ileaf, idim]
                    + (self.xmax[idim] - self.xmin[idim]) / 2 ** (level - 1) / self.nb[idim]
                )
                self.rnode[ileaf, 2 * self.ndim + idim] = (
                    (self.xmax[idim] - self.xmin[idim]) / 2 ** (level - 1) / self.dsize[idim]
                )

    cdef void _init_block_gridindex(self):
        cdef uint32_t i, j

        self.ngCo = (self.ng + 1) // 2
        self.interpolation_order = 2

        for i in range(self.ndim):
            self.ixGmin[i] = 0
            self.ixGmax[i] = self.bsize[i] + 2 * self.ng - 1
            self.ixMmin[i] = self.ixGmin[i] + self.ng
            self.ixMmax[i] = self.ixGmax[i] - self.ng

            self.ixCoGmin[i] = 0
            self.ixCoGmax[i] = self.bsize[i] // 2 + 2 * self.ng - 1
            self.ixCoMmin[i] = self.ixCoGmin[i] + self.ng
            self.ixCoMmax[i] = self.ixCoGmax[i] - self.ng

        for i in range(self.ndim):
            for j in range(4):
                self.ixS_srl_min[i][j][0] = self.ixMmin[i]
                self.ixS_srl_min[i][j][1] = self.ixMmin[i]
                self.ixS_srl_min[i][j][2] = self.ixMmax[i] + 1 - self.ng
                self.ixS_srl_max[i][j][0] = self.ixMmin[i] - 1 + self.ng
                self.ixS_srl_max[i][j][1] = self.ixMmax[i]
                self.ixS_srl_max[i][j][2] = self.ixMmax[i]

                self.ixR_srl_min[i][j][0] = self.ixGmin[i]
                self.ixR_srl_min[i][j][1] = self.ixMmin[i]
                self.ixR_srl_min[i][j][2] = self.ixMmax[i] + 1
                self.ixR_srl_max[i][j][0] = self.ng - 1
                self.ixR_srl_max[i][j][1] = self.ixMmax[i]
                self.ixR_srl_max[i][j][2] = self.ixGmax[i]

        for i in range(self.ndim):
            for j in range(3):
                self.ixS_r_min[i][j][0] = self.ixCoMmin[i]
                self.ixS_r_min[i][j][1] = self.ixCoMmin[i]
                self.ixS_r_min[i][j][2] = self.ixCoMmax[i] + 1 - self.ng
                self.ixS_r_max[i][j][0] = self.ixCoMmin[i] - 1 + self.ng
                self.ixS_r_max[i][j][1] = self.ixCoMmax[i]
                self.ixS_r_max[i][j][2] = self.ixCoMmax[i]

                self.ixR_r_min[i][j][0] = self.ixGmin[i]
                self.ixR_r_min[i][j][1] = self.ixMmin[i]
                self.ixR_r_min[i][j][2] = self.ixMmin[i] + self.bsize[i] // 2
                self.ixR_r_min[i][j][3] = self.ixMmax[i] + 1
                self.ixR_r_max[i][j][0] = self.ng - 1
                self.ixR_r_max[i][j][1] = self.ixMmin[i] - 1 + self.bsize[i] // 2
                self.ixR_r_max[i][j][2] = self.ixMmax[i]
                self.ixR_r_max[i][j][3] = self.ixGmax[i]

        for i in range(self.ndim):
            for j in range(3):
                self.ixS_p_min[i][j][0] = self.ixMmin[i] - (self.interpolation_order - 1)
                self.ixS_p_min[i][j][1] = self.ixMmin[i] - (self.interpolation_order - 1)
                self.ixS_p_min[i][j][2] = self.ixMmin[i] + self.bsize[i] // 2 - self.ngCo - (self.interpolation_order - 1)
                self.ixS_p_min[i][j][3] = self.ixMmax[i] + 1 - self.ngCo - (self.interpolation_order - 1)
                self.ixS_p_max[i][j][0] = self.ixMmin[i] - 1 + self.ngCo + (self.interpolation_order - 1)
                self.ixS_p_max[i][j][1] = self.ixMmin[i] - 1 + self.bsize[i] // 2 + self.ngCo + (self.interpolation_order - 1)
                self.ixS_p_max[i][j][2] = self.ixMmax[i] + (self.interpolation_order - 1)
                self.ixS_p_max[i][j][3] = self.ixMmax[i] + (self.interpolation_order - 1)

                self.ixR_p_min[i][j][0] = self.ixCoMmin[i] - self.ngCo - (self.interpolation_order - 1)
                self.ixR_p_min[i][j][1] = self.ixCoMmin[i] - (self.interpolation_order - 1)
                self.ixR_p_min[i][j][2] = self.ixCoMmin[i] - self.ngCo - (self.interpolation_order - 1)
                self.ixR_p_min[i][j][3] = self.ixCoMmax[i] + 1 - (self.interpolation_order - 1)
                self.ixR_p_max[i][j][0] = self.ng - 1 + (self.interpolation_order - 1)
                self.ixR_p_max[i][j][1] = self.ixCoMmax[i] + self.ngCo + (self.interpolation_order - 1)
                self.ixR_p_max[i][j][2] = self.ixCoMmax[i] + (self.interpolation_order - 1)
                self.ixR_p_max[i][j][3] = self.ixCoMmax[i] + self.ngCo + (self.interpolation_order - 1)

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
            self.ixR_r_min[i][2][2] = self.ixMmin[i] + self.bsize[i] // 2
            self.ixR_r_max[i][0][1] = self.ixMmin[i] - 1 + self.bsize[i] // 2
            self.ixR_r_max[i][2][2] = self.ixGmax[i]

            self.ixS_p_min[i][0][1] = self.ixGmin[i]
            self.ixS_p_max[i][2][2] = self.ixGmax[i]
            self.ixR_p_min[i][0][1] = self.ixGmin[i]
            self.ixR_p_max[i][2][2] = self.ixCoGmax[i]

    cpdef void uniform_grid_zero_order(
        self,
        double[:,:,:,:,:] data,
        double[:,:,:,:] uniform_grid,
        uint32_t[:] nx,
        double[:] xmin_new,
        double[:] xmax_new,
    ):
        cdef uint32_t ileaf, idim
        cdef double dx_uniform[3]
        cdef int igmin[3]
        cdef int igmax[3]
        cdef uint32_t ix_uniform_in_block[3]
        cdef int i, j, k
        cdef bint skip_leaf

        assert data.shape[0] == self.nleafs
        assert data.shape[1] == uniform_grid.shape[0]

        for idim in range(self.ndim):
            assert uniform_grid.shape[idim + 1] == nx[idim]
            dx_uniform[idim] = (xmax_new[idim] - xmin_new[idim]) / nx[idim]

        for ileaf in range(self.nleafs):
            skip_leaf = False
            for idim in range(self.ndim):
                igmin[idim] = <int>ceil((self.rnode[ileaf, idim] - xmin_new[idim]) / dx_uniform[idim] - 0.5)
                igmax[idim] = <int>floor((self.rnode[ileaf, idim + self.ndim] - xmin_new[idim]) / dx_uniform[idim] + 0.5)

                if igmin[idim] < 0:
                    igmin[idim] = 0
                if igmax[idim] > <int>nx[idim]:
                    igmax[idim] = <int>nx[idim]

                if igmin[idim] > igmax[idim]:
                    skip_leaf = True
                    break

            if skip_leaf:
                continue

            for i in range(igmin[0], igmax[0]):
                for j in range(igmin[1], igmax[1]):
                    for k in range(igmin[2], igmax[2]):
                        ix_uniform_in_block[0] = <uint32_t>floor(
                            ((i + 0.5) * dx_uniform[0] + xmin_new[0] - self.rnode[ileaf, 0]) / self.rnode[ileaf, 6]
                        )
                        ix_uniform_in_block[1] = <uint32_t>floor(
                            ((j + 0.5) * dx_uniform[1] + xmin_new[1] - self.rnode[ileaf, 1]) / self.rnode[ileaf, 7]
                        )
                        ix_uniform_in_block[2] = <uint32_t>floor(
                            ((k + 0.5) * dx_uniform[2] + xmin_new[2] - self.rnode[ileaf, 2]) / self.rnode[ileaf, 8]
                        )

                        uniform_grid[:, i, j, k] = data[
                            ileaf,
                            :,
                            ix_uniform_in_block[0],
                            ix_uniform_in_block[1],
                            ix_uniform_in_block[2],
                        ]

    cpdef void uniform_grid_linear(
        self,
        double[:,:,:,:] uniform_grid,
        uint32_t[:] nx,
        double[:] xmin_new,
        double[:] xmax_new,
        uint32_t[:] field_positions,
    ):
        cdef uint32_t ileaf, idim
        cdef uint32_t ifield_out, ifield_src
        cdef double dx_uniform[3]
        cdef int igmin[3]
        cdef int igmax[3]
        cdef int i, j, k
        cdef int i0, j0, k0
        cdef int i1, j1, k1
        cdef double x, y, z
        cdef double gx, gy, gz
        cdef double wx, wy, wz
        cdef double c00, c01, c10, c11
        cdef double c0, c1
        cdef bint skip_leaf

        if self._data_ptr is NULL or self.ng == 0:
            raise ValueError("Linear uniform interpolation requires padded ghost-cell storage.")
        if self.ndim != 3:
            raise NotImplementedError("Linear uniform interpolation is currently implemented only for 3D meshes.")

        assert uniform_grid.shape[0] == field_positions.shape[0]

        for ifield_out in range(<uint32_t>field_positions.shape[0]):
            assert field_positions[ifield_out] < self.nfields

        for idim in range(self.ndim):
            assert uniform_grid.shape[idim + 1] == nx[idim]
            dx_uniform[idim] = (xmax_new[idim] - xmin_new[idim]) / nx[idim]

        for ileaf in range(self.nleafs):
            skip_leaf = False
            for idim in range(self.ndim):
                igmin[idim] = <int>ceil((self.rnode[ileaf, idim] - xmin_new[idim]) / dx_uniform[idim] - 0.5)
                igmax[idim] = <int>floor((self.rnode[ileaf, idim + self.ndim] - xmin_new[idim]) / dx_uniform[idim] + 0.5)

                if igmin[idim] < 0:
                    igmin[idim] = 0
                if igmax[idim] > <int>nx[idim]:
                    igmax[idim] = <int>nx[idim]

                if igmin[idim] > igmax[idim]:
                    skip_leaf = True
                    break

            if skip_leaf:
                continue

            for i in range(igmin[0], igmax[0]):
                x = xmin_new[0] + (i + 0.5) * dx_uniform[0]
                gx = (x - self.rnode[ileaf, 0]) / self.rnode[ileaf, 6] - 0.5
                i0 = <int>floor(gx) + <int>self.ng
                i1 = i0 + 1
                wx = gx - (i0 - <int>self.ng)

                for j in range(igmin[1], igmax[1]):
                    y = xmin_new[1] + (j + 0.5) * dx_uniform[1]
                    gy = (y - self.rnode[ileaf, 1]) / self.rnode[ileaf, 7] - 0.5
                    j0 = <int>floor(gy) + <int>self.ng
                    j1 = j0 + 1
                    wy = gy - (j0 - <int>self.ng)

                    for k in range(igmin[2], igmax[2]):
                        z = xmin_new[2] + (k + 0.5) * dx_uniform[2]
                        gz = (z - self.rnode[ileaf, 2]) / self.rnode[ileaf, 8] - 0.5
                        k0 = <int>floor(gz) + <int>self.ng
                        k1 = k0 + 1
                        wz = gz - (k0 - <int>self.ng)

                        for ifield_out in range(<uint32_t>field_positions.shape[0]):
                            ifield_src = field_positions[ifield_out]
                            c00 = (
                                self.data[ileaf, i0, j0, k0, ifield_src] * (1.0 - wz)
                                + self.data[ileaf, i0, j0, k1, ifield_src] * wz
                            )
                            c01 = (
                                self.data[ileaf, i0, j1, k0, ifield_src] * (1.0 - wz)
                                + self.data[ileaf, i0, j1, k1, ifield_src] * wz
                            )
                            c10 = (
                                self.data[ileaf, i1, j0, k0, ifield_src] * (1.0 - wz)
                                + self.data[ileaf, i1, j0, k1, ifield_src] * wz
                            )
                            c11 = (
                                self.data[ileaf, i1, j1, k0, ifield_src] * (1.0 - wz)
                                + self.data[ileaf, i1, j1, k1, ifield_src] * wz
                            )

                            c0 = c00 * (1.0 - wy) + c01 * wy
                            c1 = c10 * (1.0 - wy) + c11 * wy
                            uniform_grid[ifield_out, i, j, k] = c0 * (1.0 - wx) + c1 * wx

    cpdef void uniform_full_level1(self, double[:,:,:,:,:] data, double[:,:,:,:] uniform_grid):
        cdef uint32_t ileaf, idim
        cdef uint32_t nxg1[3]
        cdef uint32_t nxg2[3]

        assert data.shape[0] == self.nleafs
        assert data.shape[1] == uniform_grid.shape[0]
        assert data.shape[2] == self.bsize[0]
        assert data.shape[3] == self.bsize[1]
        assert data.shape[4] == self.bsize[2]
        assert uniform_grid.shape[1] == self.dsize[0]
        assert uniform_grid.shape[2] == self.dsize[1]
        assert uniform_grid.shape[3] == self.dsize[2]

        for ileaf in range(self.nleafs):
            for idim in range(self.ndim):
                nxg1[idim] = self.forest.sfc2node[ileaf].node.ig[idim] * self.bsize[idim]
                nxg2[idim] = (self.forest.sfc2node[ileaf].node.ig[idim] + 1) * self.bsize[idim]

            if self.ndim == 2:
                nxg1[2] = 0
                nxg2[2] = 1

            uniform_grid[:, nxg1[0]:nxg2[0], nxg1[1]:nxg2[1], nxg1[2]:nxg2[2]] = (
                data[ileaf, :, :self.bsize[0], :self.bsize[1], :self.bsize[2]]
            )

    cpdef void uniform_to_sfc(self, double[:,:,:,:] uniform_data, double[:,:,:,:,:] sfc_data):
        cdef uint32_t ileaf, idim
        cdef uint32_t nxg1[3]
        cdef uint32_t nxg2[3]

        assert uniform_data.shape[0] == self.nfields
        assert sfc_data.shape[0] == self.nleafs

        for ileaf in range(self.nleafs):
            for idim in range(self.ndim):
                nxg1[idim] = self.forest.sfc2node[ileaf].node.ig[idim] * self.bsize[idim]
                nxg2[idim] = (self.forest.sfc2node[ileaf].node.ig[idim] + 1) * self.bsize[idim]

            if self.ndim == 2:
                nxg1[2] = 0
                nxg2[2] = 1

            sfc_data[ileaf, 0:self.nfields, :self.bsize[0], :self.bsize[1], :self.bsize[2]] = (
                uniform_data[0:self.nfields, nxg1[0]:nxg2[0], nxg1[1]:nxg2[1], nxg1[2]:nxg2[2]]
            )

    cdef inline uint32_t nindex(self, uint32_t n1, uint32_t n2, uint32_t n3) noexcept nogil:
        return n1 + n2 * 3 + n3 * 3 * 3

    cdef inline uint32_t ncindex(self, uint32_t nc1, uint32_t nc2, uint32_t nc3) noexcept nogil:
        return nc1 + nc2 * 4 + nc3 * 4 * 4

    cdef bint is_boundary(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):
        cdef uint32_t i

        for i in range(3 ** self.ndim):
            if neighbor_type[ileaf, i] == 1:
                return True
        return False

    cdef void getbc(self):
        cdef uint32_t ileaf, i, j, k
        cdef uint32_t[:,:] neighbor_type = self.forest.neighbor_type
        cdef bint isboundary
        cdef bint needs_coarse = self._has_coarse_or_fine_neighbors(neighbor_type)

        self._ensure_idphyb_storage()
        self._zero_idphyb_storage()
        if needs_coarse:
            self._ensure_coarse_storage()
            self._zero_coarse_storage()

        for ileaf in range(self.nleafs):
            self.fill_boundary_before_gc(ileaf, neighbor_type)

        if needs_coarse:
            for ileaf in range(self.nleafs):
                for i in range(3 ** self.ndim):
                    if neighbor_type[ileaf, i] == 2:
                        self.coarsen_grid(ileaf)
                        break

                isboundary = self.is_boundary(ileaf, neighbor_type)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if neighbor_type[ileaf, self.nindex(i, j, k)] == 2 and isboundary:
                                self.fill_coarse_boundary(ileaf, i, j, k, neighbor_type)

        for ileaf in range(self.nleafs):
            self.identifyphysbound(ileaf, neighbor_type)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if neighbor_type[ileaf, self.nindex(i, j, k)] == 2:
                            self.bc_fill_restrict(ileaf, i, j, k)
                        elif neighbor_type[ileaf, self.nindex(i, j, k)] == 3:
                            self.bc_fill_srl(ileaf, i, j, k)

        if needs_coarse:
            for ileaf in range(self.nleafs):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if neighbor_type[ileaf, self.nindex(i, j, k)] == 4:
                                self.bc_fill_prolong(ileaf, i, j, k)

            for ileaf in range(self.nleafs):
                self.gc_prolong(ileaf, neighbor_type)

        for ileaf in range(self.nleafs):
            self.fill_boundary_after_gc(ileaf, neighbor_type)

    cdef void fill_boundary_before_gc(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):
        cdef uint32_t idim, i
        cdef int ixBmin[3]
        cdef int ixBmax[3]
        cdef int i1, i2, i3, iside

        for idim in range(self.ndim):
            for i in range(self.ndim):
                ixBmin[i] = self.ixGmin[i] + (self.ng if i != idim else 0)
                ixBmax[i] = self.ixGmax[i] - (self.ng if i != idim else 0)

            if (idim > 0) and neighbor_type[ileaf, self.nindex(0, 1, 1)] == 1:
                ixBmin[0] = self.ixGmin[0]
            if (idim > 0) and neighbor_type[ileaf, self.nindex(2, 1, 1)] == 1:
                ixBmax[0] = self.ixGmax[0]
            if (idim > 1) and neighbor_type[ileaf, self.nindex(1, 0, 1)] == 1:
                ixBmin[1] = self.ixGmin[1]
            if (idim > 1) and neighbor_type[ileaf, self.nindex(1, 2, 1)] == 1:
                ixBmax[1] = self.ixGmax[1]

            for iside in range(2):
                i1 = 1 + (2 * iside - 1) * (idim == 0)
                i2 = 1 + (2 * iside - 1) * (idim == 1)
                i3 = 1 + (2 * iside - 1) * (idim == 2)
                if neighbor_type[ileaf, self.nindex(i1, i2, i3)] != 1:
                    continue
                self.bc_phys(iside, idim, ileaf, ixBmin, ixBmax, False)

    cdef void fill_boundary_after_gc(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):
        cdef uint32_t idim, i
        cdef int kmin[3]
        cdef int kmax[3]
        cdef int ixBmin[3]
        cdef int ixBmax[3]
        cdef int i1, i2, i3, iside

        for idim in range(self.ndim):
            for i in range(self.ndim):
                kmin[i] = 1 if (idim < i and neighbor_type[ileaf, self.nindex(1 - (idim == 0), 1 - (idim == 1), 1 - (idim == 2))] == 1) else 0
                kmax[i] = 1 if (idim < i and neighbor_type[ileaf, self.nindex(1 + (idim == 0), 1 + (idim == 1), 1 + (idim == 2))] == 1) else 0

            kmin[0] = 0
            kmax[0] = 0

            for i in range(self.ndim):
                ixBmin[i] = self.ixGmin[i] + kmin[i] * self.ng
                ixBmax[i] = self.ixGmax[i] - kmax[i] * self.ng

            for iside in range(2):
                i1 = 1 + (2 * iside - 1) * (idim == 0)
                i2 = 1 + (2 * iside - 1) * (idim == 1)
                i3 = 1 + (2 * iside - 1) * (idim == 2)
                if neighbor_type[ileaf, self.nindex(i1, i2, i3)] != 1:
                    continue
                self.bc_phys(iside, idim, ileaf, ixBmin, ixBmax, False)

    cdef void bc_phys(self, int iside, uint32_t idim, uint32_t ileaf, int ixBmin[3], int ixBmax[3], bint is_coarse):
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

        if iside == 1:
            for idir in range(self.ndim):
                ixOmin[idir] = ixBmin[idir] if idir != idim else ixBmax[idir] + 1 - self.ng
                ixOmax[idir] = ixBmax[idir]
                ixImin[idir] = ixOmin[idir] if idir != idim else ixOmin[idir] - 1
                ixImax[idir] = ixOmax[idir] if idir != idim else ixOmax[idir]
        else:
            for idir in range(self.ndim):
                ixOmin[idir] = ixBmin[idir]
                ixOmax[idir] = ixBmax[idir] if idir != idim else ixBmin[idir] - 1 + self.ng
                ixImin[idir] = ixOmin[idir] if idir != idim else ixOmax[idir] + 1
                ixImax[idir] = ixOmax[idir] if idir != idim else ixOmax[idir] + 2

        for o1 in range(ixOmin[0], ixOmax[0] + 1):
            i1 = o1 if idim != 0 else ixImin[0]
            for o2 in range(ixOmin[1], ixOmax[1] + 1):
                i2 = o2 if idim != 1 else ixImin[1]
                for o3 in range(ixOmin[2], ixOmax[2] + 1):
                    i3 = o3 if idim != 2 else ixImin[2]
                    for ifield in range(self.nfields):
                        data_array[o1, o2, o3, ifield] = data_array[i1, i2, i3, ifield]

    cdef void coarsen_grid(self, uint32_t ileaf):
        cdef uint32_t ixCo1, ixCo2, ixCo3
        cdef uint32_t ixFi1, ixFi2, ixFi3
        cdef uint32_t ifield
        cdef double sum_value
        cdef double CoFiratio = 0.125

        for ixCo1 in range(self.ixCoMmin[0], self.ixCoMmax[0] + 1):
            ixFi1 = (ixCo1 - self.ixCoMmin[0]) * 2 + self.ixMmin[0]
            for ixCo2 in range(self.ixCoMmin[1], self.ixCoMmax[1] + 1):
                ixFi2 = (ixCo2 - self.ixCoMmin[1]) * 2 + self.ixMmin[1]
                for ixCo3 in range(self.ixCoMmin[2], self.ixCoMmax[2] + 1):
                    ixFi3 = (ixCo3 - self.ixCoMmin[2]) * 2 + self.ixMmin[2]
                    for ifield in range(self.nfields):
                        sum_value = (
                            self.data[ileaf, ixFi1, ixFi2, ixFi3, ifield]
                            + self.data[ileaf, ixFi1 + 1, ixFi2, ixFi3, ifield]
                            + self.data[ileaf, ixFi1, ixFi2 + 1, ixFi3, ifield]
                            + self.data[ileaf, ixFi1 + 1, ixFi2 + 1, ixFi3, ifield]
                            + self.data[ileaf, ixFi1, ixFi2, ixFi3 + 1, ifield]
                            + self.data[ileaf, ixFi1 + 1, ixFi2, ixFi3 + 1, ifield]
                            + self.data[ileaf, ixFi1, ixFi2 + 1, ixFi3 + 1, ifield]
                            + self.data[ileaf, ixFi1 + 1, ixFi2 + 1, ixFi3 + 1, ifield]
                        ) * CoFiratio
                        self.datac[ileaf, ixCo1, ixCo2, ixCo3, ifield] = sum_value

    cdef void fill_coarse_boundary(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t[:,:] neighbor_type):
        cdef uint32_t idim, i
        cdef uint32_t ins[3]
        cdef int ixBmin[3]
        cdef int ixBmax[3]
        cdef int iside
        cdef int iis[3]
        cdef bint should_continue

        ins[0] = i1
        ins[1] = i2
        ins[2] = i3

        for idim in range(self.ndim):
            for i in range(self.ndim):
                ixBmin[i] = self.ixCoGmin[i] + (self.ng if i != idim else 0)
                ixBmax[i] = self.ixCoGmax[i] - (self.ng if i != idim else 0)

            if (idim > 0) and neighbor_type[ileaf, self.nindex(0, 1, 1)] == 1:
                ixBmin[0] = self.ixCoGmin[0]
            if (idim > 0) and neighbor_type[ileaf, self.nindex(2, 1, 1)] == 1:
                ixBmax[0] = self.ixCoGmax[0]
            if (idim > 1) and neighbor_type[ileaf, self.nindex(1, 0, 1)] == 1:
                ixBmin[1] = self.ixCoGmin[1]
            if (idim > 1) and neighbor_type[ileaf, self.nindex(1, 2, 1)] == 1:
                ixBmax[1] = self.ixCoGmax[1]

            for i in range(self.ndim):
                if ins[i] == 0:
                    ixBmin[i] = self.ixCoGmin[i] + self.ng
                    ixBmax[i] = self.ixCoGmin[i] + 2 * self.ng - 1
                elif ins[i] == 2:
                    ixBmin[i] = self.ixCoGmax[i] - 2 * self.ng + 1
                    ixBmax[i] = self.ixCoGmax[i] - self.ng

            for iside in range(2):
                iis[0] = 1 + (2 * iside - 1) * (idim == 0)
                iis[1] = 1 + (2 * iside - 1) * (idim == 1)
                iis[2] = 1 + (2 * iside - 1) * (idim == 2)

                should_continue = False
                for i in range(self.ndim):
                    if abs(<int>ins[i] - 1) == 1 and abs(iis[i] - 1) == 1:
                        should_continue = True
                        break
                if should_continue:
                    continue

                if neighbor_type[ileaf, self.nindex(iis[0], iis[1], iis[2])] != 1:
                    continue

                self.bc_phys(iside, idim, ileaf, ixBmin, ixBmax, True)

    cdef void identifyphysbound(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):
        cdef uint32_t idim
        cdef uint32_t low_i1, low_i2, low_i3
        cdef uint32_t high_i1, high_i2, high_i3

        for idim in range(self.ndim):
            low_i1 = 1 - (idim == 0)
            low_i2 = 1 - (idim == 1)
            low_i3 = 1 - (idim == 2)
            high_i1 = 1 + (idim == 0)
            high_i2 = 1 + (idim == 1)
            high_i3 = 1 + (idim == 2)

            if (
                neighbor_type[ileaf, self.nindex(low_i1, low_i2, low_i3)] == 1
                and neighbor_type[ileaf, self.nindex(high_i1, high_i2, high_i3)] == 1
            ):
                self.idphyb[ileaf, idim] = 2
            elif neighbor_type[ileaf, self.nindex(low_i1, low_i2, low_i3)] == 1:
                self.idphyb[ileaf, idim] = -1
            elif neighbor_type[ileaf, self.nindex(high_i1, high_i2, high_i3)] == 1:
                self.idphyb[ileaf, idim] = 1
            else:
                self.idphyb[ileaf, idim] = 0

    cdef void copy_data_to_data(self, uint32_t src_leaf, uint32_t dst_leaf, int src_min[3], int src_max[3], int dst_min[3]):
        cdef int i, j, k, ifield
        cdef int di, dj, dk

        for i in range(src_min[0], src_max[0] + 1):
            di = dst_min[0] + i - src_min[0]
            for j in range(src_min[1], src_max[1] + 1):
                dj = dst_min[1] + j - src_min[1]
                for k in range(src_min[2], src_max[2] + 1):
                    dk = dst_min[2] + k - src_min[2]
                    for ifield in range(self.nfields):
                        self.data[dst_leaf, di, dj, dk, ifield] = self.data[src_leaf, i, j, k, ifield]

    cdef void copy_datac_to_data(self, uint32_t src_leaf, uint32_t dst_leaf, int src_min[3], int src_max[3], int dst_min[3]):
        cdef int i, j, k, ifield
        cdef int di, dj, dk

        for i in range(src_min[0], src_max[0] + 1):
            di = dst_min[0] + i - src_min[0]
            for j in range(src_min[1], src_max[1] + 1):
                dj = dst_min[1] + j - src_min[1]
                for k in range(src_min[2], src_max[2] + 1):
                    dk = dst_min[2] + k - src_min[2]
                    for ifield in range(self.nfields):
                        self.data[dst_leaf, di, dj, dk, ifield] = self.datac[src_leaf, i, j, k, ifield]

    cdef void copy_data_to_datac(self, uint32_t src_leaf, uint32_t dst_leaf, int src_min[3], int src_max[3], int dst_min[3]):
        cdef int i, j, k, ifield
        cdef int di, dj, dk

        for i in range(src_min[0], src_max[0] + 1):
            di = dst_min[0] + i - src_min[0]
            for j in range(src_min[1], src_max[1] + 1):
                dj = dst_min[1] + j - src_min[1]
                for k in range(src_min[2], src_max[2] + 1):
                    dk = dst_min[2] + k - src_min[2]
                    for ifield in range(self.nfields):
                        self.datac[dst_leaf, di, dj, dk, ifield] = self.data[src_leaf, i, j, k, ifield]

    cdef void bc_fill_srl(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3):
        cdef int ineighbor
        cdef int iis[3]
        cdef int iibs[3]
        cdef int n_is[3]
        cdef int ixSmin[3]
        cdef int ixSmax[3]
        cdef int ixRmin[3]
        cdef int ixRmax[3]
        cdef int i

        ineighbor = <int>self.forest.neighbor_index[ileaf, self.nindex(i1, i2, i3)] - 1
        if ineighbor < 0:
            return

        iis[0] = i1
        iis[1] = i2
        iis[2] = i3
        iibs[0] = self.idphyb[ileaf, 0]
        iibs[1] = self.idphyb[ileaf, 1]
        iibs[2] = self.idphyb[ileaf, 2]
        n_is[0] = 2 - iis[0]
        n_is[1] = 2 - iis[1]
        n_is[2] = 2 - iis[2]

        for i in range(3):
            ixSmin[i] = self.ixS_srl_min[i][iibs[i] + 1][iis[i]]
            ixSmax[i] = self.ixS_srl_max[i][iibs[i] + 1][iis[i]]
            ixRmin[i] = self.ixR_srl_min[i][iibs[i] + 1][n_is[i]]
            ixRmax[i] = self.ixR_srl_max[i][iibs[i] + 1][n_is[i]]

        self.copy_data_to_data(ileaf, <uint32_t>ineighbor, ixSmin, ixSmax, ixRmin)

    cdef void bc_fill_restrict(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3):
        cdef treeptr nodeptr
        cdef int iis[3]
        cdef int iibs[3]
        cdef int ics[3]
        cdef int n_incs[3]
        cdef int ixSmin[3]
        cdef int ixSmax[3]
        cdef int ixRmin[3]
        cdef int ixRmax[3]
        cdef int i
        cdef int ineighbor

        nodeptr = self.forest.sfc2node[ileaf]

        iis[0] = i1
        iis[1] = i2
        iis[2] = i3
        iibs[0] = self.idphyb[ileaf, 0]
        iibs[1] = self.idphyb[ileaf, 1]
        iibs[2] = self.idphyb[ileaf, 2]

        for i in range(3):
            ics[i] = 1 + nodeptr.node.ig[i] % 2

        if not ((i1 == 1 or i1 == 2 * ics[0] - 2) and (i2 == 1 or i2 == 2 * ics[1] - 2) and (i3 == 1 or i3 == 2 * ics[2] - 2)):
            return

        ineighbor = <int>self.forest.neighbor_index[ileaf, self.nindex(i1, i2, i3)] - 1
        if ineighbor < 0:
            return

        for i in range(3):
            n_incs[i] = -2 * (iis[i] - 1) + ics[i]
            ixSmin[i] = self.ixS_r_min[i][iibs[i] + 1][iis[i]]
            ixSmax[i] = self.ixS_r_max[i][iibs[i] + 1][iis[i]]
            ixRmin[i] = self.ixR_r_min[i][iibs[i] + 1][n_incs[i]]
            ixRmax[i] = self.ixR_r_max[i][iibs[i] + 1][n_incs[i]]

        self.copy_datac_to_data(ileaf, <uint32_t>ineighbor, ixSmin, ixSmax, ixRmin)

    cdef void bc_fill_prolong(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3):
        cdef int iibs[3]
        cdef int ic1, ic2, ic3
        cdef int inc1, inc2, inc3
        cdef int n_i1, n_i2, n_i3
        cdef int n_inc1, n_inc2, n_inc3
        cdef int ixSmin[3]
        cdef int ixSmax[3]
        cdef int ixRmin[3]
        cdef int ixRmax[3]
        cdef int ineighbor
        cdef int incs[3]
        cdef int n_incs[3]
        cdef int d
        cdef int ic1_start, ic1_stop
        cdef int ic2_start, ic2_stop
        cdef int ic3_start, ic3_stop

        iibs[0] = self.idphyb[ileaf, 0]
        iibs[1] = self.idphyb[ileaf, 1]
        iibs[2] = self.idphyb[ileaf, 2]

        ic1_start = 1 + (2 - <int>i1) // 2
        ic1_stop = 2 - (<int>i1) // 2 + 1
        ic2_start = 1 + (2 - <int>i2) // 2
        ic2_stop = 2 - (<int>i2) // 2 + 1
        ic3_start = 1 + (2 - <int>i3) // 2
        ic3_stop = 2 - (<int>i3) // 2 + 1

        for ic3 in range(ic3_start, ic3_stop):
            inc3 = 2 * (<int>i3 - 1) + ic3
            for ic2 in range(ic2_start, ic2_stop):
                inc2 = 2 * (<int>i2 - 1) + ic2
                for ic1 in range(ic1_start, ic1_stop):
                    inc1 = 2 * (<int>i1 - 1) + ic1

                    ineighbor = <int>self.forest.neighbor_children[ileaf, self.ncindex(inc1, inc2, inc3)] - 1
                    if ineighbor < 0:
                        continue

                    n_i1 = 1 - <int>i1
                    n_i2 = 1 - <int>i2
                    n_i3 = 1 - <int>i3
                    n_inc1 = ic1 + n_i1
                    n_inc2 = ic2 + n_i2
                    n_inc3 = ic3 + n_i3
                    incs[0] = inc1
                    incs[1] = inc2
                    incs[2] = inc3
                    n_incs[0] = n_inc1
                    n_incs[1] = n_inc2
                    n_incs[2] = n_inc3

                    for d in range(3):
                        ixSmin[d] = self.ixS_p_min[d][iibs[d] + 1][incs[d]]
                        ixSmax[d] = self.ixS_p_max[d][iibs[d] + 1][incs[d]]
                        ixRmin[d] = self.ixR_p_min[d][iibs[d] + 1][n_incs[d]]
                        ixRmax[d] = self.ixR_p_max[d][iibs[d] + 1][n_incs[d]]

                    self.copy_data_to_datac(ileaf, <uint32_t>ineighbor, ixSmin, ixSmax, ixRmin)

    cdef void gc_prolong(self, uint32_t ileaf, uint32_t[:,:] neighbor_type):
        cdef uint32_t i1, i2, i3

        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    if neighbor_type[ileaf, self.nindex(i1, i2, i3)] == 2:
                        self.bc_prolong(ileaf, i1, i2, i3)

    cdef void bc_prolong(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3):
        cdef int iis[3]
        cdef int iibs[3]
        cdef int ixFimin[3]
        cdef int ixFimax[3]
        cdef int i
        cdef double dxFi[3]
        cdef double dxCo[3]
        cdef double invdxCo[3]
        cdef double xFimin[3]
        cdef double xComin[3]

        iis[0] = i1
        iis[1] = i2
        iis[2] = i3
        iibs[0] = self.idphyb[ileaf, 0]
        iibs[1] = self.idphyb[ileaf, 1]
        iibs[2] = self.idphyb[ileaf, 2]

        for i in range(3):
            ixFimin[i] = self.ixR_srl_min[i][iibs[i] + 1][iis[i]]
            ixFimax[i] = self.ixR_srl_max[i][iibs[i] + 1][iis[i]]
            dxFi[i] = self.rnode[ileaf, 6 + i]
            dxCo[i] = 2.0 * dxFi[i]
            invdxCo[i] = 1.0 / dxCo[i]
            xFimin[i] = self.rnode[ileaf, i] - self.ng * dxFi[i]
            xComin[i] = self.rnode[ileaf, i] - self.ng * dxCo[i]

        self.interpolation_linear(ileaf, ixFimin, ixFimax, dxFi, xFimin, dxCo, invdxCo, xComin)

    cdef void interpolation_linear(
        self,
        uint32_t ileaf,
        int ixFimin[3],
        int ixFimax[3],
        double dxFi[3],
        double xFimin[3],
        double dxCo[3],
        double invdxCo[3],
        double xComin[3],
    ):
        cdef int ixFi1, ixFi2, ixFi3
        cdef int ixCo1, ixCo2, ixCo3
        cdef int ifield
        cdef double xFi1, xFi2, xFi3
        cdef double xCo1, xCo2, xCo3
        cdef double eta1, eta2, eta3
        cdef double value
        cdef double center_value

        for ixFi1 in range(ixFimin[0], ixFimax[0] + 1):
            xFi1 = xFimin[0] + (ixFi1 + 0.5) * dxFi[0]
            ixCo1 = <int>((xFi1 - xComin[0]) * invdxCo[0])
            xCo1 = xComin[0] + (ixCo1 + 0.5) * dxCo[0]
            eta1 = (xFi1 - xCo1) * invdxCo[0]
            for ixFi2 in range(ixFimin[1], ixFimax[1] + 1):
                xFi2 = xFimin[1] + (ixFi2 + 0.5) * dxFi[1]
                ixCo2 = <int>((xFi2 - xComin[1]) * invdxCo[1])
                xCo2 = xComin[1] + (ixCo2 + 0.5) * dxCo[1]
                eta2 = (xFi2 - xCo2) * invdxCo[1]
                for ixFi3 in range(ixFimin[2], ixFimax[2] + 1):
                    xFi3 = xFimin[2] + (ixFi3 + 0.5) * dxFi[2]
                    ixCo3 = <int>((xFi3 - xComin[2]) * invdxCo[2])
                    xCo3 = xComin[2] + (ixCo3 + 0.5) * dxCo[2]
                    eta3 = (xFi3 - xCo3) * invdxCo[2]

                    for ifield in range(self.nfields):
                        center_value = self.datac[ileaf, ixCo1, ixCo2, ixCo3, ifield]
                        value = center_value
                        value += _limited_slope(
                            self.datac[ileaf, ixCo1 - 1, ixCo2, ixCo3, ifield],
                            center_value,
                            self.datac[ileaf, ixCo1 + 1, ixCo2, ixCo3, ifield],
                        ) * eta1
                        value += _limited_slope(
                            self.datac[ileaf, ixCo1, ixCo2 - 1, ixCo3, ifield],
                            center_value,
                            self.datac[ileaf, ixCo1, ixCo2 + 1, ixCo3, ifield],
                        ) * eta2
                        value += _limited_slope(
                            self.datac[ileaf, ixCo1, ixCo2, ixCo3 - 1, ifield],
                            center_value,
                            self.datac[ileaf, ixCo1, ixCo2, ixCo3 + 1, ifield],
                        ) * eta3
                        self.data[ileaf, ixFi1, ixFi2, ixFi3, ifield] = value
