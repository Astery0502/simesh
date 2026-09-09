# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Coordinate-phase ghost arithmetic on explicitly borrowed NumPy buffers.

No forest object, file, allocation owner or cache is retained by a published
field. The geometry builder consumes the shared integer forest. Numerical
methods are adapted from the pinned canonical implementation; see ASSETS.md.
"""
from libc.stdint cimport uint32_t, int64_t
from libc.math cimport isfinite
from cython.parallel cimport prange
from .primitives._contacts cimport refined_contact_target_c

cdef int BC_CONT = 0
cdef int BC_SYMM = 1
cdef int BC_ASYMM = 2
cdef int BC_NOINFLOW = 3

cdef extern from *:
    """
    #ifdef _OPENMP
    #define SIMESH_COORDINATE_OPENMP _OPENMP
    #else
    #define SIMESH_COORDINATE_OPENMP 0
    #endif
    """
    int SIMESH_COORDINATE_OPENMP


def openmp_build_info():
    return {"enabled": SIMESH_COORDINATE_OPENMP != 0,
            "openmp_version": SIMESH_COORDINATE_OPENMP}


cpdef int64_t first_unrepresentable_prolongation(
    const double[:,::1] rnode, const uint32_t[::1] block,
    const uint32_t[:,::1] neighbor_type,
):
    """Check geometry once before unchecked slope-neighbor array access."""
    cdef int64_t leaf
    cdef int axis, direction, cell
    cdef bint needed
    cdef double fine_dx, coarse_dx, inverse, fine_lower, coarse_lower, point, q
    with nogil:
        for leaf in range(rnode.shape[0]):
            needed = False
            for direction in range(27):
                if neighbor_type[leaf,direction] == 2:
                    needed = True
                    break
            if not needed:
                continue
            for axis in range(3):
                fine_dx = rnode[leaf,6+axis]
                coarse_dx = 2.0*fine_dx
                inverse = 1.0/coarse_dx
                fine_lower = rnode[leaf,axis]-2*fine_dx
                coarse_lower = rnode[leaf,axis]-2*coarse_dx
                if (fine_dx <= 0 or not isfinite(inverse) or
                        not isfinite(fine_lower) or not isfinite(coarse_lower)):
                    return leaf
                for cell in range(block[axis]+4):
                    point = fine_lower+(cell+0.5)*fine_dx
                    q = (point-coarse_lower)*inverse
                    # Integer truncation must leave both slope neighbors valid.
                    if not isfinite(q) or q < 1.0 or q >= block[axis]//2+3:
                        return leaf
    return -1


cpdef void fill_geometry(
    const int64_t[::1] root_shape, const int64_t[:, :, ::1] ranks,
    const int64_t[::1] roots, const int64_t[::1] levels,
    const int64_t[:, ::1] coordinates, const int64_t[:, ::1] children,
    const int64_t[::1] node_leaves, const int64_t[::1] leaf_nodes,
    const uint32_t[::1] block, const double[::1] lower, const double[::1] upper,
    uint32_t[:, ::1] neighbor_type, uint32_t[:, ::1] neighbor_index,
    uint32_t[:, ::1] neighbor_children, double[:, ::1] rnode,
    int64_t[:, ::1] leaf_coordinates, int[:, ::1] physical_sides,
):
    """Unchecked builder for validated Cartesian 3D mesh/owned output arrays."""
    cdef int64_t leaf, node, target, target_leaf, child_node
    cdef uint32_t level
    cdef int axis, i, j, k, row, child, bx, by, bz, x, y, z
    cdef int ig[3]
    cdef bint low, high
    with nogil:
        for leaf in range(leaf_nodes.shape[0]):
            node = leaf_nodes[leaf]
            level = <uint32_t>levels[node]
            for axis in range(3):
                ig[axis] = <int>coordinates[node, axis]
                leaf_coordinates[leaf, axis] = coordinates[node, axis]
                # Preserve the canonical coordinate arithmetic and its order.
                rnode[leaf, axis] = (ig[axis] * (upper[axis]-lower[axis]) /
                    2 ** (level-1) / root_shape[axis] + lower[axis])
                rnode[leaf, 3+axis] = (rnode[leaf, axis] +
                    (upper[axis]-lower[axis]) / 2 ** (level-1) / root_shape[axis])
                rnode[leaf, 6+axis] = ((upper[axis]-lower[axis]) /
                    2 ** (level-1) / (root_shape[axis]*block[axis]))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        row = i+3*j+9*k
                        if row == 13:
                            neighbor_type[leaf, row] = 0
                            neighbor_index[leaf, row] = <uint32_t>(leaf+1)
                            continue
                        target = refined_contact_target_c(root_shape, ranks, roots, levels,
                            coordinates, children, node_leaves, leaf_nodes, leaf, i-1, j-1, k-1)
                        if target < 0:
                            neighbor_type[leaf, row] = 1
                            continue
                        target_leaf = node_leaves[target]
                        if target_leaf >= 0:
                            neighbor_type[leaf, row] = 2 if levels[target] < level else 3
                            neighbor_index[leaf, row] = <uint32_t>(target_leaf+1)
                        else:
                            neighbor_type[leaf, row] = 4
                            for child in range(8):
                                bx = child & 1
                                by = (child >> 1) & 1
                                bz = (child >> 2) & 1
                                if (i == 0 and bx != 1) or (i == 2 and bx != 0):
                                    continue
                                if (j == 0 and by != 1) or (j == 2 and by != 0):
                                    continue
                                if (k == 0 and bz != 1) or (k == 2 and bz != 0):
                                    continue
                                x = 2*(i-1)+bx+1
                                y = 2*(j-1)+by+1
                                z = 2*(k-1)+bz+1
                                child_node = children[target, child]
                                neighbor_children[leaf, x+4*y+16*z] = <uint32_t>(node_leaves[child_node]+1)
            for axis in range(3):
                row = 1 if axis == 0 else (3 if axis == 1 else 9)
                low = neighbor_type[leaf, 13-row] == 1
                high = neighbor_type[leaf, 13+row] == 1
                physical_sides[leaf, axis] = 2 if low and high else (-1 if low else (1 if high else 0))

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



cdef class ExchangeBindings:
    """Private invocation bindings; all large storage is owned by the caller."""
    cdef uint32_t ndim, ng, nfields, nleafs, ngCo, interpolation_order
    cdef uint32_t bsize[3]
    cdef uint32_t bCosize[3]
    cdef int normal_velocity_field[3]
    cdef int ixGmin[3]
    cdef int ixGmax[3]
    cdef int ixMmin[3]
    cdef int ixMmax[3]
    cdef int ixCoGmin[3]
    cdef int ixCoGmax[3]
    cdef int ixCoMmin[3]
    cdef int ixCoMmax[3]
    cdef int ixS_srl_min[3][4][3]
    cdef int ixS_srl_max[3][4][3]
    cdef int ixR_srl_min[3][4][3]
    cdef int ixR_srl_max[3][4][3]
    cdef int ixS_r_min[3][3][3]
    cdef int ixS_r_max[3][3][3]
    cdef int ixR_r_min[3][3][4]
    cdef int ixR_r_max[3][3][4]
    cdef int ixS_p_min[3][4][4]
    cdef int ixS_p_max[3][4][4]
    cdef int ixR_p_min[3][4][4]
    cdef int ixR_p_max[3][4][4]
    cdef const uint32_t[:,::1] neighbor_type, neighbor_index, neighbor_children
    cdef const int64_t[:,::1] coordinates
    cdef const double[:,::1] rnode
    cdef const int[:,::1] idphyb, bc_type
    cdef double[:,:,:,:,::1] data, datac

    def __cinit__(self, const uint32_t[::1] block,
                  const uint32_t[:,::1] neighbor_type,
                  const uint32_t[:,::1] neighbor_index,
                  const uint32_t[:,::1] neighbor_children,
                  const int64_t[:,::1] coordinates,
                  const double[:,::1] rnode, const int[:,::1] idphyb,
                  double[:,:,:,:,::1] data, double[:,:,:,:,::1] coarse,
                  const int[:,::1] boundary_modes):
        cdef int axis
        self.ndim, self.ng = 3, 2
        self.nleafs, self.nfields = data.shape[0], data.shape[4]
        if (block.shape[0] != 3 or neighbor_type.shape[0] != self.nleafs or
                neighbor_type.shape[1] != 27 or neighbor_index.shape[0] != self.nleafs or
                neighbor_index.shape[1] != 27 or neighbor_children.shape[0] != self.nleafs or
                neighbor_children.shape[1] != 64 or coordinates.shape[0] != self.nleafs or
                coordinates.shape[1] != 3 or rnode.shape[0] != self.nleafs or rnode.shape[1] != 9 or
                idphyb.shape[0] != self.nleafs or idphyb.shape[1] != 3 or
                boundary_modes.shape[0] != self.nfields or boundary_modes.shape[1] != 6 or
                coarse.shape[4] != self.nfields or coarse.shape[0] not in (0, self.nleafs)):
            raise ValueError("coordinate exchange bindings do not match")
        for axis in range(3):
            self.bsize[axis] = block[axis]
            self.bCosize[axis] = block[axis]//2+4
            self.normal_velocity_field[axis] = -1
            if (block[axis] < 4 or block[axis] % 2 or data.shape[axis+1] != block[axis]+4 or
                    coarse.shape[axis+1] != self.bCosize[axis]):
                raise ValueError("coordinate exchange requires matching even padded blocks")
        self.neighbor_type, self.neighbor_index = neighbor_type, neighbor_index
        self.neighbor_children, self.coordinates = neighbor_children, coordinates
        self.rnode, self.idphyb = rnode, idphyb
        self.data, self.datac, self.bc_type = data, coarse, boundary_modes
        self._init_block_gridindex()

    def local_phase(self, int phase, uint32_t first, uint32_t last):
        if phase < 0 or phase > 3 or first > last or last > self.nleafs:
            raise ValueError("invalid coordinate phase range")
        with nogil:
            self._local_phase_range(phase, first, last)

    def openmp_phase(self, int phase, int workers):
        cdef int worker
        cdef uint32_t first, last
        if phase < 0 or phase > 3 or workers < 1 or not SIMESH_COORDINATE_OPENMP:
            raise ValueError("an enabled OpenMP build and valid phase/workers are required")
        for worker in prange(workers, nogil=True, num_threads=workers, schedule='static'):
            first = self.nleafs*worker//workers
            last = self.nleafs*(worker+1)//workers
            self._local_phase_range(phase, first, last)

    def exchange_same_and_restrict(self):
        cdef uint32_t leaf, i, j, k, kind
        # Retain original cross-target write order. These are not parallel tasks.
        for leaf in range(self.nleafs):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        kind = self.neighbor_type[leaf, self.nindex(i,j,k)]
                        if kind == 2:
                            self.bc_fill_restrict(leaf,i,j,k)
                        elif kind == 3:
                            self.bc_fill_srl(leaf,i,j,k)

    def exchange_coarse_support(self):
        cdef uint32_t leaf, i, j, k
        for leaf in range(self.nleafs):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if self.neighbor_type[leaf, self.nindex(i,j,k)] == 4:
                            self.bc_fill_prolong(leaf,i,j,k)

    cdef void _init_block_gridindex(self):
        cdef uint32_t i, j

        self.ngCo = (self.ng + 1) // 2
        self.interpolation_order = 2

        for i in range(3):
            self.ixGmin[i] = 0
            self.ixGmax[i] = self.bsize[i] + 2 * self.ng - 1
            self.ixMmin[i] = self.ixGmin[i] + self.ng
            self.ixMmax[i] = self.ixGmax[i] - self.ng

            self.ixCoGmin[i] = 0
            self.ixCoGmax[i] = self.bCosize[i] - 1
            self.ixCoMmin[i] = self.ixCoGmin[i] + self.ng
            self.ixCoMmax[i] = self.ixCoGmax[i] - self.ng

        for i in range(3):
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

        for i in range(3):
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

        for i in range(3):
            for j in range(4):
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

        for i in range(3):
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

            # A coarse root may span both physical sides on a transverse axis.
            # idphyb=2 selects row 3: extend BOTH ends of that support. The
            # legacy three-row tables indexed outside their declared extent.
            self.ixS_p_min[i][3][1] = self.ixGmin[i]
            self.ixS_p_max[i][3][2] = self.ixGmax[i]
            self.ixR_p_min[i][3][1] = self.ixGmin[i]
            self.ixR_p_max[i][3][2] = self.ixCoGmax[i]

    cdef inline uint32_t nindex(self, uint32_t n1, uint32_t n2, uint32_t n3) noexcept nogil:
        if self.ndim == 2:
            return n1 + n2 * 3
        return n1 + n2 * 3 + n3 * 3 * 3

    cdef inline uint32_t ncindex(self, uint32_t nc1, uint32_t nc2, uint32_t nc3) noexcept nogil:
        if self.ndim == 2:
            return nc1 + nc2 * 4
        return nc1 + nc2 * 4 + nc3 * 4 * 4

    cdef bint is_boundary(self, uint32_t ileaf, const uint32_t[:,::1] neighbor_type) noexcept nogil:
        cdef uint32_t i

        for i in range(3 ** self.ndim):
            if neighbor_type[ileaf, i] == 1:
                return True
        return False

    cdef void _local_phase_range(self, int phase, uint32_t first, uint32_t last) noexcept nogil:
        cdef uint32_t ileaf
        cdef const uint32_t[:,::1] neighbors = self.neighbor_type
        if phase == 0:
            for ileaf in range(first, last):
                self.fill_boundary_before_gc(ileaf, neighbors)
        elif phase == 1:
            for ileaf in range(first, last):
                self._coarsen_leaf(ileaf, neighbors)
        elif phase == 2:
            for ileaf in range(first, last):
                self.gc_prolong(ileaf, neighbors)
        else:
            for ileaf in range(first, last):
                self.fill_boundary_after_gc(ileaf, neighbors)

    cdef void _coarsen_leaf(self, uint32_t ileaf, const uint32_t[:,::1] neighbor_type) noexcept nogil:
        cdef uint32_t i, j, k
        cdef uint32_t k_stop = 1 if self.ndim == 2 else 3
        cdef bint isboundary
        for i in range(3 ** self.ndim):
            if neighbor_type[ileaf, i] == 2:
                self.coarsen_grid(ileaf)
                break

        isboundary = self.is_boundary(ileaf, neighbor_type)
        for i in range(3):
            for j in range(3):
                for k in range(k_stop):
                    if neighbor_type[ileaf, self.nindex(i, j, k)] == 2 and isboundary:
                        self.fill_coarse_boundary(ileaf, i, j, k, neighbor_type)

    cdef void fill_boundary_before_gc(self, uint32_t ileaf, const uint32_t[:,::1] neighbor_type) noexcept nogil:
        cdef uint32_t idim, i
        cdef int ixBmin[3]
        cdef int ixBmax[3]
        cdef int i1, i2, i3, iside

        for idim in range(self.ndim):
            for i in range(self.ndim):
                ixBmin[i] = self.ixGmin[i] + (self.ng if i != idim else 0)
                ixBmax[i] = self.ixGmax[i] - (self.ng if i != idim else 0)
            if self.ndim == 2:
                ixBmin[2] = self.ixMmin[2]
                ixBmax[2] = self.ixMmax[2]

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

    cdef void fill_boundary_after_gc(self, uint32_t ileaf, const uint32_t[:,::1] neighbor_type) noexcept nogil:
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
            if self.ndim == 2:
                ixBmin[2] = self.ixMmin[2]
                ixBmax[2] = self.ixMmax[2]

            for iside in range(2):
                i1 = 1 + (2 * iside - 1) * (idim == 0)
                i2 = 1 + (2 * iside - 1) * (idim == 1)
                i3 = 1 + (2 * iside - 1) * (idim == 2)
                if neighbor_type[ileaf, self.nindex(i1, i2, i3)] != 1:
                    continue
                self.bc_phys(iside, idim, ileaf, ixBmin, ixBmax, False)

    cdef void bc_phys(self, int iside, uint32_t idim, uint32_t ileaf, int ixBmin[3], int ixBmax[3], bint is_coarse) noexcept nogil:
        cdef int ixOmin[3]
        cdef int ixOmax[3]
        cdef int ixImin[3]
        cdef int ixImax[3]
        cdef uint32_t idir
        cdef int i1, i2, i3, o1, o2, o3, ifield
        cdef int side_index, mode
        cdef double value
        cdef double[:,:,:,:,::1] data_array

        if is_coarse:
            data_array = self.datac
        else:
            data_array = self.data

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
        if self.ndim == 2:
            ixOmin[2] = ixBmin[2]
            ixOmax[2] = ixBmax[2]
            ixImin[2] = ixBmin[2]
            ixImax[2] = ixBmax[2]

        for o1 in range(ixOmin[0], ixOmax[0] + 1):
            if idim != 0:
                i1 = o1
            elif iside == 1:
                i1 = ixImin[0]
            else:
                i1 = ixImin[0]
            for o2 in range(ixOmin[1], ixOmax[1] + 1):
                if idim != 1:
                    i2 = o2
                elif iside == 1:
                    i2 = ixImin[1]
                else:
                    i2 = ixImin[1]
                for o3 in range(ixOmin[2], ixOmax[2] + 1):
                    if idim != 2:
                        i3 = o3
                    elif iside == 1:
                        i3 = ixImin[2]
                    else:
                        i3 = ixImin[2]
                    for ifield in range(self.nfields):
                        side_index = 2 * idim + iside
                        mode = self.bc_type[ifield, side_index]
                        if mode == BC_SYMM or mode == BC_ASYMM:
                            if idim == 0:
                                if iside == 1:
                                    i1 = ixImin[0] - (o1 - ixOmin[0])
                                else:
                                    i1 = ixImin[0] + (ixOmax[0] - o1)
                            elif idim == 1:
                                if iside == 1:
                                    i2 = ixImin[1] - (o2 - ixOmin[1])
                                else:
                                    i2 = ixImin[1] + (ixOmax[1] - o2)
                            else:
                                if iside == 1:
                                    i3 = ixImin[2] - (o3 - ixOmin[2])
                                else:
                                    i3 = ixImin[2] + (ixOmax[2] - o3)
                        else:
                            if idim == 0:
                                i1 = ixImin[0]
                            elif idim == 1:
                                i2 = ixImin[1]
                            else:
                                i3 = ixImin[2]

                        value = data_array[ileaf, i1, i2, i3, ifield]
                        if mode == BC_ASYMM:
                            value = -value
                        elif mode == BC_NOINFLOW and ifield == self.normal_velocity_field[idim]:
                            if iside == 1 and value < 0.0:
                                value = 0.0
                            elif iside == 0 and value > 0.0:
                                value = 0.0
                        data_array[ileaf, o1, o2, o3, ifield] = value

    cdef void coarsen_grid(self, uint32_t ileaf) noexcept nogil:
        cdef int ixCo1, ixCo2, ixCo3
        cdef uint32_t ixFi1, ixFi2, ixFi3
        cdef uint32_t ifield
        cdef double sum_value
        cdef double CoFiratio = 0.125

        if self.ndim == 2:
            CoFiratio = 0.25
            ixCo3 = self.ixCoMmin[2]
            ixFi3 = self.ixMmin[2]
            for ixCo1 in range(self.ixCoMmin[0], self.ixCoMmax[0] + 1):
                ixFi1 = (ixCo1 - self.ixCoMmin[0]) * 2 + self.ixMmin[0]
                for ixCo2 in range(self.ixCoMmin[1], self.ixCoMmax[1] + 1):
                    ixFi2 = (ixCo2 - self.ixCoMmin[1]) * 2 + self.ixMmin[1]
                    for ifield in range(self.nfields):
                        sum_value = (
                            self.data[ileaf, ixFi1, ixFi2, ixFi3, ifield]
                            + self.data[ileaf, ixFi1 + 1, ixFi2, ixFi3, ifield]
                            + self.data[ileaf, ixFi1, ixFi2 + 1, ixFi3, ifield]
                            + self.data[ileaf, ixFi1 + 1, ixFi2 + 1, ixFi3, ifield]
                        ) * CoFiratio
                        self.datac[ileaf, ixCo1, ixCo2, ixCo3, ifield] = sum_value
            return

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

    cdef void fill_coarse_boundary(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3, const uint32_t[:,::1] neighbor_type) noexcept nogil:
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
            if self.ndim == 2:
                ixBmin[2] = self.ixCoMmin[2]
                ixBmax[2] = self.ixCoMmax[2]

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

        ineighbor = <int>self.neighbor_index[ileaf, self.nindex(i1, i2, i3)] - 1
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
        if self.ndim == 2:
            ixSmin[2] = self.ixMmin[2]
            ixSmax[2] = self.ixMmax[2]
            ixRmin[2] = self.ixMmin[2]
            ixRmax[2] = self.ixMmax[2]

        self.copy_data_to_data(ileaf, <uint32_t>ineighbor, ixSmin, ixSmax, ixRmin)

    cdef void bc_fill_restrict(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3):
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


        iis[0] = i1
        iis[1] = i2
        iis[2] = i3
        iibs[0] = self.idphyb[ileaf, 0]
        iibs[1] = self.idphyb[ileaf, 1]
        iibs[2] = self.idphyb[ileaf, 2]

        for i in range(3):
            ics[i] = 1 + self.coordinates[ileaf,i] % 2

        if not ((i1 == 1 or i1 == 2 * ics[0] - 2) and (i2 == 1 or i2 == 2 * ics[1] - 2) and (i3 == 1 or i3 == 2 * ics[2] - 2)):
            return

        ineighbor = <int>self.neighbor_index[ileaf, self.nindex(i1, i2, i3)] - 1
        if ineighbor < 0:
            return

        for i in range(3):
            n_incs[i] = -2 * (iis[i] - 1) + ics[i]
            ixSmin[i] = self.ixS_r_min[i][iibs[i] + 1][iis[i]]
            ixSmax[i] = self.ixS_r_max[i][iibs[i] + 1][iis[i]]
            ixRmin[i] = self.ixR_r_min[i][iibs[i] + 1][n_incs[i]]
            ixRmax[i] = self.ixR_r_max[i][iibs[i] + 1][n_incs[i]]
        if self.ndim == 2:
            ixSmin[2] = self.ixCoMmin[2]
            ixSmax[2] = self.ixCoMmax[2]
            ixRmin[2] = self.ixMmin[2]
            ixRmax[2] = self.ixMmax[2]

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
        if self.ndim == 2:
            ic3_start = 0
            ic3_stop = 1

        for ic3 in range(ic3_start, ic3_stop):
            inc3 = 2 * (<int>i3 - 1) + ic3
            for ic2 in range(ic2_start, ic2_stop):
                inc2 = 2 * (<int>i2 - 1) + ic2
                for ic1 in range(ic1_start, ic1_stop):
                    inc1 = 2 * (<int>i1 - 1) + ic1

                    ineighbor = <int>self.neighbor_children[ileaf, self.ncindex(inc1, inc2, inc3)] - 1
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
                    if self.ndim == 2:
                        ixSmin[2] = self.ixMmin[2]
                        ixSmax[2] = self.ixMmax[2]
                        ixRmin[2] = self.ixCoMmin[2]
                        ixRmax[2] = self.ixCoMmax[2]

                    self.copy_data_to_datac(ileaf, <uint32_t>ineighbor, ixSmin, ixSmax, ixRmin)

    cdef void gc_prolong(self, uint32_t ileaf, const uint32_t[:,::1] neighbor_type) noexcept nogil:
        cdef uint32_t i1, i2, i3
        cdef uint32_t k_stop = 1 if self.ndim == 2 else 3

        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(k_stop):
                    if neighbor_type[ileaf, self.nindex(i1, i2, i3)] == 2:
                        self.bc_prolong(ileaf, i1, i2, i3)

    cdef void bc_prolong(self, uint32_t ileaf, uint32_t i1, uint32_t i2, uint32_t i3) noexcept nogil:
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

        for i in range(self.ndim):
            ixFimin[i] = self.ixR_srl_min[i][iibs[i] + 1][iis[i]]
            ixFimax[i] = self.ixR_srl_max[i][iibs[i] + 1][iis[i]]
            dxFi[i] = self.rnode[ileaf, 2 * self.ndim + i]
            dxCo[i] = 2.0 * dxFi[i]
            invdxCo[i] = 1.0 / dxCo[i]
            xFimin[i] = self.rnode[ileaf, i] - self.ng * dxFi[i]
            xComin[i] = self.rnode[ileaf, i] - self.ng * dxCo[i]
        if self.ndim == 2:
            ixFimin[2] = self.ixMmin[2]
            ixFimax[2] = self.ixMmax[2]
            dxFi[2] = 1.0
            dxCo[2] = 1.0
            invdxCo[2] = 1.0
            xFimin[2] = 0.0
            xComin[2] = 0.0

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
    ) noexcept nogil:
        cdef int ixFi1, ixFi2, ixFi3
        cdef int ixCo1, ixCo2, ixCo3
        cdef int ifield
        cdef double xFi1, xFi2, xFi3
        cdef double xCo1, xCo2, xCo3
        cdef double eta1, eta2, eta3
        cdef double value
        cdef double center_value

        if self.ndim == 2:
            ixFi3 = self.ixMmin[2]
            ixCo3 = self.ixCoMmin[2]
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

                    for ifield in range(self.nfields):
                        center_value = self.datac[ileaf, ixCo1, ixCo2, ixCo3, ifield]
                        value = (
                            center_value
                            + _limited_slope(
                                self.datac[ileaf, ixCo1 - 1, ixCo2, ixCo3, ifield],
                                center_value,
                                self.datac[ileaf, ixCo1 + 1, ixCo2, ixCo3, ifield],
                            ) * eta1
                            + _limited_slope(
                                self.datac[ileaf, ixCo1, ixCo2 - 1, ixCo3, ifield],
                                center_value,
                                self.datac[ileaf, ixCo1, ixCo2 + 1, ixCo3, ifield],
                            ) * eta2
                        )
                        self.data[ileaf, ixFi1, ixFi2, ixFi3, ifield] = value
            return

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
                        value = (
                            center_value
                            + _limited_slope(
                                self.datac[ileaf, ixCo1 - 1, ixCo2, ixCo3, ifield],
                                center_value,
                                self.datac[ileaf, ixCo1 + 1, ixCo2, ixCo3, ifield],
                            ) * eta1
                            + _limited_slope(
                                self.datac[ileaf, ixCo1, ixCo2 - 1, ixCo3, ifield],
                                center_value,
                                self.datac[ileaf, ixCo1, ixCo2 + 1, ixCo3, ifield],
                            ) * eta2
                            + _limited_slope(
                                self.datac[ileaf, ixCo1, ixCo2, ixCo3 - 1, ifield],
                                center_value,
                                self.datac[ileaf, ixCo1, ixCo2, ixCo3 + 1, ifield],
                            ) * eta3
                        )
                        self.data[ileaf, ixFi1, ixFi2, ixFi3, ifield] = value
