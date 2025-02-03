import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from simesh.geometry.amr.morton_order import level1_Morton_order
from simesh.geometry.amr.amr_forest import AMRForest
from simesh.frontends.amrvac.datio import get_header, get_forest, get_tree_info
from simesh.utils.octree import OctreeNode, OctreeNodePointer
from .mesh import Mesh

class AMRMesh(Mesh):
    """Adaptive Mesh Refinement (AMR) implementation that supports octree-based refinement."""

    def __init__(self, xrange: Tuple[float, float], 
                 yrange: Tuple[float, float], 
                 zrange: Tuple[float, float], 
                 field_names: List[str], 
                 block_nx: np.ndarray,
                 domain_nx: np.ndarray,                 
                 forest: AMRForest,
                 nghostcells: int = 2, # default is 2 ghostcell, not recommended to decrease
                 ):
        super().__init__(xrange, yrange, zrange, field_names)

        self.block_nx = block_nx
        self.domain_nx = domain_nx
        self.nblock = domain_nx // block_nx
        assert np.all(self.nblock * block_nx == domain_nx), "domain_nx must be divisible by block_nx"
        assert np.all(block_nx % 2 == 0), "block_nx must be even"

        self.nghostcells = nghostcells
        self.forest = forest
        self.nleafs = self.forest.nleafs

        self.data = np.zeros((self.nleafs, *(block_nx+2*nghostcells), len(field_names)))
        self.datac = np.zeros((self.nleafs, *((block_nx/2).astype(int)+2*nghostcells), len(field_names)))
        self.idphyb = np.zeros((3,self.nleafs)).astype(int)

        self._init_block_gridindex()
        self._init_coordinates()
        self._init_field_idx()

    def __post_init__(self):
        super().__post_init__()
    
    def _init_block_gridindex(self):
        """Initialize the grid indices for each block."""

        self.ixGmin = np.zeros(3).astype(int)
        self.ixGmax = np.asarray(self.block_nx).astype(int) + 2*self.nghostcells - 1
        self.ixMmin = self.ixGmin + self.nghostcells
        self.ixMmax = self.ixGmax - self.nghostcells

        """Also initiate the block grid index for the ghost cells"""
        ixCoGmin = self.ixGmin
        ixCoGmax = ((self.ixGmax-2*self.nghostcells+1)/2-1+2*self.nghostcells).astype(int)
        ixCoMmin = ixCoGmin + self.nghostcells
        ixCoMmax = ixCoGmax - self.nghostcells

        self.ixCoGmin = ixCoGmin
        self.ixCoGmax = ixCoGmax
        self.ixCoMmin = ixCoMmin
        self.ixCoMmax = ixCoMmax
        nghostcellsCo = int((self.nghostcells+1)/2)
        block_nxCo = np.asarray(self.block_nx/2).astype(int)

        interpolation_order = 2 # second order interpolation

        """Send (S) and Receive (R) min and max for the sibling neighbor blocks"""
        # the first is for 3 directions, the second is for :
        # 0: a block touches the lower boundary, 2 when touches the upper, 1 away from the boundary, 3 the both boundary
        # the third: 0 is for the minimum side of neighbor block, 2 is for the maximum, 1 is for the non-advance side of neighbor block
        self.ixS_srl_min = np.zeros((3,4,3)).astype(int)
        self.ixS_srl_max = np.zeros((3,4,3)).astype(int)
        self.ixR_srl_min = np.zeros((3,4,3)).astype(int)
        self.ixR_srl_max = np.zeros((3,4,3)).astype(int)

        self.ixS_srl_min[:,:,0] = self.ixMmin[:,np.newaxis]
        self.ixS_srl_min[:,:,1] = self.ixMmin[:,np.newaxis]
        self.ixS_srl_min[:,:,2] = self.ixMmax[:,np.newaxis]+1-self.nghostcells
        self.ixS_srl_max[:,:,0] = self.ixMmin[:,np.newaxis]-1+self.nghostcells
        self.ixS_srl_max[:,:,1] = self.ixMmax[:,np.newaxis]
        self.ixS_srl_max[:,:,2] = self.ixMmax[:,np.newaxis]

        self.ixR_srl_min[:,:,0] = self.ixGmin[:,np.newaxis]
        self.ixR_srl_min[:,:,1] = self.ixMmin[:,np.newaxis]
        self.ixR_srl_min[:,:,2] = self.ixMmax[:,np.newaxis]+1
        self.ixR_srl_max[:,:,0] = self.nghostcells-1
        self.ixR_srl_max[:,:,1] = self.ixMmax[:,np.newaxis]
        self.ixR_srl_max[:,:,2] = self.ixGmax[:,np.newaxis]

        """Send restricted (r) from finer (already coarsened)"""
        # the first is for 3 directions, the second is similar (case 3 not exist)
        # the third adds the two finer blocks range
        self.ixS_r_min = np.zeros((3,3,3)).astype(int)
        self.ixS_r_max = np.zeros((3,3,3)).astype(int)
        self.ixR_r_min = np.zeros((3,3,4)).astype(int)
        self.ixR_r_max = np.zeros((3,3,4)).astype(int)

        self.ixS_r_min[:,:,0] = ixCoMmin[:,np.newaxis]
        self.ixS_r_min[:,:,1] = ixCoMmin[:,np.newaxis]
        self.ixS_r_min[:,:,2] = ixCoMmax[:,np.newaxis]+1-self.nghostcells
        self.ixS_r_max[:,:,0] = ixCoMmin[:,np.newaxis]-1+self.nghostcells
        self.ixS_r_max[:,:,1] = ixCoMmax[:,np.newaxis]
        self.ixS_r_max[:,:,2] = ixCoMmax[:,np.newaxis]

        self.ixR_r_min[:,:,0] = self.ixGmin[:,np.newaxis]
        self.ixR_r_min[:,:,1] = self.ixMmin[:,np.newaxis]
        self.ixR_r_min[:,:,2] = self.ixMmin[:,np.newaxis]+block_nxCo[:,np.newaxis]
        self.ixR_r_min[:,:,3] = self.ixMmax[:,np.newaxis]+1
        self.ixR_r_max[:,:,0] = self.nghostcells-1
        self.ixR_r_max[:,:,1] = self.ixMmin[:,np.newaxis]-1+block_nxCo[:,np.newaxis]
        self.ixR_r_max[:,:,2] = self.ixMmax[:,np.newaxis]
        self.ixR_r_max[:,:,3] = self.ixGmax[:,np.newaxis]

        """Send prolonged (p) to finer blocks"""
        # the first is for 3 directions, the second is similar
        # the third adds two finer blocks neighborhood requiring the two prolongation
        self.ixS_p_min = np.zeros((3,3,4)).astype(int)
        self.ixS_p_max = np.zeros((3,3,4)).astype(int)
        self.ixR_p_min = np.zeros((3,3,4)).astype(int)
        self.ixR_p_max = np.zeros((3,3,4)).astype(int)

        # apply the interpolation order
        self.ixS_p_min[:,:,0] = self.ixMmin[:,np.newaxis]-(interpolation_order-1)
        self.ixS_p_min[:,:,1] = self.ixMmin[:,np.newaxis]-(interpolation_order-1)
        self.ixS_p_min[:,:,2] = self.ixMmin[:,np.newaxis]+block_nxCo[:,np.newaxis]-nghostcellsCo-(interpolation_order-1)
        self.ixS_p_min[:,:,3] = self.ixMmax[:,np.newaxis]+1-nghostcellsCo-(interpolation_order-1)
        self.ixS_p_max[:,:,0] = self.ixMmin[:,np.newaxis]-1+nghostcellsCo+(interpolation_order-1)
        self.ixS_p_max[:,:,1] = self.ixMmin[:,np.newaxis]-1+block_nxCo[:,np.newaxis]+nghostcellsCo+(interpolation_order-1)
        self.ixS_p_max[:,:,2] = self.ixMmax[:,np.newaxis]+(interpolation_order-1)
        self.ixS_p_max[:,:,3] = self.ixMmax[:,np.newaxis]+(interpolation_order-1)

        self.ixR_p_min[:,:,0] = ixCoMmin[:,np.newaxis]-nghostcellsCo-(interpolation_order-1)
        self.ixR_p_min[:,:,1] = ixCoMmin[:,np.newaxis]-(interpolation_order-1)
        self.ixR_p_min[:,:,2] = ixCoMmin[:,np.newaxis]-nghostcellsCo-(interpolation_order-1)
        self.ixR_p_min[:,:,3] = ixCoMmax[:,np.newaxis]+1-(interpolation_order-1)
        self.ixR_p_max[:,:,0] = self.nghostcells-1+(interpolation_order-1)
        self.ixR_p_max[:,:,1] = ixCoMmax[:,np.newaxis]+nghostcellsCo+(interpolation_order-1)
        self.ixR_p_max[:,:,2] = ixCoMmax[:,np.newaxis]+(interpolation_order-1)
        self.ixR_p_max[:,:,3] = ixCoMmax[:,np.newaxis]+nghostcellsCo+(interpolation_order-1)

        # TODO: no staggered situation here [or mf]:
        self.ixS_srl_min[:,0,1] = self.ixGmin
        self.ixS_srl_min[:,2,1] = self.ixMmin
        self.ixS_srl_min[:,3,1] = self.ixGmin
        self.ixS_srl_max[:,0,1] = self.ixMmax
        self.ixS_srl_max[:,2,1] = self.ixGmax
        self.ixS_srl_max[:,3,1] = self.ixGmax

        self.ixR_srl_min[:,0,1] = self.ixGmin
        self.ixR_srl_min[:,2,1] = self.ixMmin
        self.ixR_srl_min[:,3,1] = self.ixGmin
        self.ixR_srl_max[:,0,1] = self.ixMmax
        self.ixR_srl_max[:,2,1] = self.ixGmax
        self.ixR_srl_max[:,3,1] = self.ixGmax

        self.ixS_r_min[:,0,1] = self.ixGmin
        self.ixS_r_min[:,2,1] = ixCoMmin
        self.ixS_r_max[:,0,1] = ixCoMmax
        self.ixS_r_max[:,2,1] = ixCoGmax

        self.ixR_r_min[:,0,1] = self.ixGmin
        self.ixR_r_min[:,2,2] = self.ixMmin+block_nxCo
        self.ixR_r_max[:,0,1] = self.ixMmin-1+block_nxCo
        self.ixR_r_max[:,2,2] = self.ixGmax

        self.ixS_p_min[:,0,1] = self.ixGmin
        self.ixS_p_max[:,2,2] = self.ixGmax
        self.ixR_p_min[:,0,1] = self.ixGmin
        self.ixR_p_max[:,2,2] = ixCoGmax

    def _init_coordinates(self):

        self.rnode = np.zeros((9,self.nleafs))

        for igrid in range(self.nleafs):
            node = self.forest.sfc_to_node[igrid].node
            level = node.level
            self.rnode[0,igrid] = node.ig1*(self.xrange[1]-self.xrange[0])/self.nblock[0]/2**(level-1)+self.xrange[0]
            self.rnode[1,igrid] = node.ig2*(self.yrange[1]-self.yrange[0])/self.nblock[1]/2**(level-1)+self.yrange[0]
            self.rnode[2,igrid] = node.ig3*(self.zrange[1]-self.zrange[0])/self.nblock[2]/2**(level-1)+self.zrange[0]
            self.rnode[3,igrid] = (node.ig1+1)*(self.xrange[1]-self.xrange[0])/self.nblock[0]/2**(level-1)+self.xrange[0]
            self.rnode[4,igrid] = (node.ig2+1)*(self.yrange[1]-self.yrange[0])/self.nblock[1]/2**(level-1)+self.yrange[0]
            self.rnode[5,igrid] = (node.ig3+1)*(self.zrange[1]-self.zrange[0])/self.nblock[2]/2**(level-1)+self.zrange[0]
            self.rnode[6,igrid] = (self.xrange[1]-self.xrange[0])/self.domain_nx[0]/2**(level-1)
            self.rnode[7,igrid] = (self.yrange[1]-self.yrange[0])/self.domain_nx[1]/2**(level-1)
            self.rnode[8,igrid] = (self.zrange[1]-self.zrange[0])/self.domain_nx[2]/2**(level-1)
    
    def _init_field_idx(self):

        if "rho" in self.field_names:
            self.rho_ = list(self.field_names).index("rho")
        if "m1" in self.field_names:
            self.m1_ = list(self.field_names).index("m1")
        if "m2" in self.field_names:
            self.m2_ = list(self.field_names).index("m2")
        if "m3" in self.field_names:
            self.m3_ = list(self.field_names).index("m3")
        if "b1" in self.field_names:
            self.b1_ = list(self.field_names).index("b1")
        if "b2" in self.field_names:
            self.b2_ = list(self.field_names).index("b2")
        if "b3" in self.field_names:
            self.b3_ = list(self.field_names).index("b3")

    def getbc(self):

        # fill physical boundary ghost cells before internal ghost-cell values exchange
        for igrid in range(self.nleafs):
            self.fill_boundary_before_gc(igrid)

        # print("Fill Boundary Before GC")

        for igrid in range(self.nleafs):
            if np.any(self.forest.neighbor_type[:,:,:,igrid]==2):
                self.coarsen_grid(igrid, self.ixMmin, self.ixMmax, self.ixCoMmin, self.ixCoMmax)

            for kdir, jdir, idir in np.ndindex(3,3,3):
                if self.forest.neighbor_type[idir,jdir,kdir,igrid] == 2 and np.any(self.forest.neighbor_type[:,:,:,igrid]==1):
                    self.fill_coarse_boundary(igrid, idir, jdir, kdir)

        # print("Fill Coarse Boundary")

        for igrid in range(self.nleafs):
            self.idphyb[:,igrid] = self.identifyphysbound(igrid)
            for k,j,i in np.ndindex(3,3,3):
                # coarse neighbor
                if self.forest.neighbor_type[i,j,k,igrid] == 2:
                    self.bc_fill_restrict(igrid,i,j,k, self.idphyb[0,igrid], self.idphyb[1,igrid], self.idphyb[2,igrid])
                # sibling neighbor
                elif self.forest.neighbor_type[i,j,k,igrid] == 3:
                    self.bc_fill_srl(igrid,i,j,k, self.idphyb[0,igrid], self.idphyb[1,igrid], self.idphyb[2,igrid])

        # print("Fill Sibling and coarser gc")

        for igrid in range(self.nleafs):
            for k,j,i in np.ndindex(3,3,3):
                if (self.forest.neighbor_type[i,j,k,igrid] == 4):
                    self.bc_fill_prolong(igrid,i,j,k, self.idphyb[0,igrid], self.idphyb[1,igrid], self.idphyb[2,igrid])

        # print("Fill Prolong gc (in coarse data)")

        for igrid in range(self.nleafs):
            self.gc_prolong(igrid)
        
        # print("Fill Prolong gc (in fine data)")

        for igrid in range(self.nleafs):
            self.fill_boundary_after_gc(igrid)
        
        # print("Fill gc After GC")
    

    def fill_boundary_before_gc(self, igrid: int):

        for idim in range(3):

            km = np.array([1 if i != idim else 0 for i in range(3)])
            ixBmin = self.ixGmin + km * self.nghostcells
            ixBmax = self.ixGmax - km * self.nghostcells

            if ((idim > 0) and self.forest.neighbor_type[0,1,1,igrid]==1):
                ixBmin[0] = self.ixGmin[0]
            if ((idim > 0) and self.forest.neighbor_type[2,1,1,igrid]==1):
                ixBmax[0] = self.ixGmax[0]
            if ((idim > 1) and self.forest.neighbor_type[1,0,1,igrid]==1):
                ixBmin[1] = self.ixGmin[1]
            if ((idim > 1) and self.forest.neighbor_type[1,2,1,igrid]==1):
                ixBmax[1] = self.ixGmax[1]
            
            for iside in range(2):
                i1, i2, i3 = (np.eye(3)[idim] * (2*iside-1) + 1).astype(int)
                if (self.forest.neighbor_type[i1,i2,i3,igrid] != 1):
                    continue
                self.bc_phys(iside,idim,igrid,ixBmin,ixBmax)
    
    def fill_boundary_after_gc(self, igrid: int):

        for idim in range(3):
            kmin = np.array([1 if idim < i and self.forest.neighbor_type[*((-np.eye(3).astype(int))[idim]+1), igrid]==1 else 0 for i in range(3)]).astype(int)
            kmax = np.array([1 if idim < i and self.forest.neighbor_type[*(np.eye(3).astype(int)[idim]+1), igrid]==1 else 0 for i in range(3)]).astype(int)
            kmin[0] = 0
            kmax[0] = 0

            ixBmin = self.ixGmin + kmin * self.nghostcells
            ixBmax = self.ixGmax - kmax * self.nghostcells

            for iside in range(2):
                iis = (np.eye(3)[idim] * (2*iside-1)).astype(int)
                # no aperiod situation here
                if (self.forest.neighbor_type[*(iis+1), igrid] != 1):
                    continue
                self.bc_phys(iside, idim, igrid, ixBmin, ixBmax)


    def bc_phys(self, iside: int, idim: int, igrid: int, ixBmin: np.ndarray, ixBmax: np.ndarray, 
                coarsen: bool = False):

        if iside == 1:
            # maximum boundary
            ixOmin = ixBmin.copy()
            ixOmin[idim] = ixBmax[idim]+1-self.nghostcells
            ixOmax = ixBmax.copy()

            # Create index tuples for all dimensions
            idx = tuple(slice(ixOmin[i], ixOmax[i]+1) if i != idim 
                       else slice(ixOmin[i]-1, ixOmin[i]-1+1) 
                       for i in range(3))
            
        elif iside == 0:
            # minimum boundary
            ixOmin = ixBmin.copy()
            ixOmax = ixBmax.copy()
            ixOmax[idim] = ixBmin[idim]-1+self.nghostcells

            # cont situation
            idx = tuple(slice(ixOmin[i], ixOmax[i]+1) if i != idim 
                       else slice(ixOmax[i]+1, ixOmax[i]+2) 
                       for i in range(3))

        # Copy boundary values using a single assignment
        if not coarsen:
            self.data[igrid][tuple(slice(ixOmin[i], ixOmax[i]+1) for i in range(3))] = \
                self.data[igrid][idx]
        else:
            self.datac[igrid][tuple(slice(ixOmin[i], ixOmax[i]+1) for i in range(3))] = \
                self.datac[igrid][idx]
            

    def coarsen_grid(self, igrid:int, ixFimin:np.ndarray, ixFimax:np.ndarray,
                ixComin:np.ndarray, ixComax:np.ndarray):
        
            CoFiratio = 1/2**3

            for ixCo3 in range(ixComin[2], ixComax[2]+1):
                for ixCo2 in range(ixComin[1], ixComax[1]+1):
                    for ixCo1 in range(ixComin[0], ixComax[0]+1):
                        ixFi = (np.array([ixCo1, ixCo2, ixCo3])-ixComin) * 2 + ixFimin
                        self.datac[igrid][ixCo1, ixCo2, ixCo3, :] = np.sum(
                            self.data[igrid][ixFi[0]:ixFi[0]+2, ixFi[1]:ixFi[1]+2, ixFi[2]:ixFi[2]+2, :]
                            , axis=(0,1,2)) * CoFiratio
    

    def fill_coarse_boundary(self, igrid:int, idir:int, jdir:int, kdir:int):

        for idim in range(3):
            km = np.array([1 if i != idim else 0 for i in range(3)]).astype(int)
            ixBmin = self.ixCoGmin+km*self.nghostcells
            ixBmax = self.ixCoGmax-km*self.nghostcells

            if (idim > 0 and self.forest.neighbor_type[0,1,1, igrid] == 1):
                ixBmin[0] = self.ixCoGmin[0]
            if (idim > 0 and self.forest.neighbor_type[2,1,1, igrid] == 1):
                ixBmax[0] = self.ixCoGmax[0]
            if (idim > 1 and self.forest.neighbor_type[1,0,1, igrid] == 1):
                ixBmin[1] = self.ixCoGmin[1]
            if (idim > 1 and self.forest.neighbor_type[1,2,1, igrid] == 1):
                ixBmax[1] = self.ixCoGmax[1]
            
            for dim, i in enumerate([idir, jdir, kdir]):
                if i == 0:
                    ixBmin[dim] = self.ixCoGmin[dim]+self.nghostcells
                    ixBmax[dim] = self.ixCoGmin[dim]+2*self.nghostcells-1
                elif i == 2:
                    ixBmin[dim] = self.ixCoGmax[dim]-2*self.nghostcells+1
                    ixBmax[dim] = self.ixCoGmax[dim]-self.nghostcells

            idirs = [idir, jdir, kdir]
            
            for iside in range(2):
                ii = (np.eye(3)[idim] * (2*iside-1)).astype(int)
                # The coarse neighbor and the boundary should be perpendicular
                if any(abs(idirs[i]-1)==1 and abs(ii[i])==1 for i in range(3)):
                    continue
                if (self.forest.neighbor_type[ii[0]+1, ii[1]+1, ii[2]+1, igrid] != 1):
                    continue
                for jdim in range(3):
                    if (abs(ii[jdim] == 1 and abs([idir,jdir,kdir][jdim]) == 1)):
                        continue
                self.bc_phys(iside, idim, igrid, ixBmin, ixBmax, coarsen=True) 

    def bc_fill_srl(self, igrid, i1, i2, i3, iib1, iib2, iib3):

        ineighbor = self.forest.neighbor[i1,i2,i3,igrid]-1
        iis = np.array([i1,i2,i3]).astype(int)
        iibs = np.array([iib1,iib2,iib3]).astype(int)
        n_is = -1 * (iis-1) + 1
        ixSmin = np.zeros(3).astype(int)
        ixSmax = np.zeros(3).astype(int)
        ixRmin = np.zeros(3).astype(int)
        ixRmax = np.zeros(3).astype(int)
        # TODO: only ipole == 0 situation disposed here
        for i in range(3):
            ixSmin[i] = self.ixS_srl_min[i,iibs[i]+1,iis[i]]
            ixSmax[i] = self.ixS_srl_max[i,iibs[i]+1,iis[i]]
            ixRmin[i] = self.ixR_srl_min[i,iibs[i]+1,n_is[i]]
            ixRmax[i] = self.ixR_srl_max[i,iibs[i]+1,n_is[i]]

        self.data[ineighbor][ixRmin[0]:ixRmax[0]+1, ixRmin[1]:ixRmax[1]+1, ixRmin[2]:ixRmax[2]+1, :] = \
            self.data[igrid][ixSmin[0]:ixSmax[0]+1, ixSmin[1]:ixSmax[1]+1, ixSmin[2]:ixSmax[2]+1, :]

    # fill coarser neighbor's ghost cells
    def bc_fill_restrict(self, igrid, i1, i2, i3, iib1, iib2, iib3):

        nodeptr = self.forest.sfc_to_node[igrid]
        assert isinstance(nodeptr, OctreeNodePointer), "Node is not an OctreeNodePointer"
        node = nodeptr.node
        assert isinstance(node, OctreeNode), "Node is not an OctreeNode"

        igs = np.array([node.ig1, node.ig2, node.ig3]).astype(int)
        iis = np.array([i1,i2,i3]).astype(int)
        iibs = np.array([iib1,iib2,iib3]).astype(int)

        ics = 1+(igs % 2)
        if (not (i1==1 or i1==2*ics[0]-2) or not (i2==1 or i2==2*ics[1]-2) or not (i3==1 or i3==2*ics[2]-2)):
            return

        ineighbor = self.forest.neighbor[i1,i2,i3,igrid]-1

        # TODO: only ipole == 0 situation disposed here

        # part at the neighbor's side
        n_incs = -2*(iis-1)+ics
        ixSmin = np.zeros(3).astype(int)
        ixSmax = np.zeros(3).astype(int)
        ixRmin = np.zeros(3).astype(int)
        ixRmax = np.zeros(3).astype(int)

        for i in range(3):
            ixSmin[i] = self.ixS_r_min[i,iibs[i]+1,iis[i]]
            ixSmax[i] = self.ixS_r_max[i,iibs[i]+1,iis[i]]
            ixRmin[i] = self.ixR_r_min[i,iibs[i]+1,n_incs[i]]
            ixRmax[i] = self.ixR_r_max[i,iibs[i]+1,n_incs[i]]

        self.data[ineighbor][ixRmin[0]:ixRmax[0]+1, ixRmin[1]:ixRmax[1]+1, ixRmin[2]:ixRmax[2]+1, :] = \
            self.datac[igrid][ixSmin[0]:ixSmax[0]+1, ixSmin[1]:ixSmax[1]+1, ixSmin[2]:ixSmax[2]+1, :]

    def bc_fill_prolong(self, igrid, i1, i2, i3, iib1, iib2, iib3):

        iibs = np.array([iib1,iib2,iib3]).astype(int)

        for ic3 in range(1+int((2-i3)/2), 2-int(i3/2)+1):
            inc3 = 2 * (i3-1) + ic3
            for ic2 in range(1+int((2-i2)/2), 2-int(i2/2)+1):
                inc2 = 2 * (i2-1) + ic2
                for ic1 in range(1+int((2-i1)/2), 2-int(i1/2)+1):
                    inc1 = 2 * (i1-1) + ic1
                    incs = np.array([inc1,inc2,inc3]).astype(int)

                    ineighbor = self.forest.neighbor_child[inc1,inc2,inc3,igrid]-1
                    n_i1=-1 * (i1-1); n_i2=-1 * (i2-1); n_i3=-1 * (i3-1)
                    n_inc1=ic1+n_i1; n_inc2=ic2+n_i2; n_inc3=ic3+n_i3

                    n_incs = np.array([n_inc1,n_inc2,n_inc3]).astype(int)

                    ixSmin = np.zeros(3).astype(int)
                    ixSmax = np.zeros(3).astype(int)
                    ixRmin = np.zeros(3).astype(int)
                    ixRmax = np.zeros(3).astype(int)

                    for i in range(3):
                        ixSmin[i] = self.ixS_p_min[i,iibs[i]+1,incs[i]]
                        ixSmax[i] = self.ixS_p_max[i,iibs[i]+1,incs[i]]
                        ixRmin[i] = self.ixR_p_min[i,iibs[i]+1,n_incs[i]]
                        ixRmax[i] = self.ixR_p_max[i,iibs[i]+1,n_incs[i]]
                    

                    self.datac[ineighbor][ixRmin[0]:ixRmax[0]+1, ixRmin[1]:ixRmax[1]+1, ixRmin[2]:ixRmax[2]+1, :] = \
                        self.data[igrid][ixSmin[0]:ixSmax[0]+1, ixSmin[1]:ixSmax[1]+1, ixSmin[2]:ixSmax[2]+1, :]

    def gc_prolong(self, igrid: int):

        iibs = self.idphyb[:,igrid]
        NeedProlong = np.zeros((3,3,3), dtype=bool)

        for i3,i2,i1 in np.ndindex(3,3,3):
            if (self.forest.neighbor_type[i1,i2,i3,igrid] == 2):
                self.bc_prolong(igrid, i1, i2, i3, iibs[0], iibs[1], iibs[2])
                NeedProlong[i1,i2,i3] = True

    def identifyphysbound(self, igrid):

        iibs = np.zeros(3).astype(int)

        for idim in range(3):
            idx = np.ones(3).astype(int)
            idx[idim] = 0
            if self.forest.neighbor_type[*idx, igrid] == 1 and \
                self.forest.neighbor_type[*(idx+2*np.eye(3)[idim].astype(int)), igrid] == 1:
                iibs[idim] = 2
            elif self.forest.neighbor_type[*idx, igrid] == 1:
                iibs[idim] = -1
            elif self.forest.neighbor_type[*(idx+2*np.eye(3)[idim].astype(int)), igrid] == 1:
                iibs[idim] = 1
            else:
                iibs[idim] = 0
        
        return iibs

    def bc_prolong(self, igrid, i1, i2, i3, iib1, iib2, iib3):

        iis = np.array([i1,i2,i3]).astype(int)
        iibs = np.array([iib1,iib2,iib3]).astype(int)
        ixFimin = np.zeros(3).astype(int)
        ixFimax = np.zeros(3).astype(int)

        for i in range(3):
            ixFimin[i] = self.ixR_srl_min[i,iibs[i]+1,iis[i]]
            ixFimax[i] = self.ixR_srl_max[i,iibs[i]+1,iis[i]]

        dxFi = self.rnode[6:9,igrid]
        dxCo = 2*dxFi
        invdxCo = 1/dxCo

        xFimin = self.rnode[:3,igrid]-self.nghostcells*dxFi
        xComin = self.rnode[:3,igrid]-self.nghostcells*dxCo

        # TODO: transform physical variables to conserved variables (do at the very first)

        self.interpolation_linear(igrid, ixFimin, ixFimax, dxFi, xFimin, dxCo, invdxCo, xComin)

    def interpolation_linear(self, igrid, ixFimin, ixFimax, dxFi, xFimin, dxCo, invdxCo, xComin):
        # Calculate dimensions once
        shape = tuple(ixFimax - ixFimin + 1)
        field_count = len(self.field_names)
        
        # Create meshgrid for fine indices - use float32 to save memory
        ixFi = np.stack(np.meshgrid(
            *[np.arange(ixFimin[i], ixFimax[i] + 1) for i in range(3)],
            indexing='ij'
        ))
        
        # Calculate fine and coarse coordinates in one step
        xFi = xFimin[:, None, None, None] + (ixFi + 0.5) * dxFi[:, None, None, None]
        ixCo = ((xFi - xComin[:, None, None, None]) * invdxCo[:, None, None, None]).astype(int)
        xCo = xComin[:, None, None, None] + (ixCo + 0.5) * dxCo[:, None, None, None]
        
        # Calculate eta once
        eta = (xFi - xCo) * invdxCo[:, None, None, None]
        
        # Pre-allocate result array
        result = np.zeros((*shape, field_count))
        
        # Calculate slopes and update result in one pass
        for dim in range(3):
            # Get neighbor indices efficiently
            ixCo_left = ixCo.copy()
            ixCo_left[dim] -= 1
            ixCo_right = ixCo.copy()
            ixCo_right[dim] += 1
            
            # Gather values using advanced indexing
            center_vals = self.datac[igrid][tuple(ixCo)]
            left_vals = self.datac[igrid][tuple(ixCo_left)]
            right_vals = self.datac[igrid][tuple(ixCo_right)]
            
            # Calculate slopes vectorized
            slopeL = center_vals - left_vals
            slopeR = right_vals - center_vals
            slopeC = (slopeR + slopeL) * 0.5
            
            # Calculate limited slopes efficiently
            signC = np.sign(slopeC)
            slope = np.minimum(
                np.minimum(
                    np.abs(slopeC),
                    np.where(signC > 0, slopeL, -slopeL)
                ),
                np.where(signC > 0, slopeR, -slopeR)
            )
            slope = signC * np.maximum(0, slope)
            
            # Add contribution to result
            result += slope * eta[dim, ..., None]
        
        # Add base values
        result += self.datac[igrid][tuple(ixCo)]
        
        # Assign results back to data array efficiently
        self.data[igrid][tuple(ixFi.astype(int))] = result
        # assert np.all((ixFi/2+1).astype(int) == ixCo), f"Invalid index: {igrid}, {ixFi}, {ixCo}"

    def interpolation_linear1(self, igrid, ixFimin, ixFimax, dxFi, xFimin, dxCo, invdxCo, xComin):

        slope = np.ones((len(self.field_names), 3))

        for ixFi3, ixFi2, ixFi1 in np.ndindex(*(ixFimax-ixFimin+1)[::-1]):

            # cell-centered coordinates of fine grid (equidistant grid), plus 0.5 for 0-based index
            ixFi = np.array([ixFi1, ixFi2, ixFi3]) + ixFimin
            xFi = xFimin + (ixFi+0.5)*dxFi

            # indices of coarse cell which contains the fine cell with left corner xFi
            ixCo = ((xFi-xComin)*invdxCo).astype(int)

            # cell-centered coordinates of coarse grid (equidistant grid)
            xCo = xComin+(ixCo+0.5)*dxCo

            # TODO: only slab_unifom coordinate disposed here, with stretch coordinate, need to be modified

            # normalized distance from the coarse cell center to the fine cell center: 1/4 or -1/4
            eta = (xFi-xCo)*invdxCo

            for idim in range(3):
                hxCo = ixCo-(np.eye(3).astype(int))[idim]
                jxCo = ixCo+(np.eye(3).astype(int))[idim]

                for iw in range(len(self.field_names)):

                    slopeL = self.datac[igrid][*ixCo,iw]-self.datac[igrid][*hxCo,iw]
                    slopeR = self.datac[igrid][*jxCo,iw]-self.datac[igrid][*ixCo,iw]
                    slopeC = (slopeR+slopeL)/2

                    signR = np.sign(slopeR)
                    signC = np.sign(slopeC)

                    slope[iw,idim] = signC*max(0,min(abs(slopeC),signC*slopeL,signC*slopeR))

            # assert np.all(self.datac[igrid][*ixCo,:]), f"coarsen grid not filled: {igrid}, {ixCo}"
            self.data[igrid][*ixFi,:] = self.datac[igrid][*ixCo,:] + np.sum(slope * eta[np.newaxis,:], axis=1)
            assert (ixCo == np.floor(ixFi/2)+1).all(), f"Invalid index: {ixFi}, {ixCo}"

        return self.datac[igrid][*ixCo,:] + np.sum(slope * eta[np.newaxis,:], axis=1)

    def export_slab_uniform(self, xmin_uniform, xmax_uniform, nx, ny, nz):
        xmin_uniform = np.asarray(xmin_uniform)
        xmax_uniform = np.asarray(xmax_uniform)
        dx_uniform = (xmax_uniform-xmin_uniform)/np.array([nx,ny,nz])
        
        slab_uniform = np.zeros((nx,ny,nz,len(self.field_names)))

        for igrid in range(self.nleafs):
            xmin_block = self.rnode[0:3,igrid]
            xmax_block = self.rnode[3:6,igrid]
            dx_block = self.rnode[6:9,igrid]

            igmin_uniform = (xmin_block-xmin_uniform)/dx_uniform
            igmax_uniform = (xmax_block-xmin_uniform)/dx_uniform

            # Skip if block is completely outside uniform grid
            if np.any(igmin_uniform > np.array([nx,ny,nz])) or np.any(igmax_uniform < 0):
                continue

            igzmin_uniform = np.maximum(igmin_uniform.astype(int), 0)
            igzmax_uniform = np.minimum(igmax_uniform.astype(int), np.array([nx,ny,nz]))
            
            # Create meshgrid for all points in this block's region
            x_idx = np.arange(igzmin_uniform[0], igzmax_uniform[0])
            y_idx = np.arange(igzmin_uniform[1], igzmax_uniform[1])
            z_idx = np.arange(igzmin_uniform[2], igzmax_uniform[2])
            xx, yy, zz = np.meshgrid(x_idx, y_idx, z_idx, indexing='ij')
            
            # Calculate uniform grid positions
            x_uniform = xmin_uniform[0] + (xx + 0.5) * dx_uniform[0]
            y_uniform = xmin_uniform[1] + (yy + 0.5) * dx_uniform[1]
            z_uniform = xmin_uniform[2] + (zz + 0.5) * dx_uniform[2]
            
            # Calculate indices within block
            igx = (x_uniform - xmin_block[0]) / dx_block[0]
            igy = (y_uniform - xmin_block[1]) / dx_block[1]
            igz = (z_uniform - xmin_block[2]) / dx_block[2]
            
            # Get integer indices and weights
            i0x = igx.astype(int) + self.nghostcells - 1
            i0y = igy.astype(int) + self.nghostcells - 1
            i0z = igz.astype(int) + self.nghostcells - 1
            i1x = i0x + 1
            i1y = i0y + 1
            i1z = i0z + 1
            
            # Calculate interpolation weights
            wx = igx - (i0x - self.nghostcells + 0.5)
            wy = igy - (i0y - self.nghostcells + 0.5)
            wz = igz - (i0z - self.nghostcells + 0.5)
            
            # Reshape weights for broadcasting
            wx = wx[..., None]
            wy = wy[..., None]
            wz = wz[..., None]
            
            # Get all corner values at once using advanced indexing
            c000 = self.data[igrid, i0x, i0y, i0z]
            c001 = self.data[igrid, i0x, i0y, i1z]
            c010 = self.data[igrid, i0x, i1y, i0z]
            c011 = self.data[igrid, i0x, i1y, i1z]
            c100 = self.data[igrid, i1x, i0y, i0z]
            c101 = self.data[igrid, i1x, i0y, i1z]
            c110 = self.data[igrid, i1x, i1y, i0z]
            c111 = self.data[igrid, i1x, i1y, i1z]
            
            # Perform trilinear interpolation in one go
            c00 = c000 * (1-wz) + c001 * wz
            c01 = c010 * (1-wz) + c011 * wz
            c10 = c100 * (1-wz) + c101 * wz
            c11 = c110 * (1-wz) + c111 * wz
            
            c0 = c00 * (1-wy) + c01 * wy
            c1 = c10 * (1-wy) + c11 * wy
            
            result = c0 * (1-wx) + c1 * wx
            
            # Store results
            slab_uniform[x_idx[:,None,None], y_idx[None,:,None], z_idx[None,None,:]] = result

        return slab_uniform
    
    def load_current(self):
        self.current = self.calculate_current()
    
    def calculate_current(self):
        if not self.b1_ or not self.b2_ or not self.b3_:
            print("No magnetic field data found")
            return
        
        # Extract magnetic field components
        b1 = self.data[..., self.b1_]
        b2 = self.data[..., self.b2_]
        b3 = self.data[..., self.b3_]

        full_shape = b1.shape

        # Reshape grid spacings for broadcasting
        # From shape (nleafs,) to (nleafs, 1, 1, 1)
        dx = self.rnode[6,:,None,None,None]
        dy = self.rnode[7,:,None,None,None]
        dz = self.rnode[8,:,None,None,None]

        # Calculate gradients using central differences
        # For each block, compute (f[i+1] - f[i-1])/(2*dx)
        db1_dy = (b1[:,:,2:,:] - b1[:,:,:-2,:]) / (2*dy)
        db1_dz = (b1[:,:,:,2:] - b1[:,:,:,:-2]) / (2*dz)
        
        db2_dx = (b2[:,2:,:,:] - b2[:,:-2,:,:]) / (2*dx)
        db2_dz = (b2[:,:,:,2:] - b2[:,:,:,:-2]) / (2*dz)
        
        db3_dx = (b3[:,2:,:,:] - b3[:,:-2,:,:]) / (2*dx)
        db3_dy = (b3[:,:,2:,:] - b3[:,:,:-2,:]) / (2*dy)

        # fill the full-sized arrays with zeros
        j1 = np.zeros(full_shape)
        j2 = np.zeros(full_shape)
        j3 = np.zeros(full_shape)

        # Compute current components
        # Note: Result will be 2 cells smaller in each direction due to central differences
        j1[:, 1:-1, 1:-1, 1:-1] = db3_dy[:, 1:-1, :, 1:-1] - db2_dz[:, 1:-1, 1:-1, :]
        j2[:, 1:-1, 1:-1, 1:-1] = db1_dz[:, 1:-1, 1:-1, :] - db3_dx[:, :, 1:-1, 1:-1]
        j3[:, 1:-1, 1:-1, 1:-1] = db2_dx[:, :, 1:-1, 1:-1] - db1_dy[:, 1:-1, :, 1:-1]

        return j1, j2, j3

    def export_uniform_current(self, xmin_uniform, xmax_uniform, nx, ny, nz):
        if not hasattr(self, 'current') or self.current is None:
            print("No current data found")
            return None
        
        j1, j2, j3 = self.current
        current = np.stack([j1, j2, j3], axis=-1)
        
        xmin_uniform = np.asarray(xmin_uniform)
        xmax_uniform = np.asarray(xmax_uniform)
        dx_uniform = (xmax_uniform-xmin_uniform)/np.array([nx,ny,nz])
        
        slab_uniform = np.zeros((nx,ny,nz,3))

        for igrid in range(self.nleafs):
            xmin_block = self.rnode[0:3,igrid]
            xmax_block = self.rnode[3:6,igrid]
            dx_block = self.rnode[6:9,igrid]

            igmin_uniform = (xmin_block-xmin_uniform)/dx_uniform
            igmax_uniform = (xmax_block-xmin_uniform)/dx_uniform

            # Skip if block is completely outside uniform grid
            if np.any(igmin_uniform > np.array([nx,ny,nz])) or np.any(igmax_uniform < 0):
                continue

            igzmin_uniform = np.maximum(igmin_uniform.astype(int), 0)
            igzmax_uniform = np.minimum(igmax_uniform.astype(int), np.array([nx,ny,nz]))
            
            # Create meshgrid for all points in this block's region
            x_idx = np.arange(igzmin_uniform[0], igzmax_uniform[0])
            y_idx = np.arange(igzmin_uniform[1], igzmax_uniform[1])
            z_idx = np.arange(igzmin_uniform[2], igzmax_uniform[2])
            xx, yy, zz = np.meshgrid(x_idx, y_idx, z_idx, indexing='ij')
            
            # Calculate uniform grid positions
            x_uniform = xmin_uniform[0] + (xx + 0.5) * dx_uniform[0]
            y_uniform = xmin_uniform[1] + (yy + 0.5) * dx_uniform[1]
            z_uniform = xmin_uniform[2] + (zz + 0.5) * dx_uniform[2]
            
            # Calculate indices within block
            igx = (x_uniform - xmin_block[0]) / dx_block[0]
            igy = (y_uniform - xmin_block[1]) / dx_block[1]
            igz = (z_uniform - xmin_block[2]) / dx_block[2]
            
            # Get integer indices and weights
            i0x = igx.astype(int) + self.nghostcells - 1
            i0y = igy.astype(int) + self.nghostcells - 1
            i0z = igz.astype(int) + self.nghostcells - 1
            i1x = i0x + 1
            i1y = i0y + 1
            i1z = i0z + 1
            
            # Calculate interpolation weights
            wx = igx - (i0x - self.nghostcells + 0.5)
            wy = igy - (i0y - self.nghostcells + 0.5)
            wz = igz - (i0z - self.nghostcells + 0.5)
            
            # Reshape weights for broadcasting
            wx = wx[..., None]
            wy = wy[..., None]
            wz = wz[..., None]
            
            # Get all corner values at once using advanced indexing
            c000 = current[igrid, i0x, i0y, i0z]
            c001 = current[igrid, i0x, i0y, i1z]
            c010 = current[igrid, i0x, i1y, i0z]
            c011 = current[igrid, i0x, i1y, i1z]
            c100 = current[igrid, i1x, i0y, i0z]
            c101 = current[igrid, i1x, i0y, i1z]
            c110 = current[igrid, i1x, i1y, i0z]
            c111 = current[igrid, i1x, i1y, i1z]
            
            # Perform trilinear interpolation in one go
            c00 = c000 * (1-wz) + c001 * wz
            c01 = c010 * (1-wz) + c011 * wz
            c10 = c100 * (1-wz) + c101 * wz
            c11 = c110 * (1-wz) + c111 * wz
            
            c0 = c00 * (1-wy) + c01 * wy
            c1 = c10 * (1-wy) + c11 * wy
            
            result = c0 * (1-wx) + c1 * wx
            
            # Store results
            slab_uniform[x_idx[:,None,None], y_idx[None,:,None], z_idx[None,None,:]] = result

        return slab_uniform

def amrmesh_from_uniform(nw_arrays:np.ndarray, w_names, xmin, xmax, block_nx):

    """
    Create an AMR mesh from uniform data

    input:
    nw_arrays: 4D numpy arrays, each 3D array is a uniform grid of the same field, shape (nx, ny, nz, nw)
    w_names: list of strings, the names of the fields, shape (nw,)
    xmin: list of floats, the minimum coordinates of the uniform grid, shape (3,)
    xmax: list of floats, the maximum coordinates of the uniform grid, shape (3,)
    block_nx: list of ints, the number of cells in each direction for the block, shape (3,)
    **kwargs: keyword arguments, passed to the AMRForest constructor

    output:
    AMRMesh object
    """

    assert isinstance(nw_arrays, np.ndarray), "nw_arrays must be a numpy array"
    assert len(nw_arrays.shape) == 4, "nw_arrays must be a 4D array"
    domain_nx = np.array(nw_arrays.shape[:3])
    block_nx = np.array(block_nx)

    assert nw_arrays.shape[3] == len(w_names), "nw_arrays and w_names must have the same length"

    assert len(xmin) == 3, "xmin must be a 3-element array"
    assert len(xmax) == 3, "xmax must be a 3-element array"
    assert len(block_nx) == 3, "block_nx must be a 3-element array"

    assert np.all(np.array(xmin) < np.array(xmax)), "xmin must be less than xmax"

    nglev1 = domain_nx // block_nx
    assert np.all(nglev1 * block_nx == domain_nx), "domain_nx must be divisible by block_nx"
    nglev1 = nglev1.astype(int)

    nleafs = np.prod(nglev1)
    print(nleafs)

    # create the forest for all level1 1 blocks (all leaf nodes)
    forest = np.ones(nleafs).astype(np.bool_)
    uamr_forest = AMRForest(ng1=nglev1[0], ng2=nglev1[1], ng3=nglev1[2], nleafs=nleafs)
    uamr_forest.read_forest(forest)
    uamr_forest.build_connectivity() # not required maybe for the created uniform mesh

    # create the uniform amr mesh
    xrange = (xmin[0], xmax[0])
    yrange = (xmin[1], xmax[1])
    zrange = (xmin[2], xmax[2])
    umesh = AMRMesh(xrange, yrange, zrange, w_names, block_nx, domain_nx, uamr_forest, nghostcells=2)
    umesh.udata = nw_arrays

    iglevel1_sfc, sfc_iglevel1 = level1_Morton_order(*nglev1)

    for igrid in range(nleafs):
        igs = sfc_iglevel1[igrid]
        x0, y0, z0 = igs * block_nx
        x1, y1, z1 = (igs+1) * block_nx
        umesh.data[igrid][umesh.ixMmin[0]:umesh.ixMmax[0]+1, 
                          umesh.ixMmin[1]:umesh.ixMmax[1]+1, 
                          umesh.ixMmin[2]:umesh.ixMmax[2]+1] = \
            umesh.udata[x0:x1, y0:y1, z0:z1]

    return umesh
