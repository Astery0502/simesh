# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

# class for forest

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t

import numpy as np
cimport numpy as np

from ..tree cimport octptr, OctreeNode
from .morton cimport fill_morton_mapping3D

cdef uint32_t neighbor_unknown = 0
cdef uint32_t neighbor_boundary = 1
cdef uint32_t neighbor_coarse = 2
cdef uint32_t neighbor_sibling = 3
cdef uint32_t neighbor_fine = 4

cdef class AMRForest:

    def __cinit__(self, uint32_t ndim, uint32_t ng1, uint32_t ng2, uint32_t ng3, bint[:] is_leaf):

        cdef uint32_t i
        cdef uint32_t num_leafs = 0
        cdef uint32_t nblocks = ng1*ng2*ng3

        # temporary set to False
        for i in range(ndim):
            self.periodB[i] = False
            self.poleB[i] = False

        self.ng[0], self.ng[1], self.ng[2] = ng1, ng2, ng3

        # ndim requires to be 2 or 3, when ndim is 2, ng3 must be 1
        self.ndim = ndim

        # count the number of leaf nodes
        for i in range(is_leaf.shape[0]):
            if is_leaf[i]:
                num_leafs += 1

        # to be updated in read_forest
        self.nleafs = 0
        self.nparents = 0
        self.max_level = 0

        # allocate memory for the arrays and create memoryviews
        self.forest = <octptr*>malloc(nblocks* sizeof(octptr))
        for i in range(nblocks):
            self.forest[i].node = NULL

        self.sfc2node = <octptr*>malloc(num_leafs* sizeof(octptr))
        for i in range(num_leafs):
            self.sfc2node[i].node = NULL

        idx1_data = <uint32_t*>malloc(nblocks* sizeof(uint32_t))
        for i in range(nblocks):
            idx1_data[i] = 0
        self.idx1 = <uint32_t[:nblocks]>idx1_data

        # to do with ndim == 2 case
        ig2morton_data = <uint32_t*>malloc(nblocks* sizeof(uint32_t))
        for i in range(nblocks):
            ig2morton_data[i] = 0
        self.ig2morton = <uint32_t[:ng1,:ng2,:ng3]>ig2morton_data

        morton2ig_data = <uint32_t*>malloc(nblocks*ndim* sizeof(uint32_t))
        for i in range(nblocks*ndim):
            morton2ig_data[i] = 0
        self.morton2ig = <uint32_t[:nblocks,:ndim]>morton2ig_data

        is_leaf_data = <bint*>malloc(is_leaf.shape[0]* sizeof(bint))
        for i in range(is_leaf.shape[0]):
            is_leaf_data[i] = is_leaf[i]
        self.is_leaf = <bint[:is_leaf.shape[0]]>is_leaf_data

        neighbor_children_data = <uint32_t*>malloc(num_leafs*4**ndim* sizeof(uint32_t))
        for i in range(num_leafs*4**ndim):
            neighbor_children_data[i] = 0
        self.neighbor_children = <uint32_t[:num_leafs, :4**ndim]>neighbor_children_data

        neighbor_index_data = <uint32_t*>malloc(num_leafs*3**ndim* sizeof(uint32_t))
        for i in range(num_leafs*3**ndim):
            neighbor_index_data[i] = 0
        self.neighbor_index = <uint32_t[:num_leafs, :3**ndim]>neighbor_index_data

        neighbor_type_data = <uint32_t*>malloc(num_leafs*3**ndim* sizeof(uint32_t))
        for i in range(num_leafs*3**ndim):
            neighbor_type_data[i] = 0
        self.neighbor_type = <uint32_t[:num_leafs, :3**ndim]>neighbor_type_data

        # read the forest boolean array
        self.read_forest(is_leaf)
        self.build_connectivity()
    
    cdef inline uint32_t findex(self, uint32_t* ig):
        if self.ndim == 2:
            return ig[0]*self.ng[1] + ig[1]
        # 3D case
        return ig[0]*self.ng[1]*self.ng[2] + ig[1]*self.ng[2] + ig[2]

    cdef inline uint32_t nindex(self, uint32_t* n):
        if self.ndim == 2:
            return n[0]*3 + n[1]
        # 3D case
        return n[0]*3*3+ n[1]*3+ n[2]

    cdef inline uint32_t ncindex(self, uint32_t* nc):
        if self.ndim == 2:
            return nc[0]*4 + nc[1]
        # 3D case
        return nc[0]*4*4 + nc[1]*4 + nc[2]

    cdef void read_forest(self, bint[:] is_leaf):

        cdef uint32_t inode = 0
        cdef uint32_t ileaf = 0
        cdef uint32_t level = 1
        cdef uint32_t* ig
        cdef uint32_t i

        cdef OctreeNode* node
        
        # Morton mapping for level 1 blocks - using numpy arrays
        # We need to create memory views of our numpy arrays for fill_morton_mapping3D
        cdef uint32_t[:,:,:] ig2morton_view = self.ig2morton
        cdef uint32_t[:,:] morton2ig_view = self.morton2ig
        
        # Call with memory views
        fill_morton_mapping3D(ig2morton_view, morton2ig_view, self.ng[0], self.ng[1], self.ng[2])

        # Iterate over all level 1 blocks with all leaf blocks inside them
        for i in range(self.ng[0]*self.ng[1]*self.ng[2]):
            # Log the index of each first leaf in the level 1 block
            self.idx1[i] = inode

            # Get a pointer to the data using the memory view of morton2ig
            ig = &morton2ig_view[i,0]

            # Allocate memory for the node and set the tree in the forest
            node = <OctreeNode*>malloc(sizeof(OctreeNode))
            node.parent.node = NULL  # level 1 block has no parent

            self.forest[self.findex(ig)].node = node
            self.read_node(self.forest[self.findex(ig)], ig, level, &inode, &ileaf)

    cdef void read_node(self, octptr tree, uint32_t* ig, uint32_t level, 
                        uint32_t* inode_ptr, uint32_t* ileaf_ptr):
        cdef uint32_t child_ig[3]
        cdef uint32_t i, j, k
        cdef OctreeNode* child_node
        
        # Increment the node counter
        inode_ptr[0] += 1

        for i in range(3):
            tree.node.ig[i] = ig[i]

        tree.node.isleaf = self.is_leaf[inode_ptr[0] - 1]
        tree.node.level = level

        # Clean the auto allocated unexpected value for pointers
        # to do with ndim == 2 case
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    tree.node.children[k][j][i].node = NULL

        for j in range(3):
            for i in range(2):
                tree.node.neighbors[j][i].node = NULL

        self.asign_tree_neighbor(tree)

        if tree.node.isleaf:
            # Save the leaf node in the sfc2node array
            self.sfc2node[ileaf_ptr[0]].node = tree.node

            # Increment the leaf index
            self.nleafs += 1
            ileaf_ptr[0] += 1
            tree.node.ileaf = ileaf_ptr[0]  # start from 1

            # Update the max level
            if level > self.max_level:
                self.max_level = level
        else:
            self.nparents += 1
            tree.node.ileaf = 0  # not a leaf

            # to do with ndim == 2 case
            # contiguous memory required to be i, j, k in c order
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        # Child ig value: double the parent (2x) then +0/1
                        child_ig[0] = 2*ig[0]+i
                        child_ig[1] = 2*ig[1]+j
                        child_ig[2] = 2*ig[2]+k
                        child_node = <OctreeNode*>malloc(sizeof(OctreeNode))

                        # to modify in 2D case
                        tree.node.children[k][j][i].node = child_node
                        child_node.parent = tree
                        self.read_node(tree.node.children[k][j][i], child_ig, level+1, 
                                       inode_ptr, ileaf_ptr)

    cpdef bint[:] write_forest(self):

        cdef uint32_t i
        cdef uint32_t* ig
        cdef bint[:] forest
        cdef uint32_t inode = 0

        forest = np.zeros((self.nparents+self.nleafs), dtype=np.int32)

        for i in range(self.ng[0]*self.ng[1]*self.ng[2]):

            # to do with ndim == 2 case
            ig = &self.morton2ig[i,0]
            self.write_node(self.forest[self.findex(ig)], forest, &inode)

        return forest

    cdef void write_node(self, octptr tree, bint[:] forest, uint32_t* inode_ptr):
        cdef uint32_t i, j, k

        forest[inode_ptr[0]] = tree.node.isleaf
        inode_ptr[0] += 1
        if not tree.node.isleaf:
            # to do with ndim == 2 case
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        self.write_node(tree.node.children[k][j][i], forest, inode_ptr)
    
    cdef void asign_tree_neighbor(self, octptr tree):

        # to do with ndim == 2 case

        cdef uint32_t neighbor_type
        cdef int iside, idim
        cdef int kr[3]
        cdef bint pole[3]
        cdef octptr neighbor

        neighbor.node = NULL

        # initialize the kronecker 
        for idim in range(3):
            kr[idim] = 0

        for idim in range(3):
            for iside in range(2):
                kr[idim] = 2*iside-1
                # find the neighbor at one side
                neighbor_type = self.find_neighbor(&neighbor, tree, kr, pole)
                # reset the kr
                kr[idim] = 0
                # only assign the neighbor when returning neighbor is at the same level as the tree
                if (neighbor_type == neighbor_fine or neighbor_type == neighbor_sibling):
                    tree.node.neighbors[idim][iside].node = neighbor.node
                    if neighbor.node is not NULL:
                        if pole[idim]:
                            neighbor.node.neighbors[idim][iside].node = tree.node
                        else:
                            neighbor.node.neighbors[idim][1-iside].node = tree.node
                else:
                    tree.node.neighbors[idim][iside].node = NULL

    cdef void find_root_neighbor(self, octptr* neighbor, octptr tree, int* ii):

        cdef uint32_t idim
        cdef uint32_t jg[3]

        for idim in range(self.ndim):
            jg[idim] = tree.node.ig[idim] + ii[idim]
            if self.periodB[idim]:
                jg[idim] = (jg[idim] % self.ng[idim] + self.ng[idim]) % self.ng[idim] # (x%n + n) % n to avoid negative

        # skip the spherical and cylindrical cases

        for idim in range(self.ndim):
            if (jg[idim] < 0 or jg[idim] >= self.ng[idim]):
                neighbor.node = NULL
                return 
        
        neighbor.node = self.forest[self.findex(jg)].node
        return

    cdef uint32_t find_neighbor(self, octptr* neighbor, octptr tree, int* ii, bint* pole):
        
        cdef uint32_t neighbor_type
        cdef uint32_t idim, level

        cdef uint32_t igc[3] # the ig position as a child of its parent
        cdef uint32_t inp[3] # to judge whether the neighbor is at another parent block (+- 1 beyond the parent)
        cdef uint32_t inc[3] # to index the child of the neighbor which is not a leaf

        for idim in range(self.ndim):
            pole[idim] = False

        level = tree.node.level 

        if (level == 1):

            self.find_root_neighbor(neighbor, tree, ii)
            if neighbor.node is NULL:
                return neighbor_boundary
            
            # to do with the spherical and cylindrical cases for poleB

            if neighbor.node.isleaf:
                return neighbor_sibling

            return neighbor_fine

        # level > 1 case

        neighbor.node = tree.node.parent.node
        assert neighbor.node is not NULL

        # check in idim direction if the neighbor is at another parent block (+- 1 beyond the local index 1)
        for idim in range(self.ndim):
            # inp value ranges from 0 to 2 
            igc[idim] = tree.node.ig[idim] % 2
            inp[idim] = (igc[idim] + (ii[idim] + 2))//2
            if (inp[idim] != 1):
                neighbor.node = neighbor.node.neighbors[idim][igc[idim]//2].node
                if neighbor.node is NULL:
                    return neighbor_boundary
        
        if (neighbor.node.isleaf):
            return neighbor_coarse
        
        # not a leaf
        for idim in range(self.ndim):
            if (ii[idim]==0 or pole[idim]):
                inc[idim] = igc[idim]
            else:
                inc[idim] = 1-igc[idim]

        neighbor.node = neighbor.node.children[inc[2]][inc[1]][inc[0]].node
        if neighbor.node is NULL:
            return neighbor_unknown
        if (neighbor.node.isleaf):
            return neighbor_sibling
        return neighbor_fine

    cdef void build_connectivity(self):

        # to modify in 2D case
        cdef uint32_t ileaf, i1, i2, i3
        cdef uint32_t ii[3]

        for ileaf in range(self.nleafs):
            for i1 in range(3):
                ii[0] = i1
                for i2 in range(3):
                    ii[1] = i2
                    for i3 in range(3):
                        ii[2] = i3
                        self.build_neighbor_children(ileaf, ii)
                    
    cdef void build_neighbor_children(self, uint32_t ileaf, uint32_t* ii):

        cdef octptr tree, neighbor
        cdef uint32_t neighbor_type
        cdef int ii1[3]

        cdef uint32_t i
        cdef uint32_t ic1, ic2, ic3
        cdef uint32_t inc[3]
        cdef uint32_t ih1, ih2, ih3

        cdef bint pole[3]

        # to do with ndim == 2 case
        if (ii[0] == 1 and ii[1] == 1 and ii[2] == 1):
            # Direct access to numpy arrays
            self.neighbor_index[ileaf, self.nindex(ii)] = ileaf + 1
            self.neighbor_type[ileaf, self.nindex(ii)] = 0
            return

        tree = self.sfc2node[ileaf]
        # here i1, i2, i3 in {-1,0,1} only for find_neighbor
        for i in range(self.ndim):
            ii1[i] = (<int>ii[i])-1
        neighbor_type = self.find_neighbor(&neighbor, tree, ii1, pole)

        # Use direct numpy array indexing 
        self.neighbor_type[ileaf, self.nindex(ii)] = neighbor_type

        if (neighbor_type == neighbor_boundary):
            self.neighbor_index[ileaf, self.nindex(ii)] = 0
        elif (neighbor_type == neighbor_fine):
            self.neighbor_index[ileaf, self.nindex(ii)] = 0

            # 1-i//2 reflects 0 and 2 (neighbor direction) to 1 (neighbor 2nd child) and 0 (neighbor 1st child)
            # but with 1, non-changed direction reserves both children (0 and 1)

            # for inc, 0,1->0, 2,0->4 1,0->2, 1,1->3
            for ic1 in range((1-ii[0]//2)-ii[0]%2, 2-ii[0]//2):
                inc[0] = 2*ii[0]+ic1
                ih1 = ic1
                for ic2 in range((1-ii[1]//2)-ii[1]%2, 2-ii[1]//2):
                    inc[1] = 2*ii[1]+ic2
                    ih2 = ic2
                    for ic3 in range((1-ii[2]//2)-ii[2]%2, 2-ii[2]//2):
                        inc[2] = 2*ii[2]+ic3
                        ih3 = ic3
                        # Access neighbor_children as a numpy array
                        self.neighbor_children[ileaf, self.ncindex(inc)] = neighbor.node.children[ih3][ih2][ih1].node.ileaf

        else: # both coarse and sibling cases
            self.neighbor_index[ileaf, self.nindex(ii)] = neighbor.node.ileaf

    cpdef void test_neighbors(self):

        cdef uint32_t ileaf, idim, iside, level, idim1
        cdef octptr tree, neighbor
        cdef uint32_t ig[3]
        cdef uint32_t ign[3]

        for ileaf in range(self.nleafs):
            tree = self.sfc2node[ileaf]

            for idim in range(self.ndim):
                ig[idim] = tree.node.ig[idim]
            level = tree.node.level

            for idim in range(self.ndim):
                for iside in range(2):
                    neighbor = tree.node.neighbors[idim][iside]
                    if neighbor.node is not NULL:
                        ign[0] = neighbor.node.ig[0]
                        ign[1] = neighbor.node.ig[1]
                        ign[2] = neighbor.node.ig[2]
                        if neighbor.node.level == level:
                            for idim1 in range(self.ndim):
                                if idim1 == idim:
                                    assert (ig[idim1]+iside*2-1 == ign[idim1])
                                else:
                                    assert (ig[idim1] == ign[idim1])
                        # neighbors are all at the same level
                        else:
                            assert False
