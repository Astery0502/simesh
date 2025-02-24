import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Iterable

from simesh.utils.octree import OctreeNode, OctreeNodePointer
from .morton_order import level1_Morton_order

class AMRForest:
    """AMR Forest managing all octrees (each as a level 1 block) """

    def __init__(self, ng1:int, ng2:int, ng3:int, nleafs:int):
        self.ng1 = ng1
        self.ng2 = ng2
        self.ng3 = ng3

        self.ngl1 = ng1 * ng2 * ng3
        self.max_level = 0

        self.nparents: int = 0

        # Initialize the forest array
        self.amr_forest = np.array([[[OctreeNodePointer() for _ in range(ng3)] for _ in range(ng2)] for _ in range(ng1)], dtype=object)

        # levelshi
        self.levelshi: int = 20

        # number of leaf nodes at each level
        self.nleafs_level = np.zeros(self.levelshi).astype(int)

        self.nleafs = nleafs
        # space filling curve morton number to node 
        self.sfc_to_node = np.empty(nleafs, dtype=object)

        # attributes from connectivity
        self.neighbor_type = np.zeros((3,3,3,nleafs), dtype=int)
        self.neighbor = np.zeros((3,3,3,nleafs), dtype=int)
        self.neighbor_child = np.zeros((4,4,4,nleafs), dtype=int)

        # save the level 1 block index in the array of all grids
        self.idx_level1_sfc = np.zeros(self.ngl1, dtype=int)

    def read_forest(self, forest):
        """
        Read the forest from the forest bool list
        """

        def read_node(tree: OctreeNodePointer, ig1:int, ig2:int, ig3:int, level:int):

            nonlocal igrid # sfc index from 1: leafs
            nonlocal inode # total index from 1: forest index

            inode += 1

            assert isinstance(tree.node, OctreeNode), "The node is not a valid OctreeNode, please initialize the forest first"
            assert inode <= len(forest), f"The forest: {len(forest)} is not large enough for inode {inode}"

            # assgin the attributes of the node
            node = tree.node
            node.is_leaf = forest[inode-1]
            node.ig1, node.ig2, node.ig3 = ig1, ig2, ig3
            node.level = level

            # Clean up all children nodes and neighbors nodes as well as next and prev nodes for update
            node.children.fill(None)
            node.neighbors.fill(None)
            node.next_node.node = None
            node.prev_node.node = None

            # assign neighbors for the tree node
            self.asign_tree_neighbor(tree)

            if node.is_leaf:

                # self.add_to_linked_list(tree, level)
                self.nleafs_level[level-1] += 1
                self.sfc_to_node[igrid] = tree
                # leaf number + 1
                igrid += 1
                node.igrid = igrid
            
            else:
                self.nparents += 1
                node.igrid = -1    # parent node morton number

                # Create child nodes in a vectorized way
                child_indices = np.array(np.meshgrid(range(2), range(2), range(2))).reshape(3,-1).T
                child_igs = 2 * np.array([ig1, ig2, ig3]) + child_indices
                
                for idx, (i,j,k) in enumerate(child_indices):
                    child_node = OctreeNode()
                    tree.node.children[i,j,k].node = child_node
                    child_node.parent.node = tree.node
                    read_node(tree.node.children[i,j,k], *child_igs[idx], level+1)

        igrid = 0
        inode = 0
        level = 1

        self.iglevel1_sfc, self.sfc_iglevel1 = level1_Morton_order(self.ng1, self.ng2, self.ng3)

        for isfc in range(self.ngl1):

            # save the level 1 block index in the array of all grids
            self.idx_level1_sfc[isfc] = igrid

            ig1, ig2, ig3 = self.sfc_iglevel1[isfc]
            node = OctreeNode()

            # check if the pointer is valid
            ptr = self.amr_forest[ig1,ig2,ig3]
            assert isinstance(ptr, OctreeNodePointer), f"amr_forest[{ig1},{ig2},{ig3}] is not an OctreeNodePointer"

            ptr.node = node
            ptr.node.parent.node = None # clear parent pointer for level 1 nodes
            read_node(ptr, ig1,ig2,ig3, level)

    def write_forest(self):
        """
        Write the forest bool list 
        """
        def write_node(tree: OctreeNodePointer):
            nonlocal ileaf

            assert isinstance(tree.node, OctreeNode), "Not a valid OctreeNode"
            ileaf += 1
            forest[ileaf-1] = tree.node.is_leaf
            if not tree.node.is_leaf:
                # Process children in Morton order sequence
                child_indices = np.array(np.meshgrid(range(2), range(2), range(2))).reshape(3,-1).T
                for i, j, k in child_indices:
                    child_node = tree.node.children[i,j,k]
                    write_node(child_node)

        forest = np.zeros(self.nparents+self.nleafs, dtype=bool)
        ileaf = 0
        for isfc in range(self.ngl1):
            ig1, ig2, ig3 = self.sfc_iglevel1[isfc] 
            node = self.amr_forest[ig1,ig2,ig3]
            assert isinstance(node, OctreeNodePointer), "The node is not a valid OctreeNodePointer, please initialize the forest first"
            write_node(node)

        return forest 

    def find_root_neighbor(self, neighbor_ptr: OctreeNodePointer, tree: OctreeNodePointer, ig1:int, ig2:int, ig3:int, 
                            periodB:List[bool]=[False, False, False]) -> None:
        
        assert len(periodB) == 3, "periodB should be 3D applied"
        assert isinstance(tree.node, OctreeNode), "The node is not a valid OctreeNode, please initialize the forest first"

        ngs = np.array([self.ng1, self.ng2, self.ng3])
        jgs = np.array([ig1,ig2,ig3]) + np.array([tree.node.ig1, tree.node.ig2, tree.node.ig3])

        # apply periodic boundary condition
        for idir, ngi in enumerate(ngs):
            if periodB[idir]:
                jgs[idir] = jgs[idir] % ngi # different because of the 0-based index
        jgs = jgs.astype(int)
        
        # TODO: pole pi-periodicity
        # ...

        # check if the neighbor is within the grid
        # in fact these combine two cases: 
        # 1. the neighbor is out of the grid
        # 2. the neighbor is within the grid but not initialized,
        # in the second case, the we expect after the neighbor node to define tree neighbor
        if (np.all(jgs>=0) and np.all(jgs<=ngs-1)):

            # check if the pointer is valid
            ptr = self.amr_forest[jgs[0],jgs[1],jgs[2]]
            assert isinstance(ptr, OctreeNodePointer), "Expected OctreeNodePointer but got different type"

            neighbor_ptr.node = ptr.node
        else:
            neighbor_ptr.node = None

    def find_neighbor(self, neighbor_ptr: OctreeNodePointer, tree: OctreeNodePointer,i1:int,i2:int,i3:int) -> int:
        
        assert isinstance(tree.node, OctreeNode), "The node is not a valid OctreeNode, please initialize the forest first"

        pole = False
        level = tree.node.level

        if (level == 1):
            self.find_root_neighbor(neighbor_ptr, tree, i1, i2, i3)
            if neighbor_ptr.node is not None:
                if (neighbor_ptr.node.is_leaf):
                    # sibling neighbor
                    neighbor_type = 3
                else:
                    # fine neighbor
                    neighbor_type = 4
            else:
                # boundary neighbor, return
                neighbor_type = 1
        
        else:
            ig1, ig2, ig3 = tree.node.ig1, tree.node.ig2, tree.node.ig3
            # TODO: pole pi-periodicity

            # ics odd to 2, even to 1 in 0-based; ic^D+i^D ranges from -1 to 3
            ics = np.array([1+ig1%2, 1+ig2%2, 1+ig3%2])
            inps = np.array([int((ics[0]+i1+1)/2)-1, int((ics[1]+i2+1)/2)-1, int((ics[2]+i3+1)/2)-1])
            # assert(np.sum(inps != 0) <= 1), f"The neighbor found in two directions in parent level, {inps}"

            neighbor_ptr.node = tree.node.parent.node
            assert isinstance(neighbor_ptr.node, OctreeNode), "Parent node not valid which should be initialized first"

            for i, inp in enumerate(inps):
                if inp != 0:
                    neighbor_ptr.node = neighbor_ptr.node.neighbors[ics[i]-1,i].node
                    # also two cases: 
                    # 1. the neighbor is out of the grid
                    # 2. the neighbor is within the grid but not initialized
                    if neighbor_ptr.node is None:
                        neighbor_type = 1
                        return neighbor_type
            
            # coarse neighbor
            if (neighbor_ptr.node.is_leaf):
                neighbor_type = 2

            else:
                n_ics = ics.copy()
                for n, i in enumerate([i1, i2, i3]):
                    if i == 0: # TODO: pole pi-periodicity
                        continue
                    else:
                        n_ics[n] = 3-n_ics[n]
                child_ptr = neighbor_ptr.node.children[n_ics[0]-1,n_ics[1]-1,n_ics[2]-1]
                assert isinstance(child_ptr, OctreeNodePointer), "Child pointer is not a valid OctreeNodePointer"
                neighbor_ptr.node = child_ptr.node

                if neighbor_ptr.node is not None:
                    # sibling neighbor
                    if neighbor_ptr.node.is_leaf:
                        neighbor_type = 3
                    # fine neighbor
                    else:
                        neighbor_type = 4
                else:
                    # not initiated yet
                    neighbor_type = 0

        return neighbor_type

    def asign_tree_neighbor(self, tree: OctreeNodePointer):

        assert isinstance(tree.node, OctreeNode), "Not a valid OctreeNode"

        for idir in range(3):
            for iside in range(2):

                i1, i2, i3 = np.eye(3)[idir] * (2*iside-1)
                neighbor_ptr = OctreeNodePointer()
                neighbor_type = self.find_neighbor(neighbor_ptr,tree,i1,i2,i3)

                # sibling or fine neighbor, coarse neighbor is symmetric for the fine neighbor
                if neighbor_type == 3 or neighbor_type == 4:

                    tree.node.neighbors[iside, idir].node = neighbor_ptr.node
                    assert neighbor_ptr.node is not None, "Neighbor node is None"

                    pass # TODO: pole pi-periodicity
                    neighbor_ptr.node.neighbors[1-iside, idir].node = tree.node
                
                elif neighbor_type == 2:
                    pass # coarse neighbor not done here
                
                # boundary neighbor and uninitialized neighbor
                else:
                    tree.node.neighbors[iside, idir].node = None
    
    def build_connectivity(self):

        for igrid in range(self.nleafs):

            nodeptr = self.sfc_to_node[igrid]
            assert isinstance(nodeptr, OctreeNodePointer), "Node pointer is not a valid OctreeNodePointer"

            node = nodeptr.node
            assert isinstance(node, OctreeNode), "Node is not a valid OctreeNode"

            for zi, yi, xi in np.ndindex(3,3,3):
                if (xi == 1 and yi == 1 and zi == 1):
                    self.neighbor_type[xi,yi,zi,igrid] = 0
                    self.neighbor[xi,yi,zi,igrid] = igrid+1
                else:
                    neighbor_ptr = OctreeNodePointer()
                    neighbor_type = self.find_neighbor(neighbor_ptr, nodeptr, xi-1, yi-1, zi-1)
                    if neighbor_type == 1:
                        self.neighbor[xi,yi,zi,igrid] = 0
                    elif neighbor_type == 4:
                        self.neighbor[xi,yi,zi,igrid] = 0
                        self._process_fine_neighbor(igrid, xi, yi, zi, neighbor_ptr)
                    else: 
                        assert isinstance(neighbor_ptr.node, OctreeNode), "Neighbor node is not a valid OctreeNode"
                        self.neighbor[xi,yi,zi,igrid] = neighbor_ptr.node.igrid
                    self.neighbor_type[xi,yi,zi,igrid] = neighbor_type
    
    def _process_fine_neighbor(self, igrid:int, xi:int, yi:int, zi:int, neighbor_ptr: OctreeNodePointer):
        # loop over the local indices of children ic^D
        # calculate the child neighbor indices in the 4x4x4 box
        for ic3 in range(1+int((2-zi)/2), 2-int(zi/2)+1):
            inc3 = 2*(zi-1)+ic3
            ih3 = ic3 # ignore the pole
            for ic2 in range(1+int((2-yi)/2), 2-int(yi/2)+1):
                inc2 = 2*(yi-1)+ic2
                ih2 = ic2 # ignore the 
                for ic1 in range(1+int((2-xi)/2), 2-int(xi/2)+1):
                    inc1 = 2*(xi-1)+ic1
                    ih1 = ic1 # ignore the pole
                    child_node = neighbor_ptr.node.children[ih1-1,ih2-1,ih3-1].node
                    self.neighbor_child[inc1,inc2,inc3,igrid] = child_node.igrid

    @classmethod
    def add_to_linked_list(cls, tree: OctreeNodePointer, level: int):
        # Initialize next pointer to None regardless of level
        # TODO: add prev pointer
        pass
