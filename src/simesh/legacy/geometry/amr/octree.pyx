# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject

cdef class OctreeNode:
    """
    Octree node class referred to the amrvac/src/amr/mod_forest.t
    """
    cdef:
        public int ig1, ig2, ig3, level, igrid
        public bint is_leaf
        public OctreeNode* parent
        public OctreeNode* children[2][2][2]
        public OctreeNode* neighbors[2][3]
        public OctreeNode* next_node
        public OctreeNode* prev_node

    def __cinit__(self, int ig1=0, int ig2=0, int ig3=0, int level=0, 
                  int igrid=0, bint is_leaf=True):
        self.ig1 = ig1
        self.ig2 = ig2
        self.ig3 = ig3
        self.level = level
        self.igrid = igrid
        self.is_leaf = is_leaf
        
        # Initialize arrays with None
        self.children = np.full((2, 2, 2), None, dtype=object)
        self.neighbors = np.full((2, 3), None, dtype=object)
        self.parent = None
        self.next_node = None
        self.prev_node = None

    def __repr__(self) -> str:
        return f"OctreeNode(ig=({self.ig1},{self.ig2},{self.ig3}), level={self.level}, morton={self.igrid}, leaf={self.is_leaf})"