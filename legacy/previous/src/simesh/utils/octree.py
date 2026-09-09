import numpy as np
from weakref import ref, proxy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import cached_property

class OctreeNodePointer:

    __slots__ = ['node']

    def __init__(self, node: Optional['OctreeNode'] = None):
        self.node = node

    # def __repr__(self) -> str:
    #     return f"OctreeNodePointer(node={self.node})""

@dataclass
class OctreeNode:
    """
    Octree node class referred to the amrvac/src/amr/mod_forest.t
    """
    __slots__ = ['ig1', 'ig2', 'ig3', 'level', 'igrid', 'is_leaf', 
                 'parent', 'children', 'neighbors', 'next_node', 'prev_node']

    def __init__(self):
        # spatial indices of the grid block, ranges from 1 to ngl1, ngl2, ngl3
        self.ig1: int = 0
        self.ig2: int = 0
        self.ig3: int = 0
        # refinement level
        self.level: int = 0
        # sfc index starting from 1, leave 0 for illegal node in neighbor search
        self.igrid: int = 0
        # if the node is a leaf node
        self.is_leaf: bool = True
        
        # Just create new instances directly
        self.parent: 'OctreeNodePointer' = OctreeNodePointer()
        self.children: np.ndarray = np.array([[[OctreeNodePointer() for _ in range(2)] 
                                             for _ in range(2)] for _ in range(2)])
        self.neighbors: np.ndarray = np.array([[OctreeNodePointer() for _ in range(3)] 
                                            for _ in range(2)])
        self.next_node: 'OctreeNodePointer' = OctreeNodePointer()
        self.prev_node: 'OctreeNodePointer' = OctreeNodePointer()

    def __repr__(self) -> str:
        return f"OctreeNode(ig=({self.ig1},{self.ig2},{self.ig3}), level={self.level}, morton={self.igrid}, leaf={self.is_leaf})"
