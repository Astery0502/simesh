import numpy as np
from weakref import ref, proxy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import cached_property

class OctreeNodePointer:
    def __init__(self, node: Optional['OctreeNode'] = None):
        self.node = node

    # def __repr__(self) -> str:
    #     return f"OctreeNodePointer(node={self.node})"

@dataclass
class OctreeNode:
    """
    Octree node class referred to the amrvac/src/amr/mod_forest.t
    """

    # spatial indices of the grid block, ranges from 1 to ngl1, ngl2, ngl3
    ig1: int = 0
    ig2: int = 0
    ig3: int = 0
    # refinement level
    level: int = 0
    # morton index
    igrid: int = 0
    # if the node is a leaf node
    is_leaf: bool = True

    # parent node
    parent: 'OctreeNodePointer' = field(default_factory=lambda: OctreeNodePointer())
    # children nodes
    children: np.ndarray = field(default_factory=lambda: np.array([[[OctreeNodePointer() for _ in range(2)] for _ in range(2)] for _ in range(2)]))
    # neighbor nodes 
    neighbors: np.ndarray = field(default_factory=lambda: np.array([[OctreeNodePointer() for _ in range(3)] for _ in range(2)]))
    # next node at refinement level
    next_node: 'OctreeNodePointer' = field(default_factory=lambda: OctreeNodePointer())
    # previous node at refinement level
    prev_node: 'OctreeNodePointer' = field(default_factory=lambda: OctreeNodePointer())


    def __repr__(self) -> str:
        return f"OctreeNode(ig=({self.ig1},{self.ig2},{self.ig3}), level={self.level}, morton={self.igrid}, leaf={self.is_leaf})"
