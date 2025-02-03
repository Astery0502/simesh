import numpy as np
import pytest
from simesh.geometry.amr.amr_forest import AMRForest
from simesh.utils.octree import OctreeNodePointer

@pytest.fixture
def common_forest():
    """
    The common forest structure for tests, and it is like:
    2 x 2 x 2 grid, with 8 level 1 nodes, and the first node is non-leaf, 
    thus the total number of nodes is 16, leaf nodes are 15.
    """
    forest = AMRForest(ng1=2, ng2=2, ng3=2, nleafs=15)
    forest_config = np.ones(16, dtype=bool)
    forest_config[1] = False  # Second element is non-leaf
    forest.read_forest(forest_config)
    return forest

def test_forest_structure(common_forest):
    """Test the basic structure of the forest with the given configuration"""
    # Verify total number of leaf nodes
    assert common_forest.nleafs == 15  # One node is non-leaf
    
    # Check specific node states
    node_ptr = common_forest.sfc_to_node[1]  # Second node
    assert not node_ptr.node.is_leaf  # Should be non-leaf
    
    # Check other nodes are leaves
    for i in range(16):
        if i != 1:
            node_ptr = common_forest.sfc_to_node[i]
            assert node_ptr.node.is_leaf

def test_neighbor_relationships(common_forest):
    """Test neighbor relationships in the forest"""
    # Get the non-leaf node and its neighbors
    non_leaf_node = common_forest.sfc_to_node[1]
    neighbor_ptr = OctreeNodePointer()
    
    # Test neighbors in different directions
    directions = [
        (-1, 0, 0),  # Left
        (1, 0, 0),   # Right
        (0, -1, 0),  # Down
        (0, 1, 0),   # Up
        (0, 0, -1),  # Back
        (0, 0, 1),   # Front
    ]
    
    for dx, dy, dz in directions:
        neighbor_type = common_forest.find_neighbor(neighbor_ptr, non_leaf_node, dx, dy, dz)
        assert neighbor_type in [1, 2, 3, 4]  # Valid neighbor types

def test_periodic_boundaries(common_forest):
    """Test periodic boundary conditions with the forest structure"""
    # Test with periodic boundaries in all directions
    periodic = [True, True, True]
    
    # Test for first node (0,0,0)
    first_node = common_forest.sfc_to_node[0]
    neighbor_ptr = OctreeNodePointer()
    
    # Test negative direction with periodic boundary
    neighbor_type = common_forest.find_neighbor(
        neighbor_ptr, first_node, -1, 0, 0, periodB=periodic
    )
    assert neighbor_type != 1  # Should not be boundary due to periodicity
    assert neighbor_ptr.node is not None

def test_level_distribution(common_forest):
    """Test the distribution of nodes across levels"""
    # Check number of nodes at each level
    level_counts = [0] * common_forest.levelshi
    
    for i in range(16):
        node = common_forest.sfc_to_node[i].node
        level_counts[node.level] += 1
    
    # The non-leaf node should have children at a higher level
    assert level_counts[1] == 15  # 15 leaf nodes at level 1
    assert level_counts[0] == 1   # 1 non-leaf node at level 0

def test_write_forest_config(common_forest):
    """Test writing the forest configuration"""
    # Write the forest configuration
    forest_written = common_forest.write_forest()
    
    # Verify the written configuration matches the original
    assert np.all(forest_written == common_forest.forest)

def test_build_connectivity(common_forest):
    """ mainly test the fine child neighbor from build_connectivity method """
    common_forest.build_connectivity()
    # test the fine child neighbor of the non-leaf node
    test_neighbor = common_forest.neighbor
    test_neighbor_type = common_forest.neighbor_type
    test_neighbor_child = common_forest.neighbor_child

    assert test_neighbor.shape[-1] == common_forest.nleafs
    assert test_neighbor_type.shape[-1] == common_forest.nleafs
    assert test_neighbor_child.shape[-1] == common_forest.nleafs

    for i in range(common_forest.nleafs):
        assert test_neighbor[1,1,1,i] == i

    assert test_neighbor_type[0,0,0,0] == 1
    assert test_neighbor_type[0,1,1,1] == 2
    assert test_neighbor_type[2,2,2,0] == 3
    assert test_neighbor_type[2,1,1,0] == 4

    for i in range(4):
        for j in range(4):
            for k in range(4):
                if (j in [1,2] and k in [1,2]):
                    assert test_neighbor_child[3,j,k,0]>0
                else:
                    assert test_neighbor_child[3,j,k,0] == 0
