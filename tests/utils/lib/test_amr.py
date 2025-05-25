import numpy as np
from src.simesh.utils.lib.amr.morton import pmorton3D, fill_morton_mapping3D
from src.simesh.utils.lib.amr.forest import AMRForest
from src.simesh.utils.lib.amr.mesh import AMRMesh

def interleave_bits(ign):
    answer = 0
    ndim = len(ign)
    for i in range(0,64//ndim):  

        if ndim == 1:
            return ign[0]

        elif ndim == 2:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1

            answer |= (bit_x << (2*i)) | (bit_y << (2*i + 1))
        
        elif ndim == 3:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1
            bit_z = (ign[2] >> i) & 1

            answer |= (bit_x << (3*i)) | (bit_y << (3*i + 1)) | (bit_z << (3*i + 2))
        
    return answer

def test_pmorton3D():

    for i,j,k in np.ndindex(5,5,5):
        assert pmorton3D(i, j, k) == interleave_bits([i, j, k]), f"pmorton3D({i}, {j}, {k}) = {pmorton3D(i, j, k)}, but interleave_bits([{i}, {j}, {k}]) = {interleave_bits([i, j, k])}"

def test_morton_mapping3D():

    # test 4x4x4 grid
    n1, n2, n3 = 4, 4, 4
    ig2morton = np.zeros((n1, n2, n3), dtype=np.uint32)
    morton2ig = np.zeros((n1*n2*n3, 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, n1, n2, n3)
    # test the mapping is correct
    for i,j,k in np.ndindex(n1, n2, n3):
        assert ig2morton[i,j,k] == pmorton3D(i, j, k)
        assert np.all(morton2ig[pmorton3D(i, j, k)] == [i, j, k])

    n1, n2, n3 = 5, 5, 5
    ig2morton = np.zeros((n1, n2, n3), dtype=np.uint32)
    morton2ig = np.zeros((n1*n2*n3, 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, n1, n2, n3)
    # test the mapping on non-power of 2 grid
    mortonlist = ig2morton.flatten()
    assert np.array_equal(np.sort(mortonlist), np.arange(n1*n2*n3))
    assert np.all(morton2ig < np.array([n1, n2, n3]))

def test_init_amr_forest():
    ng1 = ng2 = ng3 = 2
    is_leaf = np.ones(16, dtype=np.int32)
    is_leaf[0] = 0
    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    forest_list = forest.write_forest()
    assert forest.nleafs == 15
    assert forest.nparents == 1
    assert forest.max_level == 2

    for i in range(forest_list.shape[0]):
        assert forest_list[i] == is_leaf[i]

def test_find_neighbors():
    ng1 = ng2 = ng3 = 2
    is_leaf = np.ones(16, dtype=np.int32)
    is_leaf[0] = 0
    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    forest.test_neighbors()

def test_getbc():
    ng1 = ng2 = ng3 = 2
    is_leaf = np.ones(16, dtype=np.int32)
    is_leaf[0] = 0
    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    bsize = np.array([10, 10, 10], dtype=np.uint32)
    dsize = np.array([20, 20, 20], dtype=np.uint32)
    mesh = AMRMesh(3, bsize, dsize, 2, 1, forest)
    mesh.getbc()
    
def run_tests():
    print("Running tests for amr submodule...")
    test_pmorton3D()
    print("test_pmorton3D passed")
    test_morton_mapping3D()
    print("test_morton_mapping3D passed")
    test_init_amr_forest()
    print("test_init_amr_forest passed")
    test_find_neighbors()
    print("test_find_neighbors passed")
    print("All tests passed for amr submodule!")

if __name__ == "__main__":
    run_tests()