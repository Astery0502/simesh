import numpy as np

from simesh.legacy.geometry.amr.amr_forest import AMRForest as LegacyAMRForest
from simesh.legacy.meshes.amr_mesh import AMRMesh as LegacyAMRMesh
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh.utils.lib.amr.morton import fill_morton_mapping3D, pmorton3D


def interleave_bits(ign):
    answer = 0
    ndim = len(ign)
    for i in range(0, 64 // ndim):
        if ndim == 1:
            return ign[0]
        if ndim == 2:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1
            answer |= (bit_x << (2 * i)) | (bit_y << (2 * i + 1))
        elif ndim == 3:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1
            bit_z = (ign[2] >> i) & 1
            answer |= (bit_x << (3 * i)) | (bit_y << (3 * i + 1)) | (bit_z << (3 * i + 2))
    return answer


def _level1_sfc_index(root_grid, coord):
    ig2morton = np.zeros(root_grid, dtype=np.uint32)
    morton2ig = np.zeros((np.prod(root_grid), 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, *root_grid)
    return int(ig2morton[coord])


def _forest_flags(root_grid, refined_coords=()):
    refined_roots = {_level1_sfc_index(root_grid, coord) for coord in refined_coords}
    flags = []

    for isfc in range(np.prod(root_grid)):
        if isfc in refined_roots:
            flags.append(False)
            flags.extend([True] * 8)
        else:
            flags.append(True)

    return np.array(flags, dtype=np.int32)


def _mesh_pair(is_leaf, root_grid=(2, 2, 2), nghost=2, nfields=1, block_nx=(4, 4, 4)):
    ng1, ng2, ng3 = root_grid

    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    block_nx = np.array(block_nx, dtype=np.uint32)
    domain_nx = np.array([ng1, ng2, ng3], dtype=np.uint32) * block_nx
    xmin = np.array([0.0, 0.0, 0.0], dtype=np.double)
    xmax = np.array([1.0, 1.0, 1.0], dtype=np.double)
    mesh = AMRMesh(3, block_nx, domain_nx, xmin, xmax, np.uint32(nghost), np.uint32(nfields), forest)

    legacy_forest = LegacyAMRForest(ng1, ng2, ng3, int(forest.nleafs))
    legacy_forest.read_forest(is_leaf.astype(bool))
    legacy_forest.build_connectivity()
    legacy_mesh = LegacyAMRMesh(
        (float(xmin[0]), float(xmax[0])),
        (float(xmin[1]), float(xmax[1])),
        (float(xmin[2]), float(xmax[2])),
        [f"w{i}" for i in range(nfields)],
        block_nx.astype(int),
        domain_nx.astype(int),
        legacy_forest,
        nghostcells=nghost,
    )

    return forest, mesh, legacy_mesh, block_nx


def _refined_fixture(nghost=2, nfields=1, block_nx=(4, 4, 4)):
    root_grid = (2, 2, 2)
    is_leaf = _forest_flags(root_grid, refined_coords=[(0, 0, 0)])
    return _mesh_pair(is_leaf, root_grid=root_grid, nghost=nghost, nfields=nfields, block_nx=block_nx)


def _sample_block_data(nleafs, nfields, block_nx):
    data = np.zeros((nleafs, nfields, *block_nx), dtype=np.double)
    x = np.arange(block_nx[0], dtype=np.double)[:, None, None]
    y = np.arange(block_nx[1], dtype=np.double)[None, :, None]
    z = np.arange(block_nx[2], dtype=np.double)[None, None, :]

    for ileaf in range(nleafs):
        for ifield in range(nfields):
            data[ileaf, ifield] = 1000.0 * ileaf + 100.0 * ifield + 10.0 * x + y + 0.01 * z

    return data


def _patterned_block_data(nleafs, nfields, block_nx, pattern):
    data = _sample_block_data(nleafs, nfields, block_nx)

    if pattern == "affine":
        return data

    x = np.arange(block_nx[0], dtype=np.double)[:, None, None]
    y = np.arange(block_nx[1], dtype=np.double)[None, :, None]
    z = np.arange(block_nx[2], dtype=np.double)[None, None, :]

    if pattern == "step":
        for ileaf in range(nleafs):
            for ifield in range(nfields):
                data[ileaf, ifield] = (
                    100.0 * ileaf
                    + 10.0 * ifield
                    + 3.0 * (x >= block_nx[0] // 2)
                    - 5.0 * (y < block_nx[1] // 2)
                    + 7.0 * (z == block_nx[2] - 1)
                )
        return data

    if pattern == "quadratic":
        for ileaf in range(nleafs):
            for ifield in range(nfields):
                data[ileaf, ifield] = 0.5 * x * x - 0.25 * y * y + z + 11.0 * ileaf + ifield
        return data

    raise ValueError(f"unknown data pattern: {pattern}")


def _load_legacy_interior(mesh, data):
    mesh.data[...] = 0.0
    mesh.data[
        :,
        mesh.ixMmin[0]:mesh.ixMmax[0] + 1,
        mesh.ixMmin[1]:mesh.ixMmax[1] + 1,
        mesh.ixMmin[2]:mesh.ixMmax[2] + 1,
        :,
    ] = np.transpose(data, (0, 2, 3, 4, 1))


def _legacy_neighbor_type_as_cython_layout(legacy_forest):
    nleafs = legacy_forest.neighbor_type.shape[3]
    neighbor_type = np.zeros((nleafs, 27), dtype=np.uint32)

    for i, j, k in np.ndindex(3, 3, 3):
        neighbor_type[:, i + j * 3 + k * 9] = legacy_forest.neighbor_type[i, j, k, :]

    return neighbor_type


def test_pmorton3D():
    for i, j, k in np.ndindex(5, 5, 5):
        assert pmorton3D(i, j, k) == interleave_bits([i, j, k])


def test_morton2D():
    morton2D = np.zeros(25)
    for i, j in np.ndindex(5, 5):
        morton2D[i] = interleave_bits([i, j])
    sorted_idx = np.argsort(morton2D)

    morton3D = np.zeros(25)
    for i, j in np.ndindex(5, 5):
        morton3D[i] = interleave_bits([i, j, 0])
    sorted_idx_3D = np.argsort(morton3D)

    assert np.all(sorted_idx == sorted_idx_3D)


def test_morton_mapping3D():
    n1, n2, n3 = 4, 4, 4
    ig2morton = np.zeros((n1, n2, n3), dtype=np.uint32)
    morton2ig = np.zeros((n1 * n2 * n3, 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, n1, n2, n3)

    for i, j, k in np.ndindex(n1, n2, n3):
        assert ig2morton[i, j, k] == pmorton3D(i, j, k)
        assert np.all(morton2ig[pmorton3D(i, j, k)] == [i, j, k])

    n1, n2, n3 = 5, 5, 5
    ig2morton = np.zeros((n1, n2, n3), dtype=np.uint32)
    morton2ig = np.zeros((n1 * n2 * n3, 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, n1, n2, n3)
    mortonlist = ig2morton.flatten()
    assert np.array_equal(np.sort(mortonlist), np.arange(n1 * n2 * n3))
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
    assert np.all(forest_list == is_leaf)

    ng1 = ng2 = 4
    ng3 = 1
    is_leaf = np.ones(20, dtype=np.int32)
    is_leaf[0] = 0
    forest2d = AMRForest(2, ng1, ng2, ng3, is_leaf)
    forest_list2d = forest2d.write_forest()
    assert forest2d.nleafs == 19
    assert forest2d.nparents == 1
    assert forest2d.max_level == 2
    assert np.all(forest_list2d == is_leaf)


def test_find_neighbors():
    ng1 = ng2 = ng3 = 2
    is_leaf = np.ones(16, dtype=np.int32)
    is_leaf[0] = 0
    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    forest.test_neighbors()

    ng1 = ng2 = 4
    ng3 = 1
    is_leaf = np.ones(20, dtype=np.int32)
    is_leaf[0] = 0
    forest2d = AMRForest(2, ng1, ng2, ng3, is_leaf)
    forest2d.test_neighbors()


def test_init_amr_mesh():
    forest, mesh, _, _ = _refined_fixture(nghost=0, nfields=3, block_nx=(10, 10, 10))
    assert mesh.rnode.shape == (forest.nleafs, 9)
    assert not mesh.has_padded_data()


def test_init_amr_mesh_with_ghost_storage():
    forest, mesh, _, block_nx = _refined_fixture(nghost=2, nfields=3, block_nx=(4, 6, 8))
    padded = mesh.padded_view()
    assert padded.shape == (
        forest.nleafs,
        int(block_nx[0]) + 4,
        int(block_nx[1]) + 4,
        int(block_nx[2]) + 4,
        3,
    )


def test_mesh_load_interior_data_exposes_view():
    forest, mesh, _, block_nx = _refined_fixture(nghost=1, nfields=2, block_nx=(4, 4, 4))
    data = _sample_block_data(int(forest.nleafs), 2, tuple(int(v) for v in block_nx))
    mesh.load_interior_data(data)

    interior = mesh.interior_view()
    padded = mesh.padded_view()

    assert np.array_equal(interior, data)
    assert np.shares_memory(interior, padded)

    interior[0, 1, 0, 0, 0] = -7.0
    assert mesh.interior_view()[0, 1, 0, 0, 0] == -7.0


def test_uniform_grid_zero_order():
    forest, mesh, _, _ = _refined_fixture(nghost=0, nfields=3, block_nx=(10, 10, 10))
    data = np.ones((forest.nleafs, 3, 10, 10, 10), dtype=np.double)
    uniform_grid = np.zeros((3, 20, 20, 20), dtype=np.double)
    mesh.uniform_grid_zero_order(
        data,
        uniform_grid,
        np.array([20, 20, 20], dtype=np.uint32),
        np.array([0.0, 0.0, 0.0], dtype=np.double),
        np.array([1.0, 1.0, 1.0], dtype=np.double),
    )
    assert np.all(uniform_grid == 1)


def test_uniform_full_level1():
    ng1 = ng2 = ng3 = 2
    is_leaf = np.ones(ng1 * ng2 * ng3, dtype=np.int32)
    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    bsize = np.array([2, 2, 2], dtype=np.uint32)
    dsize = np.array([4, 4, 4], dtype=np.uint32)
    xmin = np.array([0.0, 0.0, 0.0], dtype=np.double)
    xmax = np.array([1.0, 1.0, 1.0], dtype=np.double)
    mesh = AMRMesh(3, bsize, dsize, xmin, xmax, 0, 2, forest)

    data = np.zeros((forest.nleafs, 2, 2, 2, 2), dtype=np.double)
    expected = np.zeros((2, 4, 4, 4), dtype=np.double)

    morton2ig = np.zeros((ng1 * ng2 * ng3, 3), dtype=np.uint32)
    fill_morton_mapping3D(np.zeros((ng1, ng2, ng3), dtype=np.uint32), morton2ig, ng1, ng2, ng3)

    for ileaf in range(forest.nleafs):
        ig = morton2ig[ileaf]
        value0 = 100 * ig[0] + 10 * ig[1] + ig[2]
        value1 = value0 + 1000
        data[ileaf, 0, :, :, :] = value0
        data[ileaf, 1, :, :, :] = value1

        x0, y0, z0 = ig * bsize
        x1, y1, z1 = (ig + 1) * bsize
        expected[0, x0:x1, y0:y1, z0:z1] = value0
        expected[1, x0:x1, y0:y1, z0:z1] = value1

    uniform_grid = np.zeros((2, 4, 4, 4), dtype=np.double)
    mesh.uniform_full_level1(data, uniform_grid)
    assert np.array_equal(uniform_grid, expected)


def test_getbc_matches_legacy_reference():
    forest, mesh, legacy_mesh, block_nx = _refined_fixture(nghost=2, nfields=2, block_nx=(4, 4, 4))
    data = _sample_block_data(int(forest.nleafs), 2, tuple(int(v) for v in block_nx))

    mesh.load_interior_data(data)
    _load_legacy_interior(legacy_mesh, data)

    mesh.apply_ghost_cells()
    legacy_mesh.getbc()

    assert np.allclose(mesh.padded_view(), legacy_mesh.data)


def test_getbc_matches_legacy_across_topologies_and_patterns():
    cases = [
        ("uniform_level1", (2, 2, 2), [], 1, 2, (4, 4, 4)),
        ("corner_refined", (2, 2, 2), [(0, 0, 0)], 2, 2, (4, 4, 4)),
        ("opposite_corner_refined", (2, 2, 2), [(1, 1, 1)], 2, 3, (4, 4, 4)),
        ("interior_refined", (3, 3, 3), [(1, 1, 1)], 2, 2, (4, 4, 4)),
        ("adjacent_refined", (3, 2, 2), [(0, 0, 0), (1, 0, 0)], 2, 2, (4, 4, 4)),
        ("uniform_wide_blocks", (2, 2, 2), [], 1, 2, (6, 4, 8)),
    ]

    for name, root_grid, refined_coords, nghost, nfields, block_nx in cases:
        is_leaf = _forest_flags(root_grid, refined_coords)
        forest, mesh, legacy_mesh, block_nx_array = _mesh_pair(
            is_leaf,
            root_grid=root_grid,
            nghost=nghost,
            nfields=nfields,
            block_nx=block_nx,
        )
        expected_neighbor_type = _legacy_neighbor_type_as_cython_layout(legacy_mesh.forest)
        assert np.array_equal(np.asarray(forest.neighbor_type), expected_neighbor_type), name

        for pattern in ["affine", "step", "quadratic"]:
            data = _patterned_block_data(int(forest.nleafs), nfields, tuple(int(v) for v in block_nx_array), pattern)
            mesh.load_interior_data(data)
            _load_legacy_interior(legacy_mesh, data)

            before_interior = data.copy()
            mesh.apply_ghost_cells()
            legacy_mesh.getbc()

            assert np.array_equal(mesh.interior_view(), before_interior), (name, pattern)
            np.testing.assert_allclose(
                mesh.padded_view(),
                legacy_mesh.data,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"{name} with {pattern} data",
            )


def run_tests():
    print("Running tests for amr submodule...")
    test_pmorton3D()
    print("test_pmorton3D passed")
    test_morton2D()
    print("test_morton2D passed")
    test_morton_mapping3D()
    print("test_morton_mapping3D passed")
    test_init_amr_forest()
    print("test_init_amr_forest passed")
    test_find_neighbors()
    print("test_find_neighbors passed")
    test_init_amr_mesh()
    print("test_init_amr_mesh passed")
    test_init_amr_mesh_with_ghost_storage()
    print("test_init_amr_mesh_with_ghost_storage passed")
    test_mesh_load_interior_data_exposes_view()
    print("test_mesh_load_interior_data_exposes_view passed")
    test_uniform_grid_zero_order()
    print("test_uniform_grid_zero_order passed")
    test_uniform_full_level1()
    print("test_uniform_full_level1 passed")
    test_getbc_matches_legacy_reference()
    print("test_getbc_matches_legacy_reference passed")
    test_getbc_matches_legacy_across_topologies_and_patterns()
    print("test_getbc_matches_legacy_across_topologies_and_patterns passed")
    print("All tests passed for amr submodule!")


if __name__ == "__main__":
    run_tests()
