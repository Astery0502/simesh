import os
import tempfile

import numpy as np

from simesh.amrvac.amrvac_dataset import AMRVACDataSet
from simesh.amrvac.datio import header_template, write_datfile_from_sfc
from simesh.utils.lib.amr.morton import fill_morton_mapping3D


def _sfc_input(domain_nx=(4, 4, 4), block_nx=(2, 2, 2), nw=3):
    domain_nx = np.array(domain_nx, dtype=np.int32)
    block_nx = np.array(block_nx, dtype=np.int32)
    nblev1 = domain_nx // block_nx
    nleafs = int(np.prod(nblev1))

    morton2ig = np.zeros((nleafs, 3), dtype=np.uint32)
    fill_morton_mapping3D(np.zeros(tuple(nblev1), dtype=np.uint32), morton2ig, *nblev1)

    data = np.zeros((nleafs, nw, block_nx[0], block_nx[1], block_nx[2]), dtype=np.double)
    expected = np.zeros((nw, domain_nx[0], domain_nx[1], domain_nx[2]), dtype=np.double)

    for ileaf in range(nleafs):
        ig = morton2ig[ileaf]
        x0, y0, z0 = ig * block_nx
        x1, y1, z1 = (ig + 1) * block_nx

        for ifield in range(nw):
            value = 1000 * ifield + 100 * ig[0] + 10 * ig[1] + ig[2]
            data[ileaf, ifield, :, :, :] = value
            expected[ifield, x0:x1, y0:y1, z0:z1] = value

    header = header_template.copy()
    header["nw"] = nw
    header["w_names"] = [f"w{i}" for i in range(nw)]
    header["levmax"] = 1
    header["nleafs"] = nleafs
    header["nparents"] = 0
    header["xmin"] = np.array([0.0, 0.0, 0.0], dtype=np.double)
    header["xmax"] = np.array([1.0, 1.0, 1.0], dtype=np.double)
    header["domain_nx"] = domain_nx
    header["block_nx"] = block_nx

    is_leaf = np.ones(nleafs, dtype=np.int32)
    block_lvls = np.ones(nleafs, dtype=np.int32)
    block_ixs = morton2ig.astype(np.int32) + 1
    tree_info = (block_lvls, block_ixs, np.zeros(nleafs, dtype=np.int64))

    return data, header, is_leaf, tree_info, expected


def test_write_datfile_from_sfc_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        data, header, is_leaf, tree_info, expected = _sfc_input()
        write_datfile_from_sfc(path, data, header, is_leaf, tree_info, overwrite=True)

        ds = AMRVACDataSet(path)
        assert ds.wnames == header["w_names"]
        reconstructed = ds.uniform_grid(ds.domain_nx, xmin=ds.physical_domain[0], xmax=ds.physical_domain[1])
        assert np.array_equal(reconstructed, expected)
        assert np.array_equal(ds.uniform_full(), reconstructed)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_write_datfile_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp_in:
        path_in = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp_out:
        path_out = tmp_out.name

    try:
        data, header, is_leaf, tree_info, expected = _sfc_input()
        write_datfile_from_sfc(path_in, data, header, is_leaf, tree_info, overwrite=True)

        ds = AMRVACDataSet(path_in)
        ds.load_data(field_indices=[0, 2])
        original_grid = ds.uniform_grid(ds.domain_nx, xmin=ds.physical_domain[0], xmax=ds.physical_domain[1])

        os.remove(path_out)
        ds.write_datfile(path_out)

        rewritten = AMRVACDataSet(path_out)
        assert rewritten.wnames == ["w0", "w2"]
        rewritten_grid = rewritten.uniform_grid(
            rewritten.domain_nx,
            xmin=rewritten.physical_domain[0],
            xmax=rewritten.physical_domain[1],
        )
        assert np.array_equal(original_grid, expected[[0, 2]])
        assert np.array_equal(rewritten_grid, original_grid)
        assert np.array_equal(rewritten.uniform_full(), rewritten_grid)
    finally:
        if os.path.exists(path_in):
            os.remove(path_in)
        if os.path.exists(path_out):
            os.remove(path_out)


def run_tests():
    print("Running tests for AMRVAC writing...")
    test_write_datfile_from_sfc_roundtrip()
    print("test_write_datfile_from_sfc_roundtrip passed")
    test_dataset_write_datfile_roundtrip()
    print("test_dataset_write_datfile_roundtrip passed")
    print("All tests passed for AMRVAC writing!")


if __name__ == "__main__":
    run_tests()
