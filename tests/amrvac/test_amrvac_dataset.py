import os
import tempfile

import numpy as np

from simesh.amrvac import (
    datfile_to_vtk,
    load_from_uniform,
    load_uniform_data,
    open_dataset,
    read_blocks,
    read_uniform,
    write_datfile,
    write_datfile_from_uniform,
)
from simesh.amrvac.amrvac_dataset import AMRVACDataSet
from simesh.amrvac.datio import extract_uniform_data
from simesh.amrvac.layouts import datau_to_udata, udata_to_datau


def _uniform_input(domain_nx=(4, 4, 4), nw=7):
    domain_nx = tuple(domain_nx)
    udata = np.zeros((*domain_nx, nw), dtype=np.double)

    for ix, iy, iz in np.ndindex(*domain_nx):
        for ifield in range(nw):
            udata[ix, iy, iz, ifield] = 1000 * ifield + 100 * ix + 10 * iy + iz

    return udata


def _read_structured_points_vtk(path: str):
    with open(path, "rb") as fh:
        assert fh.readline() == b"# vtk DataFile Version 2.0\n"
        title = fh.readline().decode("ascii").strip()
        assert title == "Uniform grid data"
        assert fh.readline() == b"BINARY\n"
        assert fh.readline() == b"DATASET STRUCTURED_POINTS\n"

        dims = tuple(int(value) for value in fh.readline().decode("ascii").split()[1:])
        origin = np.array([float(value) for value in fh.readline().decode("ascii").split()[1:]], dtype=np.double)
        spacing = np.array([float(value) for value in fh.readline().decode("ascii").split()[1:]], dtype=np.double)
        npoints = int(fh.readline().decode("ascii").split()[1])

        fields = {}
        for _ in range(1000):
            line = fh.readline()
            if not line:
                break

            tokens = line.decode("ascii").split()
            if not tokens:
                continue
            assert tokens[0] == "SCALARS", f"Unexpected VTK token line: {line!r}"
            field_name = tokens[1]
            assert fh.readline() == b"LOOKUP_TABLE default\n"

            raw = fh.read(npoints * 8)
            flat = np.frombuffer(raw, dtype=">f8").astype(np.double, copy=False)
            fields[field_name] = flat.reshape((dims[2], dims[1], dims[0]), order="C").transpose(2, 1, 0)

            trailing = fh.read(1)
            if trailing not in (b"", b"\n"):
                raise AssertionError(f"Unexpected VTK binary terminator: {trailing!r}")

        return {
            "dims": dims,
            "origin": origin,
            "spacing": spacing,
            "fields": fields,
        }


def test_dataset_uniform_full():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input()
        expected = np.transpose(udata, (3, 0, 1, 2))
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path)
        assert ds.ghost_width == 0
        assert not ds.mesh.has_padded_data()

        full_general = ds.uniform_grid(ds.domain_nx, xmin=ds.physical_domain[0], xmax=ds.physical_domain[1])
        full_direct = ds.uniform_full()

        assert full_general.shape == expected.shape
        assert np.array_equal(full_general, expected)
        assert np.array_equal(full_direct, full_general)

        ds.load_data(field_indices=[0, 4, 5, 6])
        subset_general = ds.uniform_grid(
            ds.domain_nx,
            xmin=ds.physical_domain[0],
            xmax=ds.physical_domain[1],
            field_indices=[4, 6],
        )
        subset = ds.uniform_full(field_indices=[4, 6])
        assert subset.shape == (2, 4, 4, 4)
        assert np.array_equal(subset_general[0], expected[4])
        assert np.array_equal(subset_general[1], expected[6])
        assert np.array_equal(subset, subset_general)

        try:
            ds.uniform_full(field_indices=[1])
        except ValueError as exc:
            assert "not loaded" in str(exc)
        else:
            raise AssertionError("uniform_full should fail on unloaded field indices")

        ds.levmax = np.uint32(2)
        try:
            ds.uniform_full()
        except ValueError as exc:
            assert "levmax == 1" in str(exc)
        else:
            raise AssertionError("uniform_full should fail when levmax > 1")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_ghost_mode_uses_mesh_storage():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 4), nw=3)
        expected = np.transpose(udata[..., [0, 2]], (3, 0, 1, 2))
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=1)
        ds.load_data(field_indices=[0, 2])

        padded = ds.mesh.padded_view()
        interior = ds.mesh.interior_view()

        assert padded.shape == (8, 4, 4, 4, 2)
        assert ds.data.shape == (8, 2, 2, 2, 2)
        assert np.array_equal(ds.data, interior)
        assert np.shares_memory(ds.data, padded)
        assert np.array_equal(ds.uniform_full(), expected)

        ds.data[0, 1, 0, 0, 0] = -5.0
        assert ds.mesh.interior_view()[0, 1, 0, 0, 0] == -5.0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_public_api_reads_blocks_and_uniform_data():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 4), nw=3)
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = open_dataset(path, ghost_width=1)
        assert ds.has_ghost_cells
        ds.load_data(field_indices=[0, 2])
        assert ds.blocks().shape == (8, 2, 2, 2, 2)
        assert ds.blocks(include_ghosts=True).shape == (8, 2, 4, 4, 4)

        blocks = read_blocks(path, field_indices=[0, 2])
        ghosted = read_blocks(path, field_indices=[0, 2], ghost_width=1, include_ghosts=True)
        grid = read_uniform(path, resolution=(4, 4, 4), field_indices=[1])
        linear_grid = read_uniform(
            path,
            resolution=(4, 4, 4),
            field_indices=[1],
            ghost_width=1,
            interpolation="linear",
        )

        assert blocks.shape == (8, 2, 2, 2, 2)
        assert ghosted.shape == (8, 2, 4, 4, 4)
        assert grid.shape == (4, 4, 4, 1)
        assert np.array_equal(grid[..., 0], udata[..., 1])
        assert np.array_equal(linear_grid[..., 0], udata[..., 1])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_public_api_2d_singleton_z_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        os.remove(output_path)
        udata = _uniform_input(domain_nx=(4, 4, 1), nw=3)
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = open_dataset(path, ghost_width=1)
        assert int(ds.ndim) == 2
        assert np.array_equal(ds.domain_nx, np.array([4, 4], dtype=np.uint32))
        assert np.array_equal(ds.block_nx, np.array([2, 2], dtype=np.uint32))

        ds.load_data(field_indices=[0, 2])
        assert ds.blocks().shape == (4, 2, 2, 2, 1)
        assert ds.blocks(include_ghosts=True).shape == (4, 2, 4, 4, 3)

        blocks = read_blocks(path, field_indices=[0, 2])
        ghosted = read_blocks(path, field_indices=[0, 2], ghost_width=1, include_ghosts=True)
        grid = read_uniform(path, resolution=(4, 4), field_indices=[1])
        linear_grid = read_uniform(
            path,
            resolution=(4, 4),
            field_indices=[1],
            ghost_width=1,
            interpolation="linear",
        )

        assert blocks.shape == (4, 2, 2, 2, 1)
        assert ghosted.shape == (4, 2, 4, 4, 3)
        assert grid.shape == (4, 4, 1, 1)
        assert np.array_equal(grid[..., 0], udata[..., 1])
        assert np.array_equal(linear_grid[..., 0], udata[..., 1])

        write_datfile(path, output_path, field_indices=[0, 2], overwrite=True)
        rewritten = open_dataset(output_path)
        rewritten_grid = rewritten.uniform_full()
        expected = np.transpose(udata[..., [0, 2]], (3, 0, 1, 2))
        assert int(rewritten.ndim) == 2
        assert rewritten_grid.shape == expected.shape
        assert np.array_equal(rewritten_grid, expected)
    finally:
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(output_path):
            os.remove(output_path)


def test_extract_uniform_data():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input()
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        extracted, header = extract_uniform_data(path)
        assert extracted.shape == udata.shape
        assert np.array_equal(extracted, udata)
        assert np.array_equal(header["domain_nx"], np.array([4, 4, 4]))
        assert np.array_equal(header["block_nx"], np.array([2, 2, 2]))
        assert header["w_names"] == [f"w{i}" for i in range(udata.shape[-1])]

        selected, selected_header = extract_uniform_data(path, field_indices=[0, 4, 6])
        assert selected.shape == (4, 4, 4, 3)
        assert np.array_equal(selected[..., 0], udata[..., 0])
        assert np.array_equal(selected[..., 1], udata[..., 4])
        assert np.array_equal(selected[..., 2], udata[..., 6])
        assert selected_header["w_names"] == ["w0", "w4", "w6"]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_uniform_layout_helpers():
    udata = _uniform_input(domain_nx=(3, 4, 5), nw=2)
    datau = udata_to_datau(udata)

    assert datau.shape == (2, 3, 4, 5)
    assert np.array_equal(datau_to_udata(datau), udata)
    assert np.array_equal(udata_to_datau(datau_to_udata(datau)), datau)


def test_load_uniform_data():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input()
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        loaded, geometry = load_uniform_data(path)
        assert np.array_equal(loaded, udata)
        assert np.array_equal(geometry["xmin"], np.array([0.0, 0.0, 0.0]))
        assert np.array_equal(geometry["xmax"], np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(geometry["domain_nx"], np.array([4, 4, 4]))
        assert np.array_equal(geometry["block_nx"], np.array([2, 2, 2]))
        assert geometry["w_names"] == [f"w{i}" for i in range(udata.shape[-1])]

        loaded_only = load_uniform_data(path, field_indices=[1, 3], return_geometry=False)
        assert loaded_only.shape == (4, 4, 4, 2)
        assert np.array_equal(loaded_only[..., 0], udata[..., 1])
        assert np.array_equal(loaded_only[..., 1], udata[..., 3])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_load_from_uniform_2d_and_3d():
    udata_3d = _uniform_input(domain_nx=(4, 4, 4), nw=3)
    ds_3d = load_from_uniform(
        udata_3d,
        [f"w{i}" for i in range(udata_3d.shape[-1])],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    assert int(ds_3d.ndim) == 3
    assert np.array_equal(ds_3d.uniform_full(), udata_to_datau(udata_3d))

    udata_2d = _uniform_input(domain_nx=(4, 4, 1), nw=3)
    ds_2d = load_from_uniform(
        udata_2d,
        [f"w{i}" for i in range(udata_2d.shape[-1])],
        xmin=np.array([0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2], dtype=np.int32),
    )
    assert int(ds_2d.ndim) == 2
    assert np.array_equal(ds_2d.domain_nx, np.array([4, 4], dtype=np.uint32))
    assert np.array_equal(ds_2d.block_nx, np.array([2, 2], dtype=np.uint32))
    assert np.array_equal(ds_2d.uniform_full(), udata_to_datau(udata_2d))


def test_dataset_getitem_returns_udata_layout():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input()
        write_datfile_from_uniform(
            path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path)
        sub = ds[1:3:4j, 0:2:4j, 2:4:4j]
        assert sub.shape == (2, 2, 2, udata.shape[-1])
        assert np.array_equal(sub, udata[1:3, 0:2, 2:4, :])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_datfile_to_vtk():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as dat_tmp:
        dat_path = dat_tmp.name
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as vtk_tmp:
        vtk_path = vtk_tmp.name

    try:
        os.remove(vtk_path)
        udata = _uniform_input()
        write_datfile_from_uniform(
            dat_path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        datfile_to_vtk(dat_path, vtk_path, field_indices=[1, 3])

        vtk = _read_structured_points_vtk(vtk_path)

        assert vtk["dims"] == (4, 4, 4)
        assert np.allclose(vtk["origin"], np.array([0.0, 0.0, 0.0]))
        assert np.allclose(vtk["spacing"], np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
        assert set(vtk["fields"]) == {"w1", "w3"}
        assert np.array_equal(vtk["fields"]["w1"], udata[..., 1])
        assert np.array_equal(vtk["fields"]["w3"], udata[..., 3])
    finally:
        if os.path.exists(dat_path):
            os.remove(dat_path)
        if os.path.exists(vtk_path):
            os.remove(vtk_path)


def test_datfile_to_vtk_2d_singleton_z():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as dat_tmp:
        dat_path = dat_tmp.name
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as vtk_tmp:
        vtk_path = vtk_tmp.name

    try:
        os.remove(vtk_path)
        udata = _uniform_input(domain_nx=(4, 4, 1), nw=3)
        write_datfile_from_uniform(
            dat_path,
            udata,
            [f"w{i}" for i in range(udata.shape[-1])],
            xmin=np.array([0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2], dtype=np.int32),
            overwrite=True,
        )

        datfile_to_vtk(dat_path, vtk_path, field_indices=[1])

        vtk = _read_structured_points_vtk(vtk_path)

        assert vtk["dims"] == (4, 4, 1)
        assert np.allclose(vtk["origin"], np.array([0.0, 0.0, 0.0]))
        assert np.allclose(vtk["spacing"], np.array([1.0 / 3.0, 1.0 / 3.0, 1.0]))
        assert set(vtk["fields"]) == {"w1"}
        assert np.array_equal(vtk["fields"]["w1"], udata[..., 1])
    finally:
        if os.path.exists(dat_path):
            os.remove(dat_path)
        if os.path.exists(vtk_path):
            os.remove(vtk_path)


def run_tests():
    print("Running tests for AMRVAC dataset...")
    test_dataset_uniform_full()
    print("test_dataset_uniform_full passed")
    test_dataset_ghost_mode_uses_mesh_storage()
    print("test_dataset_ghost_mode_uses_mesh_storage passed")
    test_public_api_reads_blocks_and_uniform_data()
    print("test_public_api_reads_blocks_and_uniform_data passed")
    test_public_api_2d_singleton_z_roundtrip()
    print("test_public_api_2d_singleton_z_roundtrip passed")
    test_extract_uniform_data()
    print("test_extract_uniform_data passed")
    test_uniform_layout_helpers()
    print("test_uniform_layout_helpers passed")
    test_load_uniform_data()
    print("test_load_uniform_data passed")
    test_load_from_uniform_2d_and_3d()
    print("test_load_from_uniform_2d_and_3d passed")
    test_dataset_getitem_returns_udata_layout()
    print("test_dataset_getitem_returns_udata_layout passed")
    test_datfile_to_vtk()
    print("test_datfile_to_vtk passed")
    test_datfile_to_vtk_2d_singleton_z()
    print("test_datfile_to_vtk_2d_singleton_z passed")
    print("All tests passed for AMRVAC dataset!")


if __name__ == "__main__":
    run_tests()
