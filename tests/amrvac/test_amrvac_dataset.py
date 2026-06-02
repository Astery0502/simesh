import os
import tempfile
import zlib
from pathlib import Path
from struct import pack

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


HEAVY_AMRVAC_CURRENT_FIXTURE = Path("data/weno509_sub_0000.dat")
HEAVY_AMRVAC_CURRENT_REPORT_DIR = Path("report/amrvac-current")
HEAVY_AMRVAC_CURRENT_THRESHOLD = 300.0
HEAVY_AMRVAC_CURRENT_EPSILON = np.finfo(np.double).eps
HEAVY_AMRVAC_CURRENT_SMOOTHING_PASSES = 2


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


def _smooth_current_plane(values, passes):
    smoothed = np.asarray(values, dtype=np.double).copy()
    for _ in range(passes):
        previous = smoothed.copy()
        smoothed[1:-1, 1:-1] = (
            0.25 * previous[1:-1, 1:-1]
            + 0.125
            * (
                previous[:-2, 1:-1]
                + previous[2:, 1:-1]
                + previous[1:-1, :-2]
                + previous[1:-1, 2:]
            )
            + 0.0625
            * (
                previous[:-2, :-2]
                + previous[:-2, 2:]
                + previous[2:, :-2]
                + previous[2:, 2:]
            )
        )
    return smoothed


def _current_center_slice(bfield, spacing, smoothing_passes=0):
    center_z = bfield.shape[3] // 2
    if center_z == 0 or center_z == bfield.shape[3] - 1:
        raise ValueError("Current-density validation needs at least three z samples.")

    dx, dy, dz = spacing
    bx = bfield[0]
    by = bfield[1]
    bz = bfield[2]

    bx_slice = bx[:, :, center_z]
    bx_zlo = bx[:, :, center_z - 1]
    bx_zhi = bx[:, :, center_z + 1]
    by_slice = by[:, :, center_z]
    by_zlo = by[:, :, center_z - 1]
    by_zhi = by[:, :, center_z + 1]
    bz_slice = bz[:, :, center_z]

    if smoothing_passes > 0:
        bx_slice = _smooth_current_plane(bx_slice, smoothing_passes)
        bx_zlo = _smooth_current_plane(bx_zlo, smoothing_passes)
        bx_zhi = _smooth_current_plane(bx_zhi, smoothing_passes)
        by_slice = _smooth_current_plane(by_slice, smoothing_passes)
        by_zlo = _smooth_current_plane(by_zlo, smoothing_passes)
        by_zhi = _smooth_current_plane(by_zhi, smoothing_passes)
        bz_slice = _smooth_current_plane(bz_slice, smoothing_passes)

    jx = np.gradient(bz_slice, dy, axis=1, edge_order=2) - (by_zhi - by_zlo) / (2.0 * dz)
    jy = (bx_zhi - bx_zlo) / (2.0 * dz) - np.gradient(
        bz_slice,
        dx,
        axis=0,
        edge_order=2,
    )
    jz = np.gradient(by_slice, dx, axis=0, edge_order=2) - np.gradient(
        bx_slice,
        dy,
        axis=1,
        edge_order=2,
    )

    return center_z, np.sqrt(jx * jx + jy * jy + jz * jz)


def _current_smoothness_metric(jmag_slice, spacing):
    dx, dy = spacing[:2]
    grad_x = np.gradient(jmag_slice, dx, axis=0, edge_order=2)
    grad_y = np.gradient(jmag_slice, dy, axis=1, edge_order=2)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    numerator = np.percentile(grad_mag, 99.0)
    denominator = np.median(grad_mag) + HEAVY_AMRVAC_CURRENT_EPSILON
    return float(numerator / denominator)


def _write_grayscale_png(path, image, text_chunks=None):
    text_chunks = text_chunks or {}
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape

    def chunk(chunk_type, payload):
        return (
            pack(">I", len(payload))
            + chunk_type
            + payload
            + pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    for key, value in text_chunks.items():
        payload += chunk(b"tEXt", key.encode("latin-1") + b"\x00" + str(value).encode("latin-1", errors="replace"))
    payload += chunk(b"IDAT", zlib.compress(raw, level=9))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_current_slice_png(path, jmag_slice, fixture_path, slice_index, metric, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    title = (
        f"{fixture_path.name} center z={slice_index} "
        f"smoothness={metric:.6g} threshold={threshold:.6g}"
    )
    try:
        (path.parent / ".cache").mkdir(exist_ok=True)
        (path.parent / ".matplotlib-cache").mkdir(exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(path.parent / ".cache"))
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        finite = np.asarray(jmag_slice[np.isfinite(jmag_slice)])
        if finite.size == 0:
            image = np.zeros(jmag_slice.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if hi <= lo:
                hi = lo + 1.0
            image = np.clip((jmag_slice - lo) / (hi - lo), 0.0, 1.0)
            image = (255.0 * image).astype(np.uint8)
        _write_grayscale_png(
            path,
            image.T,
            text_chunks={
                "Title": title,
                "Fixture": str(fixture_path),
                "Smoothness": f"{metric:.12g}",
                "Threshold": f"{threshold:.12g}",
            },
        )
        return

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(jmag_slice.T, origin="lower", cmap="magma", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("|J|")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_current_diagnostic_pngs(report_dir, jmag_slice, slice_index, metric):
    try:
        (report_dir / ".cache").mkdir(exist_ok=True)
        (report_dir / ".matplotlib-cache").mkdir(exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(report_dir / ".cache"))
        os.environ.setdefault("MPLCONFIGDIR", str(report_dir / ".matplotlib-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    log_path = report_dir / "current_center_z_log.png"
    clipped_path = report_dir / "current_center_z_p95_clip.png"

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(np.log10(jmag_slice.T + 1e-12), origin="lower", cmap="magma", aspect="equal")
    ax.set_title(f"log10 |J| center z={slice_index}, smoothness={metric:.6g}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(image, ax=ax, label="log10(|J|)")
    fig.savefig(log_path, dpi=150)
    plt.close(fig)

    vmax = np.percentile(jmag_slice, 95.0)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(jmag_slice.T, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=vmax)
    ax.set_title(f"|J| center z={slice_index}, clipped at p95={vmax:.6g}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(image, ax=ax, label="|J|, p95 clipped")
    fig.savefig(clipped_path, dpi=150)
    plt.close(fig)

    return [log_path, clipped_path]


def _write_current_report(
    path,
    fixture_path,
    resolution,
    slice_index,
    smoothing_passes,
    metric,
    threshold,
    passed,
    png_path,
    diagnostic_paths,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    relative_png = png_path.name
    diagnostic_lines = "\n".join(f"- `{diagnostic_path.name}`" for diagnostic_path in diagnostic_paths)
    if not diagnostic_lines:
        diagnostic_lines = "- no additional diagnostic PNGs were generated"
    result = "PASS" if passed else "FAIL"
    content = f"""# AMRVAC Current Validation

- fixture path: `{fixture_path}`
- enable gate: `SIMESH_RUN_HEAVY_TESTS=1`
- validation pipeline: open with `ghost_width=2`, load fields `[4, 5, 6]`, exchange ghost cells, sample `b1`, `b2`, and `b3` with linear interpolation, compute `J = curl(B)`, and measure the center `z` slice of `|J|`
- uniform-grid resolution: `{tuple(int(value) for value in resolution)}`
- selected slice: center `z` index `{slice_index}`
- artifact-suppression smoothing passes: `{smoothing_passes}`
- metric formula: `p99(|grad |J||) / (median(|grad |J||) + epsilon)`
- measured metric value: `{metric:.12g}`
- threshold: `{threshold:.12g}`
- result: `{result}`
- embedded PNG path: `{relative_png}`
- additional diagnostic PNGs:
{diagnostic_lines}

![current center z]({relative_png})
"""
    path.write_text(content, encoding="utf-8")


def test_heavy_amrvac_current_validation():
    if os.environ.get("SIMESH_RUN_HEAVY_TESTS") != "1":
        print("Skipping heavy AMRVAC current validation: set SIMESH_RUN_HEAVY_TESTS=1 to enable.")
        return

    fixture_path = HEAVY_AMRVAC_CURRENT_FIXTURE
    if not fixture_path.exists():
        print(f"Skipping heavy AMRVAC current validation: fixture not found at {fixture_path}.")
        return

    ds = open_dataset(str(fixture_path), ghost_width=2)
    ds.load_data(field_indices=[4, 5, 6])
    ds.exchange_ghost_cells()

    resolution = ds.domain_nx.astype(np.int64) * (2 ** (int(ds.levmax) - 1))
    bfield = ds.uniform_grid(resolution, field_indices=[4, 5, 6], interpolation="linear")
    spacing = (ds.physical_domain[1] - ds.physical_domain[0]) / (resolution.astype(np.double) - 1.0)
    slice_index, jmag_slice = _current_center_slice(
        bfield,
        spacing,
        smoothing_passes=HEAVY_AMRVAC_CURRENT_SMOOTHING_PASSES,
    )
    metric = _current_smoothness_metric(jmag_slice, spacing)
    passed = metric <= HEAVY_AMRVAC_CURRENT_THRESHOLD

    png_path = HEAVY_AMRVAC_CURRENT_REPORT_DIR / "current_center_z.png"
    report_path = HEAVY_AMRVAC_CURRENT_REPORT_DIR / "report.md"
    _write_current_slice_png(
        png_path,
        jmag_slice,
        fixture_path,
        slice_index,
        metric,
        HEAVY_AMRVAC_CURRENT_THRESHOLD,
    )
    diagnostic_paths = _write_current_diagnostic_pngs(
        HEAVY_AMRVAC_CURRENT_REPORT_DIR,
        jmag_slice,
        slice_index,
        metric,
    )
    _write_current_report(
        report_path,
        fixture_path,
        resolution,
        slice_index,
        HEAVY_AMRVAC_CURRENT_SMOOTHING_PASSES,
        metric,
        HEAVY_AMRVAC_CURRENT_THRESHOLD,
        passed,
        png_path,
        diagnostic_paths,
    )

    assert passed, (
        "Heavy AMRVAC current validation failed: "
        f"smoothness metric {metric:.12g} exceeds threshold {HEAVY_AMRVAC_CURRENT_THRESHOLD:.12g}. "
        f"See {report_path}."
    )


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


def test_boundary_condition_normalization_api():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 4), nw=3)
        write_datfile_from_uniform(
            path,
            udata,
            ["rho", "m1", "e"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = open_dataset(path, ghost_width=1, boundary_conditions="symm")
        ds.load_data(field_indices=[2, 0])
        assert np.array_equal(ds.mesh.boundary_conditions, np.full((2, 6), 1, dtype=np.int32))

        ds = open_dataset(path, ghost_width=1, boundary_conditions={"e": "asymm", "rho": "cont"})
        ds.load_data(field_indices=[2, 0])
        expected = np.array([[2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]], dtype=np.int32)
        assert np.array_equal(ds.mesh.boundary_conditions, expected)

        ds = open_dataset(path, ghost_width=1, boundary_conditions={"e": {"xlo": "asymm", "zhi": "symm"}})
        ds.load_data(field_indices=[2, 0])
        expected = np.zeros((2, 6), dtype=np.int32)
        expected[0, 0] = 2
        expected[0, 5] = 1
        assert np.array_equal(ds.mesh.boundary_conditions, expected)

        ds = open_dataset(path, ghost_width=1, boundary_conditions={"rho": {"xlo": "noinflow"}})
        ds.load_data(field_indices=[0, 1])
        assert np.array_equal(ds.mesh.boundary_conditions[0], np.array([3, 0, 0, 0, 0, 0], dtype=np.int32))
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_boundary_condition_validation_api():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 1), nw=2)
        write_datfile_from_uniform(
            path,
            udata,
            ["rho", "m1"],
            xmin=np.array([0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2], dtype=np.int32),
            overwrite=True,
        )

        cases = [
            ({"rho": "bad"}, "Unknown boundary condition mode"),
            ({"rho": {"zlo": "cont"}}, "Invalid boundary side"),
            ({"missing": "cont"}, "Unknown boundary condition field"),
            (np.zeros((2, 4), dtype=np.int32), "must have shape"),
            ({"rho": {"xlo": "noinflow"}}, "requires a loaded normal velocity"),
        ]

        for boundary_conditions, message in cases:
            try:
                read_blocks(
                    path,
                    field_indices=[0],
                    ghost_width=1,
                    include_ghosts=True,
                    boundary_conditions=boundary_conditions,
                )
            except ValueError as exc:
                assert message in str(exc)
            else:
                raise AssertionError(f"Expected ValueError containing {message!r}")
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
    test_boundary_condition_normalization_api()
    print("test_boundary_condition_normalization_api passed")
    test_boundary_condition_validation_api()
    print("test_boundary_condition_validation_api passed")
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
    test_heavy_amrvac_current_validation()
    print("test_heavy_amrvac_current_validation passed or skipped")
    print("All tests passed for AMRVAC dataset!")


if __name__ == "__main__":
    run_tests()
