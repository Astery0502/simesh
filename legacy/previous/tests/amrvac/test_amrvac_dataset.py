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


def _linear_field_input(domain_nx=(4, 4, 4)):
    domain_nx = tuple(domain_nx)
    udata = np.zeros((*domain_nx, 3), dtype=np.double)
    spacing = np.array([1.0 / domain_nx[0], 1.0 / domain_nx[1], 1.0 / domain_nx[2]], dtype=np.double)

    for ix, iy, iz in np.ndindex(*domain_nx):
        x = (ix + 0.5) * spacing[0]
        y = (iy + 0.5) * spacing[1]
        z = (iz + 0.5) * spacing[2]
        udata[ix, iy, iz, 0] = x
        udata[ix, iy, iz, 1] = 2.0 * y
        udata[ix, iy, iz, 2] = 5.0 * z

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


def test_dataset_loaded_field_names_track_loaded_columns():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 4), nw=4)
        write_datfile_from_uniform(
            path,
            udata,
            ["rho", "m1", "m2", "e"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path)
        assert ds.loaded_field_indices == [0, 1, 2, 3]
        assert ds.loaded_field_names == ["rho", "m1", "m2", "e"]
        assert [(column.name, column.source_kind, column.original_index) for column in ds._field_columns] == [
            ("rho", "original", 0),
            ("m1", "original", 1),
            ("m2", "original", 2),
            ("e", "original", 3),
        ]
        assert ds.derived_definitions == {}
        assert ds.derived_field_names == []

        ds.load_data(field_indices=[2, 0])
        assert ds.loaded_field_indices == [2, 0]
        assert ds.loaded_field_names == ["m2", "rho"]
        assert [(column.name, column.source_kind, column.original_index) for column in ds._field_columns] == [
            ("m2", "original", 2),
            ("rho", "original", 0),
        ]
        assert ds._loaded_field_map() == {2: 0, 0: 1}
        assert ds._loaded_field_name_map() == {"m2": 0, "rho": 1}
        assert ds._columns_for_field_names(["rho", "m2"]) == [1, 0]

        try:
            ds._columns_for_field_names(["e"])
        except ValueError as exc:
            assert "not loaded" in str(exc)
            assert "e" in str(exc)
        else:
            raise AssertionError("missing loaded field names should fail")

        try:
            ds._validate_field_selectors(field_indices=[0], field_names=["rho"])
        except ValueError as exc:
            assert "cannot both be supplied" in str(exc)
        else:
            raise AssertionError("mixed field selectors should fail")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_registers_derived_recipes_without_materializing():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    original_shape = ds.data.shape

    def pressure(_ctx):
        raise AssertionError("register_derived should not compute fields")

    ds.register_derived("p", pressure, dependencies=["rho", "e"])

    definition = ds.derived_definitions["p"]
    assert definition.func is pressure
    assert definition.dependencies == ("rho", "e")
    assert definition.requires_ghosts is False
    assert ds.derived_field_names == []
    assert ds.data.shape == original_shape
    assert ds.loaded_field_names == ["rho", "m1", "e"]


def test_dataset_registers_derived_recipe_replacements():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )

    def first(_ctx):
        return None

    def second(_ctx):
        return None

    ds.register_derived("p", first, dependencies=["rho", "e"])
    ds.register_derived("p", second, dependencies=["rho"], requires_ghosts=True)

    definition = ds.derived_definitions["p"]
    assert definition.func is second
    assert definition.dependencies == ("rho",)
    assert definition.requires_ghosts is True
    assert ds.derived_field_names == []


def test_dataset_rejects_invalid_derived_recipe_registration():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )

    def recipe(_ctx):
        return None

    invalid_cases = [
        ("", recipe, ["rho"], "non-empty string"),
        (3, recipe, ["rho"], "non-empty string"),
        ("rho", recipe, ["rho"], "collides"),
        ("p", None, ["rho"], "callable"),
        ("p", recipe, None, "dependencies"),
        ("p", recipe, "rho", "dependencies"),
        ("p", recipe, [0], "field names"),
        ("p", recipe, [""], "field names"),
    ]

    for name, func, dependencies, expected_message in invalid_cases:
        try:
            ds.register_derived(name, func, dependencies=dependencies)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Invalid derived registration should fail: {name!r}")

    assert ds.derived_definitions == {}
    assert ds.derived_field_names == []


def test_dataset_registers_derivative_recipes_without_materializing():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["b1", "b2", "b3"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    original_shape = ds.data.shape

    ds.register_derivative(
        "j1",
        [
            ("b3", "y", 1),
            ("b2", "z", -1),
        ],
    )

    definition = ds.derived_definitions["j1"]
    assert definition.dependencies == ("b3", "b2")
    assert definition.requires_ghosts is True
    assert [(term.field_name, term.axis, term.coefficient) for term in definition.terms] == [
        ("b3", 1, 1.0),
        ("b2", 2, -1.0),
    ]
    assert ds.derived_field_names == []
    assert ds.data.shape == original_shape


def test_dataset_rejects_invalid_derivative_recipe_registration():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )

    invalid_cases = [
        ("", [("rho", "x", 1.0)], "non-empty string"),
        (3, [("rho", "x", 1.0)], "non-empty string"),
        ("rho", [("rho", "x", 1.0)], "collides"),
        ("drho", None, "terms"),
        ("drho", [], "terms"),
        ("drho", [("rho", "x")], "field_name, axis, coefficient"),
        ("drho", [("", "x", 1.0)], "field names"),
        ("drho", [("rho", "q", 1.0)], "invalid axis"),
        ("drho", [("rho", 3, 1.0)], "invalid axis"),
        ("drho", [("rho", "x", object())], "float64"),
    ]

    for name, terms, expected_message in invalid_cases:
        try:
            ds.register_derivative(name, terms)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"invalid derivative recipe should fail: {name!r}")


def test_dataset_materializes_arithmetic_derived_field():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )

    expected = ds.data[:, 2, :, :, :] - 0.5 * ds.data[:, 0, :, :, :]
    ds.register_derived(
        "p",
        lambda ctx: ctx.field("e") - 0.5 * ctx.field("rho"),
        dependencies=["rho", "e"],
    )
    ds.materialize_fields(["p"])

    assert ds.loaded_field_indices == [0, 1, 2]
    assert ds.loaded_field_names == ["rho", "m1", "e", "p"]
    assert ds.derived_field_names == ["p"]
    assert [(column.name, column.source_kind, column.original_index, column.ghost_valid_layers) for column in ds._field_columns] == [
        ("rho", "original", 0, 0),
        ("m1", "original", 1, 0),
        ("e", "original", 2, 0),
        ("p", "derived", None, 0),
    ]
    assert ds.data.shape == (8, 4, 2, 2, 2)
    assert np.array_equal(ds.data[:, 3, :, :, :], expected)

    ds.materialize_fields(["p"])
    assert ds.loaded_field_names == ["rho", "m1", "e", "p"]
    assert ds.data.shape == (8, 4, 2, 2, 2)


def test_dataset_materializes_single_derivative_from_cython_backend():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["v1", "b2", "b3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1, 2])
        ds.register_derivative("dvx_dx", [("v1", "x", 1.0)])
        ds.materialize_fields(["dvx_dx"])

        assert ds.loaded_field_names == ["v1", "b2", "b3", "dvx_dx"]
        assert ds.derived_field_names == ["dvx_dx"]
        assert ds.derived_field_ghost_valid_layers["dvx_dx"] == 1
        assert [(column.name, column.source_kind, column.original_index, column.ghost_valid_layers) for column in ds._field_columns] == [
            ("v1", "original", 0, 2),
            ("b2", "original", 1, 2),
            ("b3", "original", 2, 2),
            ("dvx_dx", "derived", None, 1),
        ]
        derivative_grid = ds.uniform_full(field_names=["dvx_dx"])[0]
        assert np.allclose(derivative_grid[1:-1, :, :], 1.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_materializes_requested_current_component_only():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["b1", "b2", "b3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1, 2])
        ds.register_derivative("j1", [("b3", "y", 1.0), ("b2", "z", -1.0)])
        ds.register_derivative("j2", [("b1", "z", 1.0), ("b3", "x", -1.0)])
        ds.register_derivative("j3", [("b2", "x", 1.0), ("b1", "y", -1.0)])

        ds.materialize_fields(["j1"])

        assert ds.loaded_field_names == ["b1", "b2", "b3", "j1"]
        assert ds.derived_field_names == ["j1"]
        assert "j2" not in ds.loaded_field_names
        assert "j3" not in ds.loaded_field_names
        assert np.allclose(ds.uniform_full(field_names=["j1"])[0], 0.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_derivative_materialization_uses_existing_ghost_exchange():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["b1", "b2", "b3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1, 2])
        original_exchange = ds.exchange_ghost_cells
        exchange_count = {"value": 0}

        def counted_exchange():
            exchange_count["value"] += 1
            original_exchange()

        ds.exchange_ghost_cells = counted_exchange
        ds.register_derivative("db1_dx", [("b1", "x", 1.0)])
        ds.register_derivative("db2_dy", [("b2", "y", 1.0)])
        ds.materialize_fields(["db1_dx", "db2_dy"])

        assert exchange_count["value"] == 0
        assert ds.loaded_field_names == ["b1", "b2", "b3", "db1_dx", "db2_dy"]
        assert np.allclose(ds.uniform_full(field_names=["db1_dx"])[0, 1:-1, :, :], 1.0)
        assert np.allclose(ds.uniform_full(field_names=["db2_dy"])[0, :, 1:-1, :], 2.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_rejects_ghost_width_one_during_construction():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["v1", "b2", "b3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        try:
            AMRVACDataSet(path, ghost_width=1, boundary_conditions="cont")
        except ValueError as exc:
            assert "ghost_width must be 0 or >= 2" in str(exc)
        else:
            raise AssertionError("ghost_width=1 should be rejected during construction")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_rejects_derivative_materialization_with_missing_dependency():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=2),
        ["rho", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    ds.register_derivative("db1_dx", [("b1", "x", 1.0)])

    try:
        ds.materialize_fields(["db1_dx"])
    except ValueError as exc:
        assert "dependencies are not loaded" in str(exc)
        assert "b1" in str(exc)
    else:
        raise AssertionError("missing derivative dependencies should fail")


def test_derivative_padded_output_leaves_outermost_ghost_layer_zero():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["v1", "b2", "b3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1, 2])
        ds.register_derivative("dvx_dx", [("v1", "x", 1.0)])
        ds.materialize_fields(["dvx_dx"])

        ghosted = ds.blocks(include_ghosts=True, field_names=["dvx_dx"])[:, 0]
        assert np.isclose(ghosted.max(), 1.0)
        assert np.all(ghosted[:, 0, :, :] == 0.0)
        assert np.all(ghosted[:, -1, :, :] == 0.0)
        assert np.all(ghosted[:, :, 0, :] == 0.0)
        assert np.all(ghosted[:, :, -1, :] == 0.0)
        assert np.all(ghosted[:, :, :, 0] == 0.0)
        assert np.all(ghosted[:, :, :, -1] == 0.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_ghost_exchange_does_not_fill_materialized_derived_fields():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["v1", "b2", "b3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1, 2])
        ds.register_derived("p", lambda ctx: ctx.field("v1") + 1.0, dependencies=["v1"])
        ds.register_derivative("dvx_dx", [("v1", "x", 1.0)])
        ds.materialize_fields(["p", "dvx_dx"])

        ds.exchange_ghost_cells()

        p_ghosted = ds.blocks(include_ghosts=True, field_names=["p"])[:, 0]
        derivative_ghosted = ds.blocks(include_ghosts=True, field_names=["dvx_dx"])[:, 0]

        assert np.all(p_ghosted[:, 0, :, :] == 0.0)
        assert np.all(p_ghosted[:, -1, :, :] == 0.0)
        assert np.all(p_ghosted[:, :, 0, :] == 0.0)
        assert np.all(p_ghosted[:, :, -1, :] == 0.0)
        assert np.all(p_ghosted[:, :, :, 0] == 0.0)
        assert np.all(p_ghosted[:, :, :, -1] == 0.0)
        assert np.all(derivative_ghosted[:, 0, :, :] == 0.0)
        assert np.all(derivative_ghosted[:, -1, :, :] == 0.0)
        assert np.all(derivative_ghosted[:, :, 0, :] == 0.0)
        assert np.all(derivative_ghosted[:, :, -1, :] == 0.0)
        assert np.all(derivative_ghosted[:, :, :, 0] == 0.0)
        assert np.all(derivative_ghosted[:, :, :, -1] == 0.0)
        assert ds.derived_field_ghost_valid_layers == {"p": 0, "dvx_dx": 1}
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_rejects_ghost_required_materialized_dependencies():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["rho", "m2", "m3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1, 2])
        ds.register_derived("rho2", lambda ctx: 2.0 * ctx.field("rho"), dependencies=["rho"])
        ds.materialize_fields(["rho2"])
        ds.register_derivative("drho2_dx", [("rho2", "x", 1.0)])

        try:
            ds.materialize_fields(["drho2_dx"])
        except ValueError as exc:
            assert "only original loaded fields" in str(exc)
            assert "rho2" in str(exc)
        else:
            raise AssertionError("derivatives of materialized fields should fail without a ghost exchange contract")

        ds.register_derived(
            "rho2_xlo",
            lambda ctx: ctx.padded_field("rho2")[:, 0:2, 1:3, 1:3],
            dependencies=["rho2"],
            requires_ghosts=True,
        )
        try:
            ds.materialize_fields(["rho2_xlo"])
        except ValueError as exc:
            assert "only original loaded fields" in str(exc)
            assert "rho2" in str(exc)
        else:
            raise AssertionError("ghost-required recipes should fail for materialized dependencies")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_ghost_required_python_derived_uses_existing_ghost_exchange():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _linear_field_input(domain_nx=(4, 4, 4)),
            ["rho", "m2", "m3"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0, 1])
        original_exchange = ds.exchange_ghost_cells
        exchange_count = {"value": 0}

        def counted_exchange():
            exchange_count["value"] += 1
            original_exchange()

        ds.exchange_ghost_cells = counted_exchange
        ds.register_derived(
            "rho_xlo",
            lambda ctx: ctx.padded_field("rho")[:, 0:2, 1:3, 1:3],
            dependencies=["rho"],
            requires_ghosts=True,
        )
        ds.register_derived(
            "m2_xlo",
            lambda ctx: ctx.padded_field("m2")[:, 0:2, 1:3, 1:3],
            dependencies=["m2"],
            requires_ghosts=True,
        )
        ds.materialize_fields(["rho_xlo", "m2_xlo"])

        assert exchange_count["value"] == 0
        assert ds.loaded_field_names == ["rho", "m2", "rho_xlo", "m2_xlo"]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_padded_field_rejects_materialized_derived_fields():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        write_datfile_from_uniform(
            path,
            _uniform_input(domain_nx=(4, 4, 4), nw=1),
            ["rho"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0])
        ds.register_derived("rho2", lambda ctx: 2.0 * ctx.field("rho"), dependencies=["rho"])
        ds.materialize_fields(["rho2"])

        try:
            ds._derived_context().padded_field("rho2")
        except ValueError as exc:
            assert "materialized derived field" in str(exc)
            assert "ghost-cell exchange contract" in str(exc)
        else:
            raise AssertionError("padded_field should reject materialized derived fields")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_selects_materialized_fields_by_name_through_blocks():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    expected_p = ds.data[:, 2, :, :, :] - ds.data[:, 0, :, :, :]
    expected_rho = ds.data[:, 0, :, :, :].copy()
    ds.register_derived(
        "p",
        lambda ctx: ctx.field("e") - ctx.field("rho"),
        dependencies=["rho", "e"],
    )
    ds.materialize_fields(["p"])

    selected = ds.blocks(field_names=["p", "rho"])

    assert selected.shape == (8, 2, 2, 2, 2)
    assert np.array_equal(selected[:, 0, :, :, :], expected_p)
    assert np.array_equal(selected[:, 1, :, :, :], expected_rho)


def test_dataset_selects_materialized_fields_by_name_through_uniform_paths():
    udata = _uniform_input(domain_nx=(4, 4, 4), nw=3)
    ds = load_from_uniform(
        udata,
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    expected_p = udata_to_datau(udata[..., 2:3])
    expected_p[0] -= udata[..., 0]
    ds.register_derived(
        "p",
        lambda ctx: ctx.field("e") - ctx.field("rho"),
        dependencies=["rho", "e"],
    )
    ds.materialize_fields(["p"])

    grid = ds.uniform_grid(ds.domain_nx, field_names=["p"])
    full = ds.uniform_full(field_names=["p"])

    assert grid.shape == (1, 4, 4, 4)
    assert full.shape == (1, 4, 4, 4)
    assert np.array_equal(grid, expected_p)
    assert np.array_equal(full, expected_p)


def test_dataset_rejects_mixed_downstream_field_selectors():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=2),
        ["rho", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )

    cases = [
        lambda: ds.blocks(field_indices=[0], field_names=["rho"]),
        lambda: ds.uniform_grid(ds.domain_nx, field_indices=[0], field_names=["rho"]),
        lambda: ds.uniform_full(field_indices=[0], field_names=["rho"]),
    ]
    for call in cases:
        try:
            call()
        except ValueError as exc:
            assert "cannot both be supplied" in str(exc)
        else:
            raise AssertionError("mixed downstream field selectors should fail")


def test_dataset_write_datfile_can_opt_in_to_materialized_field_names():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        ds = load_from_uniform(
            _uniform_input(domain_nx=(4, 4, 4), nw=3),
            ["rho", "m1", "e"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
        )
        expected = ds.data[:, 2, :, :, :] - ds.data[:, 0, :, :, :]
        ds.register_derived(
            "p",
            lambda ctx: ctx.field("e") - ctx.field("rho"),
            dependencies=["rho", "e"],
        )
        ds.materialize_fields(["p"])

        output_header = ds.write_datfile(path, field_names=["p"], overwrite=True)
        rewritten = open_dataset(path)

        assert output_header["nw"] == 1
        assert output_header["w_names"] == ["p"]
        assert rewritten.wnames == ["p"]
        assert np.array_equal(rewritten.blocks(), expected[:, np.newaxis, :, :, :])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_rejects_missing_derived_dependencies():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=3),
        ["rho", "m1", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    ds.data = ds.data[:, [0, 1], :, :, :]
    ds._set_field_columns(ds._original_field_columns([0, 1]))
    ds.register_derived(
        "p",
        lambda ctx: ctx.field("e") - ctx.field("rho"),
        dependencies=["rho", "e"],
    )

    try:
        ds.materialize_fields(["p"])
    except ValueError as exc:
        assert "dependencies are not loaded" in str(exc)
        assert "e" in str(exc)
    else:
        raise AssertionError("missing derived dependencies should fail")

    assert ds.loaded_field_names == ["rho", "m1"]
    assert ds.derived_field_names == []
    assert ds.data.shape == (8, 2, 2, 2, 2)


def test_dataset_rejects_invalid_derived_result_shape():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=2),
        ["rho", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    ds.register_derived(
        "p",
        lambda ctx: ctx.field("e")[:, :-1, :, :],
        dependencies=["e"],
    )

    try:
        ds.materialize_fields(["p"])
    except ValueError as exc:
        assert "returned shape" in str(exc)
        assert "expected" in str(exc)
    else:
        raise AssertionError("derived fields with invalid shape should fail")

    assert ds.loaded_field_names == ["rho", "e"]
    assert ds.derived_field_names == []


def test_dataset_drops_materialized_derived_fields():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=2),
        ["rho", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    original = ds.data.copy()
    ds.register_derived("p", lambda ctx: ctx.field("e") + 1.0, dependencies=["e"])
    ds.register_derived("q", lambda ctx: ctx.field("rho") + 2.0, dependencies=["rho"])
    ds.materialize_fields(["p", "q"])

    ds.drop_derived_fields(["p"])

    assert ds.loaded_field_names == ["rho", "e", "q"]
    assert ds.derived_field_names == ["q"]
    assert np.array_equal(ds.data[:, :2, :, :, :], original)
    assert np.array_equal(ds.data[:, 2, :, :, :], original[:, 0, :, :, :] + 2.0)

    ds.drop_derived_fields("q")
    assert ds.loaded_field_names == ["rho", "e"]
    assert ds.derived_field_names == []
    assert np.array_equal(ds.data, original)


def test_dataset_drops_derived_fields_refreshes_ghost_storage():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 4), nw=1)
        write_datfile_from_uniform(
            path,
            udata,
            ["rho"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0])
        ds.register_derived("p", lambda ctx: ctx.field("rho") + 1.0, dependencies=["rho"])
        ds.materialize_fields(["p"])
        assert ds.blocks(include_ghosts=True).shape == (8, 2, 6, 6, 6)

        ds.drop_derived_fields("p")

        assert ds.loaded_field_names == ["rho"]
        assert ds.derived_field_names == []
        assert [(column.name, column.source_kind, column.original_index) for column in ds._field_columns] == [
            ("rho", "original", 0),
        ]
        assert ds.blocks(include_ghosts=True).shape == (8, 1, 6, 6, 6)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_reloads_clear_materialized_derived_fields():
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

        ds = AMRVACDataSet(path)
        ds.load_data(field_indices=[0, 2])
        ds.register_derived("p", lambda ctx: ctx.field("e") + 1.0, dependencies=["e"])
        ds.materialize_fields(["p"])
        assert ds.loaded_field_names == ["rho", "e", "p"]
        assert ds.derived_field_names == ["p"]

        ds.load_data(field_indices=[1])

        assert ds.loaded_field_indices == [1]
        assert ds.loaded_field_names == ["m1"]
        assert ds.derived_definitions["p"].dependencies == ("e",)
        assert ds.derived_field_names == []
        assert ds.data.shape == (8, 1, 2, 2, 2)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_reregistering_derived_field_drops_materialized_result():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=2),
        ["rho", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    ds.register_derived("p", lambda ctx: ctx.field("e") + 1.0, dependencies=["e"])
    ds.materialize_fields(["p"])

    ds.register_derived("p", lambda ctx: ctx.field("rho") + 2.0, dependencies=["rho"])

    assert ds.loaded_field_names == ["rho", "e"]
    assert ds.derived_field_names == []
    ds.materialize_fields(["p"])
    assert np.array_equal(ds.data[:, 2, :, :, :], ds.data[:, 0, :, :, :] + 2.0)


def test_dataset_rejects_ghost_required_derived_field_without_ghost_storage():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=2),
        ["rho", "e"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )
    ds.register_derived(
        "j1",
        lambda ctx: ctx.field("rho"),
        dependencies=["rho"],
        requires_ghosts=True,
    )

    try:
        ds.materialize_fields(["j1"])
    except ValueError as exc:
        assert "requires ghost cells" in str(exc)
    else:
        raise AssertionError("ghost-required derived fields should require ghost storage")


def test_dataset_materializes_ghost_required_field_from_padded_data():
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        path = tmp.name

    try:
        udata = _uniform_input(domain_nx=(4, 4, 4), nw=2)
        write_datfile_from_uniform(
            path,
            udata,
            ["rho", "e"],
            xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
            xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
            block_nx=np.array([2, 2, 2], dtype=np.int32),
            overwrite=True,
        )

        ds = AMRVACDataSet(path, ghost_width=2, boundary_conditions="cont")
        ds.load_data(field_indices=[0])
        ds.data[0, 0, 0, 0, 0] = -123.0
        ds.exchange_ghost_cells()
        expected = ds._derived_context().padded_field("rho")[:, 1:3, 2:4, 2:4].copy()

        def xlo_ghost(ctx):
            padded = ctx.padded_field("rho")
            assert padded.shape == (8, 6, 6, 6)
            assert ctx.spacing.shape == (8, 3)
            return padded[:, 1:3, 2:4, 2:4]

        ds.register_derived(
            "rho_xlo",
            xlo_ghost,
            dependencies=["rho"],
            requires_ghosts=True,
        )
        ds.materialize_fields(["rho_xlo"])

        assert ds.loaded_field_names == ["rho", "rho_xlo"]
        assert ds.derived_field_names == ["rho_xlo"]
        assert ds.data.shape == (8, 2, 2, 2, 2)
        assert expected[0, 0, 0, 0] == -123.0
        assert np.array_equal(ds.data[:, 1, :, :, :], expected)
        assert ds.blocks(include_ghosts=True).shape == (8, 2, 6, 6, 6)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_dataset_padded_field_requires_ghost_storage():
    ds = load_from_uniform(
        _uniform_input(domain_nx=(4, 4, 4), nw=1),
        ["rho"],
        xmin=np.array([0.0, 0.0, 0.0], dtype=np.double),
        xmax=np.array([1.0, 1.0, 1.0], dtype=np.double),
        block_nx=np.array([2, 2, 2], dtype=np.int32),
    )

    try:
        ds._derived_context().padded_field("rho")
    except ValueError as exc:
        assert "requires ghost cells" in str(exc)
    else:
        raise AssertionError("padded_field should require ghost storage")


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

        ds = AMRVACDataSet(path, ghost_width=2)
        ds.load_data(field_indices=[0, 2])

        padded = ds.mesh.padded_view()
        interior = ds.mesh.interior_view()

        assert padded.shape == (8, 6, 6, 6, 2)
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

        ds = open_dataset(path, ghost_width=2)
        assert ds.has_ghost_cells
        ds.load_data(field_indices=[0, 2])
        assert ds.blocks().shape == (8, 2, 2, 2, 2)
        assert ds.blocks(include_ghosts=True).shape == (8, 2, 6, 6, 6)

        blocks = read_blocks(path, field_indices=[0, 2])
        ghosted = read_blocks(path, field_indices=[0, 2], ghost_width=2, include_ghosts=True)
        grid = read_uniform(path, resolution=(4, 4, 4), field_indices=[1])
        linear_grid = read_uniform(
            path,
            resolution=(4, 4, 4),
            field_indices=[1],
            ghost_width=2,
            interpolation="linear",
        )

        assert blocks.shape == (8, 2, 2, 2, 2)
        assert ghosted.shape == (8, 2, 6, 6, 6)
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

        ds = open_dataset(path, ghost_width=2, boundary_conditions="symm")
        ds.load_data(field_indices=[2, 0])
        assert np.array_equal(ds.mesh.boundary_conditions, np.full((2, 6), 1, dtype=np.int32))

        ds = open_dataset(path, ghost_width=2, boundary_conditions={"e": "asymm", "rho": "cont"})
        ds.load_data(field_indices=[2, 0])
        expected = np.array([[2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]], dtype=np.int32)
        assert np.array_equal(ds.mesh.boundary_conditions, expected)

        ds = open_dataset(path, ghost_width=2, boundary_conditions={"e": {"xlo": "asymm", "zhi": "symm"}})
        ds.load_data(field_indices=[2, 0])
        expected = np.zeros((2, 6), dtype=np.int32)
        expected[0, 0] = 2
        expected[0, 5] = 1
        assert np.array_equal(ds.mesh.boundary_conditions, expected)

        ds = open_dataset(path, ghost_width=2, boundary_conditions={"rho": {"xlo": "noinflow"}})
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
                    ghost_width=2,
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

        ds = open_dataset(path, ghost_width=2)
        assert int(ds.ndim) == 2
        assert np.array_equal(ds.domain_nx, np.array([4, 4], dtype=np.uint32))
        assert np.array_equal(ds.block_nx, np.array([2, 2], dtype=np.uint32))

        ds.load_data(field_indices=[0, 2])
        assert ds.blocks().shape == (4, 2, 2, 2, 1)
        assert ds.blocks(include_ghosts=True).shape == (4, 2, 6, 6, 5)

        blocks = read_blocks(path, field_indices=[0, 2])
        ghosted = read_blocks(path, field_indices=[0, 2], ghost_width=2, include_ghosts=True)
        grid = read_uniform(path, resolution=(4, 4), field_indices=[1])
        linear_grid = read_uniform(
            path,
            resolution=(4, 4),
            field_indices=[1],
            ghost_width=2,
            interpolation="linear",
        )

        assert blocks.shape == (4, 2, 2, 2, 1)
        assert ghosted.shape == (4, 2, 6, 6, 5)
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
    tests = [
        test_dataset_uniform_full,
        test_dataset_loaded_field_names_track_loaded_columns,
        test_dataset_registers_derived_recipes_without_materializing,
        test_dataset_registers_derived_recipe_replacements,
        test_dataset_rejects_invalid_derived_recipe_registration,
        test_dataset_registers_derivative_recipes_without_materializing,
        test_dataset_rejects_invalid_derivative_recipe_registration,
        test_dataset_materializes_arithmetic_derived_field,
        test_dataset_materializes_single_derivative_from_cython_backend,
        test_dataset_materializes_requested_current_component_only,
        test_dataset_derivative_materialization_uses_existing_ghost_exchange,
        test_dataset_rejects_ghost_width_one_during_construction,
        test_dataset_rejects_derivative_materialization_with_missing_dependency,
        test_derivative_padded_output_leaves_outermost_ghost_layer_zero,
        test_ghost_exchange_does_not_fill_materialized_derived_fields,
        test_dataset_rejects_ghost_required_materialized_dependencies,
        test_dataset_ghost_required_python_derived_uses_existing_ghost_exchange,
        test_dataset_padded_field_rejects_materialized_derived_fields,
        test_dataset_selects_materialized_fields_by_name_through_blocks,
        test_dataset_selects_materialized_fields_by_name_through_uniform_paths,
        test_dataset_rejects_mixed_downstream_field_selectors,
        test_dataset_write_datfile_can_opt_in_to_materialized_field_names,
        test_dataset_rejects_missing_derived_dependencies,
        test_dataset_rejects_invalid_derived_result_shape,
        test_dataset_drops_materialized_derived_fields,
        test_dataset_drops_derived_fields_refreshes_ghost_storage,
        test_dataset_reloads_clear_materialized_derived_fields,
        test_dataset_reregistering_derived_field_drops_materialized_result,
        test_dataset_rejects_ghost_required_derived_field_without_ghost_storage,
        test_dataset_materializes_ghost_required_field_from_padded_data,
        test_dataset_padded_field_requires_ghost_storage,
        test_dataset_ghost_mode_uses_mesh_storage,
        test_public_api_reads_blocks_and_uniform_data,
        test_boundary_condition_normalization_api,
        test_boundary_condition_validation_api,
        test_public_api_2d_singleton_z_roundtrip,
        test_extract_uniform_data,
        test_uniform_layout_helpers,
        test_load_uniform_data,
        test_load_from_uniform_2d_and_3d,
        test_dataset_getitem_returns_udata_layout,
        test_datfile_to_vtk,
        test_datfile_to_vtk_2d_singleton_z,
        test_heavy_amrvac_current_validation,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} passed")
    print("All tests passed for AMRVAC dataset!")


if __name__ == "__main__":
    run_tests()
