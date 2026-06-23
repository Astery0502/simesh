import os
import zlib
from pathlib import Path
from struct import pack

import numpy as np

from simesh.amrvac import open_dataset


FIXTURE_PATH = Path("data/weno509_sub_0000.dat")
REPORT_DIR = Path("report/amrvac-derived-current-diagnostic")
MAGNETIC_FIELD_INDICES = [4, 5, 6]
CURRENT_FIELD_NAMES = ["j1", "j2", "j3"]


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


def _field_stats(values):
    finite = np.asarray(values[np.isfinite(values)], dtype=np.double)
    if finite.size == 0:
        return {
            "finite_count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "p95": np.nan,
        }
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _scaled_image(values):
    values = np.asarray(values, dtype=np.double)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    image = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return (255.0 * image).astype(np.uint8)


def _slice_fields(grid, axis, index):
    if axis == "x":
        return {name: grid[position, index, :, :] for position, name in enumerate(CURRENT_FIELD_NAMES)}
    if axis == "y":
        return {name: grid[position, :, index, :] for position, name in enumerate(CURRENT_FIELD_NAMES)}
    if axis == "z":
        return {name: grid[position, :, :, index] for position, name in enumerate(CURRENT_FIELD_NAMES)}
    raise ValueError(f"Unknown slice axis: {axis}")


def _render_slice_png(path, fields, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_names = ["j1", "j2", "j3", "|J|", "|J|/|B|"]
    try:
        (path.parent / ".cache").mkdir(exist_ok=True)
        (path.parent / ".matplotlib-cache").mkdir(exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(path.parent / ".cache"))
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        panels = [_scaled_image(fields[name]).T for name in panel_names]
        separator = np.full((panels[0].shape[0], 4), 255, dtype=np.uint8)
        image = panels[0]
        for panel in panels[1:]:
            image = np.hstack((image, separator, panel))
        _write_grayscale_png(path, image, text_chunks={"Title": title})
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, name in zip(axes, panel_names):
        values = fields[name]
        finite = values[np.isfinite(values)]
        kwargs = {}
        if name in {"|J|", "|J|/|B|"} and finite.size > 0:
            kwargs["vmin"] = 0.0
            kwargs["vmax"] = np.percentile(finite, 99.0)
        image = ax.imshow(values.T, origin="lower", cmap="magma", aspect="equal", **kwargs)
        ax.set_title(name)
        ax.set_xlabel("index")
        ax.set_ylabel("index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[-1].axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_report(path, fixture_path, ds, resolution, slice_indices, stats_by_slice, image_paths):
    stats_lines = []
    for slice_name, field_stats in stats_by_slice.items():
        stats_lines.append(f"## {slice_name}")
        stats_lines.append("")
        stats_lines.append("| field | finite count | min | max | mean | p95 |")
        stats_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for field_name, stats in field_stats.items():
            stats_lines.append(
                f"| `{field_name}` | {stats['finite_count']} | "
                f"{stats['min']:.12g} | {stats['max']:.12g} | "
                f"{stats['mean']:.12g} | {stats['p95']:.12g} |"
            )
        stats_lines.append("")

    image_lines = "\n".join(f"![{path.stem}]({path.name})" for path in image_paths)
    content = f"""# AMRVAC Derived Current Diagnostic

- fixture path: `{fixture_path}`
- enable gate: `SIMESH_RUN_HEAVY_TESTS=1`
- command: `SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src .venv/bin/python tests/amrvac/test_amrvac_current_derived_optional.py`
- validation pipeline: open with `ghost_width=2`, load fields `{MAGNETIC_FIELD_INDICES}`, register derivative fields `j1`, `j2`, and `j3`, materialize them, sample derived fields by `field_names`, and render central slices
- loaded field names: `{ds.loaded_field_names}`
- materialized derived fields: `{ds.derived_field_names}`
- uniform-grid resolution: `{tuple(int(value) for value in resolution)}`
- central slice indices: `{slice_indices}`
- output images: `{[path.name for path in image_paths]}`

{chr(10).join(stats_lines)}
{image_lines}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _register_current_derivatives(ds):
    ds.register_derivative("j1", [("b3", "y", 1.0), ("b2", "z", -1.0)])
    ds.register_derivative("j2", [("b1", "z", 1.0), ("b3", "x", -1.0)])
    ds.register_derivative("j3", [("b2", "x", 1.0), ("b1", "y", -1.0)])


def test_heavy_amrvac_derived_current_report():
    if os.environ.get("SIMESH_RUN_HEAVY_TESTS") != "1":
        print("Skipping heavy AMRVAC derived-current report: set SIMESH_RUN_HEAVY_TESTS=1 to enable.")
        return

    if not FIXTURE_PATH.exists():
        print(f"Skipping heavy AMRVAC derived-current report: fixture not found at {FIXTURE_PATH}.")
        return

    ds = open_dataset(str(FIXTURE_PATH), ghost_width=2)
    ds.load_data(field_indices=MAGNETIC_FIELD_INDICES)
    if ds.loaded_field_names != ["b1", "b2", "b3"]:
        raise AssertionError(
            "Expected magnetic fields [4, 5, 6] to load as ['b1', 'b2', 'b3']; "
            f"got {ds.loaded_field_names}."
        )

    _register_current_derivatives(ds)
    ds.materialize_fields(CURRENT_FIELD_NAMES)

    resolution = ds.domain_nx.astype(np.int64) * (2 ** (int(ds.levmax) - 1))
    current_grid = ds.uniform_grid(
        resolution,
        field_names=CURRENT_FIELD_NAMES,
        interpolation="linear",
    )
    magnetic_grid = ds.uniform_grid(
        resolution,
        field_names=["b1", "b2", "b3"],
        interpolation="linear",
    )
    jmag = np.sqrt(np.sum(current_grid * current_grid, axis=0))
    bmag = np.sqrt(np.sum(magnetic_grid * magnetic_grid, axis=0))
    ratio = np.divide(jmag, bmag, out=np.zeros_like(jmag), where=bmag > 0.0)

    if not np.all(np.isfinite(jmag)):
        raise AssertionError("Derived |J| contains non-finite values.")
    if not np.any(jmag > 0.0):
        raise AssertionError("Derived |J| is identically zero.")
    if not np.all(np.isfinite(ratio)):
        raise AssertionError("Derived |J|/|B| contains non-finite values.")

    slice_indices = {
        "mid_x": int(resolution[0] // 2),
        "mid_y": int(resolution[1] // 2),
        "mid_z": int(resolution[2] // 2),
    }
    slice_axes = {
        "mid_x": "x",
        "mid_y": "y",
        "mid_z": "z",
    }
    image_paths = []
    stats_by_slice = {}
    for slice_name, axis in slice_axes.items():
        index = slice_indices[slice_name]
        fields = _slice_fields(current_grid, axis, index)
        if axis == "x":
            fields["|J|"] = jmag[index, :, :]
            fields["|J|/|B|"] = ratio[index, :, :]
        elif axis == "y":
            fields["|J|"] = jmag[:, index, :]
            fields["|J|/|B|"] = ratio[:, index, :]
        else:
            fields["|J|"] = jmag[:, :, index]
            fields["|J|/|B|"] = ratio[:, :, index]

        stats_by_slice[slice_name] = {
            field_name: _field_stats(values)
            for field_name, values in fields.items()
        }
        image_path = REPORT_DIR / f"current_{slice_name}.png"
        _render_slice_png(
            image_path,
            fields,
            f"{FIXTURE_PATH.name} derived current {slice_name} index={index}",
        )
        image_paths.append(image_path)

    _write_report(
        REPORT_DIR / "report.md",
        FIXTURE_PATH,
        ds,
        resolution,
        slice_indices,
        stats_by_slice,
        image_paths,
    )


if __name__ == "__main__":
    test_heavy_amrvac_derived_current_report()
