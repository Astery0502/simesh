"""Recompute thermal diagnostics from a snapshot without persistent field caches.

This author-side calculation requires an installed simesh package. It writes a
new result directory and never updates the separately published plotting inputs.
"""

import argparse
import gc
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import simesh as sm
from simesh import applications as app

CHUNK = 512
MEMORY_LIMIT = 3 * 1024**3


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def recover_thermodynamics(source, model):
    """Recover native interior density and temperature once for this run."""
    mesh = source.mesh
    state = np.empty((mesh.leaf_count, 2, *mesh.block_shape), dtype=np.float64)
    for first in range(0, mesh.leaf_count, CHUNK):
        last = min(first + CHUNK, mesh.leaf_count)
        raw = sm.read_fields(source, leaf_ids=np.arange(first, last))
        therm = sm.mhd_fields(
            raw, model=model, outputs=("density", "temperature"), invalid="raise"
        )
        values = therm.interior()
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("Thermodynamic recovery requires positive finite values")
        state[first:last] = np.moveaxis(values, -1, 1)
    return state


def make_rays(mesh, view, pixels):
    """Use the original lower-face rays and full-domain projection geometry."""
    axis = {"x": 0, "y": 1}[view]
    horizontal = 1 - axis
    width = mesh.upper - mesh.lower
    first, second, direction = np.zeros((3, 3))
    first[horizontal] = width[horizontal]
    second[2] = width[2]
    direction[axis] = 1
    plane = sm.Plane(mesh.lower.copy(), first, second, pixels)
    return sm.RaySet.from_plane(plane, direction)


def prepare_scalar(mesh, values, name, units=None):
    """Detach explicit exact-phase support from the temporary scalar array."""
    with sm.source_from_arrays(
        mesh, values, (name,), units=units, copy=False
    ) as source:
        return sm.prepare(
            source, scheme="exact-phase", workers=1, memory_limit=MEMORY_LIMIT
        )


def integrate(fields, rays, workers):
    result = app.los(
        fields,
        rays,
        quadrature="gauss2",
        step_fraction=0.5,
        workers=workers,
        ray_batch=2048,
        memory_limit=MEMORY_LIMIT,
    )
    if not result.complete or not np.isfinite(result.values).all():
        raise ValueError("Line-of-sight integration requires complete finite coverage")
    return result


def verify_integral(prepared, retained, length_cm, workers):
    """Check the same emissivity reconstruction using independent quadrature."""
    indices = np.unique(
        np.r_[
            np.linspace(0, len(retained.values) - 1, 64).astype(int),
            np.argmax(retained.values),
        ]
    )
    mask = np.zeros(len(retained.values), dtype=bool)
    mask[indices] = True
    rays = retained.rays.select(mask.reshape(retained.rays.origins.shape))
    reference = app.los(
        prepared, rays, quadrature="midpoint", step_fraction=0.125, workers=workers
    )
    if not reference.complete:
        raise ValueError("Independent quadrature has incomplete coverage")
    expected = retained.values[indices]
    error = np.abs(reference.values * length_cm - expected)
    scale = float(np.max(expected))
    if scale == 0:
        if np.any(error):
            raise ValueError("Zero-emission quadrature check failed")
        relative = np.zeros_like(error)
    else:
        relative = error / np.maximum(np.abs(expected), scale * 1e-12)
    if not np.isfinite(relative).all() or np.max(relative) >= 1e-3:
        raise ValueError("Independent quadrature exceeds the original tolerance")
    return {
        "ray_count": len(indices),
        "max_relative_error": float(np.max(relative)),
        "max_absolute_error": float(np.max(error)),
        "scope": "Quadrature of the same emissivity-first reconstruction; not model convergence",
    }


def compute_bands(
    mesh, state, model, views, wavelengths, pixels, output, workers, verify
):
    """Share each emissivity field across views and write only final results."""
    rays = {view: make_rays(mesh, view, pixels) for view in views}
    for view in views:
        (output / f"view-{view}").mkdir()
    for wave in wavelengths:
        emission = np.empty((mesh.leaf_count, 1, *mesh.block_shape), dtype=np.float64)
        euv = sm.EUV(
            wavelength=wave,
            density_convention="electron-hydrogen",
            composition=model.composition,
        )
        for first in range(0, mesh.leaf_count, CHUNK):
            last = min(first + CHUNK, mesh.leaf_count)
            emission[first:last, 0] = euv.emissivity(
                state[first:last, 0], state[first:last, 1]
            )
        prepared = prepare_scalar(
            mesh, emission, "emissivity", {"emissivity": "DN s^-1 pixel^-1 cm^-1"}
        )
        del emission
        for view in views:
            print(f"Integrating {wave} Angstrom along +{view.upper()}", flush=True)
            result = integrate(prepared, rays[view], workers)
            if np.any(result.values < 0):
                raise ValueError("Emission must be nonnegative")
            result = replace(
                result,
                values=result.values * model.units.length_cm,
                units="DN s^-1 pixel^-1",
            )
            target = output / f"view-{view}" / f"euv{wave}-plus-{view}.result.npz"
            sm.save_result(
                target,
                result,
                metadata={
                    "wavelength_angstrom": wave,
                    "model": euv.identity,
                    "energy_kind": model.energy_kind,
                    "units": asdict(model.units),
                    "absorption": False,
                },
            )
            np.testing.assert_array_equal(
                sm.load_result(target).result.values, result.values
            )
            if verify and wave == 94:
                write_json(
                    target.parent / "independent-quadrature-check.json",
                    verify_integral(prepared, result, model.units.length_cm, workers),
                )
            del result
        del prepared
        gc.collect()


def compute_diagnostics(source, state, model, output, workers):
    """Recompute Figure 7 diagnostics using the original geometry and weights."""
    mesh = source.mesh
    geometry = sm.AxisSlice(mesh, "y", 0.0)
    raw = sm.read_fields(source, leaf_ids=geometry.leaf_ids)
    fields = sm.mhd_fields(
        raw, model=model, outputs=("density", "temperature", "speed", "velocity")
    )
    section = sm.slice_axis(fields, "y", 0.0)
    data = {
        "slice_values": section.values.copy(),
        "slice_edges": np.array(
            [geometry.cell_edges(i) for i in range(len(geometry.leaf_ids))]
        ),
    }
    del raw, fields, section
    boxes = [
        np.array([[-4, -20, 2], [4, 20, 6]]),
        np.array([[-4, -20, 6], [4, 20, 10]]),
    ]
    bins = np.geomspace(1e3, 1e8, 81)
    with sm.source_from_arrays(
        mesh,
        state,
        ("density", "temperature"),
        units={"density": "g cm^-3", "temperature": "K"},
        copy=False,
    ) as thermal_source:
        for index, box in enumerate(boxes):
            hit = np.all(
                (mesh.bounds[:, 1] > box[0]) & (mesh.bounds[:, 0] < box[1]), axis=1
            )
            fields = sm.read_fields(thermal_source, leaf_ids=np.flatnonzero(hit))
            units = sm.LengthUnits(model.units.length_cm, "cm")
            histogram = sm.histogram(
                fields,
                bins,
                "temperature",
                weights=fields,
                weight_component="density",
                region=box,
                units=units,
            )
            mass = sm.volume_integral(fields, "density", region=box, units=units)
            np.testing.assert_allclose(
                histogram.bin_weights.sum(), mass.value, rtol=1e-12
            )
            data[f"hist_{index}"] = histogram.bin_weights
            data[f"mass_{index}"] = mass.value
    data["temperature_edges"] = bins
    plane = sm.Plane([-10, -20, 0], [20, 0, 0], [0, 0, 16], (240, 192))
    rays = sm.RaySet.from_plane(plane, [0, 1, 0])
    for name in ("density", "rhoT"):
        values = np.empty((mesh.leaf_count, 1, *mesh.block_shape), dtype=np.float64)
        for first in range(0, mesh.leaf_count, CHUNK):
            last = min(first + CHUNK, mesh.leaf_count)
            values[first:last, 0] = (
                state[first:last, 0]
                if name == "density"
                else state[first:last, 0] * state[first:last, 1]
            )
        prepared = prepare_scalar(mesh, values, name)
        del values
        result = integrate(prepared, rays, workers)
        data[name] = result.values.reshape(240, 192).T * model.units.length_cm
        del prepared, result
        gc.collect()
    if not (data["density"] > 0).all():
        raise ValueError("Column mass must be positive")
    data["weighted_temperature"] = data["rhoT"] / data["density"]
    np.savez_compressed(output / "diagnostics.npz", **data)
    return {
        "slice_y_code": 0,
        "region_boxes_code": [box.tolist() for box in boxes],
        "projection_pixels": [240, 192],
        "projection": "Full Y domain",
        "histogram": "Native cell-overlap mass weights; 80 log temperature bins",
    }


def main(default_views=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Scientific configuration with units, gamma, energy_kind, and composition",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty result directory, outside published plotting inputs",
    )
    parser.add_argument(
        "--views", nargs="+", choices=("x", "y"), default=default_views or ["y"]
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        help="Wavelengths in Angstrom; default: all configured bands",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also compute Figure 7 thermal diagnostics",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check 94 Angstrom with independent quadrature",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    snapshot, config_path = args.snapshot.resolve(), args.config.resolve()
    protected = (Path(__file__).resolve().parent / "data").resolve()
    for path in (snapshot, config_path, protected):
        if (
            output == path
            or output in path.parents
            or (path == protected and path in output.parents)
        ):
            parser.error(
                "output must be separate from raw inputs, configuration, and published data"
            )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(
            "output must be a new or empty directory; previous results are never overwritten"
        )
    if args.workers < 1:
        parser.error("workers must be positive")
    config = json.loads(config_path.read_text())
    model = sm.IdealMHD(
        gamma=config["gamma"],
        energy_kind=config["energy_kind"],
        units=sm.MHDUnits(**config["units"]),
        composition=sm.CoronalComposition(**config["composition"]),
    )
    wavelengths = list(dict.fromkeys(args.bands or config["wavelengths_angstrom"]))
    if not set(wavelengths).issubset(config["wavelengths_angstrom"]):
        parser.error("bands must be present in the supplied scientific configuration")
    if args.verify and 94 not in wavelengths:
        parser.error("--verify requires the 94 Angstrom band")
    views = list(dict.fromkeys(args.views))
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "snapshot": {"name": snapshot.name, "sha256": file_hash(snapshot)},
        "configuration_sha256": file_hash(config_path),
        "script_sha256": file_hash(__file__),
        "simesh_version": sm.__version__,
        "numpy_version": np.__version__,
        "gamma": model.gamma,
        "energy_kind": model.energy_kind,
        "units": asdict(model.units),
        "composition": asdict(model.composition),
        "wavelengths_angstrom": wavelengths,
        "views": views,
        "pixels": config["pixels"],
        "preparation": "exact-phase",
        "quadrature": "gauss2",
        "step_fraction": 0.5,
        "model": "Optically thin electron-hydrogen emission; no absorption, PSF or noise",
        "calibration_status": config.get(
            "calibration_status", "Not independently established"
        ),
    }
    write_json(output / "configuration.json", provenance)
    with sm.open_amrvac(snapshot) as source:
        print("Recovering density and temperature for this run", flush=True)
        state = recover_thermodynamics(source, model)
        compute_bands(
            source.mesh,
            state,
            model,
            views,
            wavelengths,
            tuple(config["pixels"]),
            output,
            args.workers,
            args.verify,
        )
        if args.diagnostics:
            print("Computing native thermal diagnostics", flush=True)
            provenance["diagnostics"] = compute_diagnostics(
                source, state, model, output, args.workers
            )
        del state
    write_json(output / "summary.json", dict(provenance, status="complete"))
    print(
        "Completed. Only final results and scientific provenance were saved.",
        flush=True,
    )


if __name__ == "__main__":
    main()
