"""Create a teaching snapshot, sample a magnetic map and verify its saved result.

Run from the repository root:
    .venv/bin/python examples/user_quickstart.py --output example-output/user-quickstart

The output directory must be new. Values are synthetic SI quantities on a
unit-coordinate cube, with one coordinate unit equal to 1e6 meters.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import simesh as sm
from simesh import amrvac, applications as app


def run(output):
    output.mkdir(parents=True)
    names = ["rho", "m1", "m2", "m3", "e", "b1", "b2", "b3"]
    units = dict(zip(names, ["kg m^-3"] + ["kg m^-2 s^-1"] * 3
                     + ["J m^-3"] + ["T"] * 3))
    gamma = 5 / 3
    density, velocity_z, pressure, magnetic_z = 1e-12, 1e4, 0.02, 1e-3
    length_m = 1e6
    magnetic_units = sm.MagneticUnits(field_tesla=1.0, length_m=length_m)
    data = np.zeros((16, 16, 16, len(names)), dtype=np.float64)
    data[..., 0] = density
    data[..., 3] = density * velocity_z
    data[..., 4] = (pressure / (gamma - 1) + 0.5 * density * velocity_z**2
                    + magnetic_z**2 / (2 * magnetic_units.permeability_h_m))
    data[..., 7] = magnetic_z
    snapshot = output / "snapshot.dat"
    amrvac.write_datfile_from_uniform(
        str(snapshot), data, names, np.zeros(3), np.ones(3),
        np.array([8, 8, 8]), params=np.array([gamma]),
    )

    with sm.open_amrvac(snapshot, units=units) as source:
        magnetic = sm.prepare(source, ("b1", "b2", "b3"), scheme="coordinate-phase")
        rho = sm.read_fields(source, "rho")
    plane = sm.Plane([0, 0, 0.5], [1, 0, 0], [0, 1, 0], (8, 8))
    field_map = app.field_map(magnetic, plane, components="b3")
    mass = sm.volume_integral(rho, "rho", units=sm.LengthUnits(length_m, "m"))
    assert field_map.usable.all()
    assert mass.coverage.complete
    np.testing.assert_allclose(field_map.values, magnetic_z, rtol=1e-12, atol=0)
    np.testing.assert_allclose(mass.value, density * length_m**3, rtol=1e-12, atol=0)

    metadata = {
        "description": "Synthetic constant ideal-MHD teaching snapshot",
        "coordinate_length_m": length_m,
        "field_units": units,
        "scales_to_si": {
            "density_kg_m3": 1.0, "momentum_kg_m2_s": 1.0,
            "energy_j_m3": 1.0, "magnetic_field_tesla": 1.0,
        },
        "gamma": gamma,
        "energy_kind": "total",
        "helium_abundance": 0.1,
        "preparation_scheme": "coordinate-phase",
        "known_values": {
            "density_kg_m3": density, "velocity_z_m_s": velocity_z,
            "pressure_Pa": pressure, "magnetic_z_T": magnetic_z,
            "mass_kg": density * length_m**3,
        },
    }
    result_path = sm.save_result(
        output / "magnetic-map.result.npz", field_map, metadata=metadata,
        source={"path": str(snapshot), "description": metadata["description"]},
    )
    loaded = sm.load_result(result_path)
    np.testing.assert_array_equal(loaded.result.values, field_map.values)
    np.testing.assert_array_equal(loaded.result.points.ids, field_map.points.ids)
    (output / "snapshot-info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Usable map samples: {int(field_map.usable.sum())}/64")
    print(f"Magnetic Bz: {field_map.values[0, 0]:g} T")
    print(f"Mass: {mass.value:g} kg; complete coverage: {mass.coverage.complete}")
    print("Result round trip: values and point IDs match")
    print(f"Files: {snapshot}, {result_path}, {output / 'snapshot-info.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example-output/user-quickstart"))
    run(parser.parse_args().output)
