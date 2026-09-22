"""Real-world Cartesian header spelling through the public slice workflow."""

import numpy as np
import pytest
import simesh as sm

from fixtures import write_dat


@pytest.mark.parametrize("geometry", ["Cartesian", "Cartesian_3D"])
def test_cartesian_periodic_native_slice(tmp_path, geometry):
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]),
                              lower=(-1, -1, -1), upper=(1, 1, 1),
                              block_shape=(4, 4, 4), periodic=(False, False, True))
    raw = np.arange(64, dtype=float).reshape(1, 1, 4, 4, 4)
    path = tmp_path / "cartesian.dat"
    write_dat(path, mesh, raw, geometry=geometry)
    with sm.open_amrvac(path) as source:
        assert source.mesh.periodic == (False, False, True)
        result = sm.slice_axis(sm.read_fields(source), "x", 0.)
    assert result.usable.all()
    np.testing.assert_array_equal(result.values[0, ..., 0], raw[0, 0, 2])


def test_noncartesian_source_remains_rejected(tmp_path):
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]), lower=(0, 0, 0),
                              upper=(1, 1, 1), block_shape=(4, 4, 4))
    path = tmp_path / "cylindrical.dat"
    write_dat(path, mesh, np.zeros((1, 1, 4, 4, 4)), geometry="cylindrical")
    with pytest.raises(ValueError, match="Cartesian 3D"):
        sm.open_amrvac(path)
