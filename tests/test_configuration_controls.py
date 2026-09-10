"""Physical consistency of the public bipolar strength and RBSL helicity controls."""

import numpy as np
import pytest

from simesh.tools import configurations as config


def curl_at(function, positions, step):
    derivatives = []
    for axis in range(3):
        offset = np.zeros_like(positions)
        offset[axis] = step
        derivatives.append((function(positions+offset)-function(positions-offset))/(2*step))
    dx, dy, dz = derivatives
    return np.array([dy[2]-dz[1], dz[0]-dx[2], dx[1]-dy[0]])


@pytest.mark.parametrize("strength", [-2., 0., 3.])
def test_bipolar_field_matches_curl_of_vector_potential(strength):
    positions = np.array([[.2, .7], [.3, .2], [.4, .6]])
    expected = curl_at(lambda p: config.bipolar_Avec(p, strength, 1., 1.), positions, 1e-5)
    np.testing.assert_allclose(config.bipolar_Bvec(positions, strength, 1., 1.), expected,
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("flux", [-1., 1.])
def test_helicity_reverses_current_helicity_and_preserves_axial_field(flux):
    angle = np.linspace(0, 2*np.pi, 128, endpoint=False)
    axis = np.array([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
    positions = np.array([[1.05], [0.], [.05]])
    axial = []
    for positive in (True, False):
        def magnetic(p):
            return curl_at(lambda x: config.rbsl_Avec(x, axis, .2, flux, positive), p, 1e-4)
        field = magnetic(positions)
        current = curl_at(magnetic, positions, 1e-3)
        helicity = float(np.sum(field*current))
        assert helicity > 0 if positive else helicity < 0
        axial.append(field[1])  # Ring tangent at y=0 is the y direction.
    np.testing.assert_allclose(axial[0], axial[1], rtol=1e-8)


def test_tdm_propagates_strength_and_helicity_to_its_components():
    def field(strength, positive):
        return config.TDm_slab((-.4, -.4, .1), (.4, .4, .6), (4, 4, 4),
                               1., .2, positive, 64, strength, .5, .7)
    baseline = field(1., True)
    np.testing.assert_allclose(field(2., True), 2*baseline, rtol=1e-12, atol=1e-12)
    assert np.isfinite(baseline).all()
    assert not np.allclose(field(1., False), baseline)
