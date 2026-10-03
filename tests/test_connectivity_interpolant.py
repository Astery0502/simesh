"""Small analytic and endpoint checks for local interpolant gradients."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from test_connectivity import magnetic_source, BOX


METHOD = "variational-interpolant"


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("scale", [1e-150, 1., 1e150])
def test_interpolant_affine_q_on_small_amr(mixed, scale):
    with magnetic_source(lambda x,y,z: scale*np.array([.5*x,-.5*y,np.ones_like(z)]),
                         cells=4, mixed=mixed) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    result = sm.qsl(fields, np.array([[0.,0.,.5],[.12,-.15,.5]]), bounds=BOX,
                    method=METHOD, twist=False, step_fraction=.125)
    assert result.valid.all() and result.complete.all()
    np.testing.assert_allclose(result.q, 2*np.cosh(.75), rtol=2e-8)


def test_interpolant_gradient_avoids_large_vector_difference_overflow():
    # Adjacent x nodes differ by 2e308, but the normalized gradient is finite.
    with magnetic_source(lambda x,y,z: np.array([np.where(x<0,-1e308,1e308),
                                                -1e308*y,np.full_like(z,1e308)]),
                         cells=4) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    result = sm.qsl(fields, np.array([[0.,0.,.5]]), bounds=BOX,
                    method=METHOD, twist=False, step_fraction=.015625)
    assert result.valid.all()
    np.testing.assert_allclose(result.q, 2*np.cosh(3.75), rtol=2e-8)


def test_interpolant_nonlinear_mapping_matches_independent_footpoints():
    with magnetic_source(lambda x,y,z: np.array([y*z,z*x,1+x*y]), cells=4) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.13,.17,.5],[-.21,.16,.5]])
    controls = dict(bounds=BOX, twist=False, step_fraction=.125)
    result = sm.qsl(fields, seeds, method=METHOD, **controls)
    endpoint = sm.qsl(fields, seeds, method="finite-difference", delta=1e-4, **controls)
    refined = sm.qsl(fields, seeds, method=METHOD, **{**controls,"step_fraction":.0625})
    flux = sm.qsl(fields, seeds, method=METHOD, normalization="flux", **controls)
    assert result.valid.all() and endpoint.valid.all() and refined.valid.all() and flux.valid.all()
    np.testing.assert_allclose(result.q, endpoint.q, rtol=2e-6)
    np.testing.assert_allclose(result.q, refined.q, rtol=2e-7)
    np.testing.assert_allclose(result.q, flux.q, rtol=2e-7)
    np.testing.assert_allclose(result.footpoints, endpoint.footpoints, atol=2e-10)


def test_interpolant_piecewise_map_across_interpolation_knots():
    with magnetic_source(lambda x,y,z: np.array([.8*x*x,-1.6*x*y,np.ones_like(z)]),
                         cells=8) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.36,.17,.5]])
    controls = dict(bounds=BOX, twist=False, step_fraction=.0625)
    result = sm.qsl(fields, seeds, method=METHOD, **controls)
    endpoint = sm.qsl(fields, seeds, method="finite-difference", delta=1e-4, **controls)
    assert result.valid.all() and endpoint.valid.all()
    assert result.footpoints[0,0,0] < .375 < result.footpoints[0,1,0]
    np.testing.assert_allclose(result.q, endpoint.q, rtol=3e-6)


def test_interpolant_support_budget_and_roundtrip(tmp_path):
    with magnetic_source(lambda x,y,z: np.array([.5*x,-.5*y,np.ones_like(z)]), cells=4) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    # Allocated outer padding must not silently supply a second valid layer.
    values = fields.values.copy()
    for axis in (1,2,3):
        index = [slice(None)]*5
        index[axis] = [0,-1]
        values[tuple(index)] = np.nan
    fields = replace(fields, _values=values, valid_halo=1)
    points = sm.PointSet(np.array([[0.,0.,.5],[.1,.1,.5]]))
    controls = dict(bounds=BOX, method=METHOD, quantities="q", seed_batch=1)
    result = app.connectivity(fields, points, memory_limit=50000, **controls)
    assert result.data.valid.all() and result.data.twist is None
    restored = sm.load_result(sm.save_result(tmp_path/"interpolant.npz", result)).result
    assert restored.data.method == METHOD
    np.testing.assert_array_equal(restored.data.q, result.data.q)
    empty = app.connectivity(fields, sm.PointSet(np.empty((0,3))), **controls)
    assert empty.data.method == METHOD and empty.data.q.shape == (0,)
    batches = list(app.iter_connectivity(fields, points, **controls))
    np.testing.assert_array_equal(np.concatenate([batch.data.q for batch in batches]), result.data.q)
    for options in ({"twist":True}, {"twist":False,"delta":1e-4}):
        with pytest.raises(ValueError):
            sm.qsl(fields, points.positions, method=METHOD, **options)
    with pytest.raises(ValueError, match="requires at least 1"):
        sm.qsl(replace(fields, valid_halo=0), points.positions, method=METHOD, twist=False)
