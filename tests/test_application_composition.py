"""Name-based composition and consistent diagnostic selection across user entry points."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from test_connectivity import magnetic_source


@pytest.fixture
def magnetic():
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)])) as source:
        return sm.prepare(source, scheme="exact-phase")


def test_diagnostic_selection_is_shared_by_point_surface_bottom_and_batches(magnetic):
    magnetic = replace(magnetic, valid_halo=1)  # Q-only must not require automatic curl support.
    points = sm.PointSet.boundary(magnetic.mesh, "zmin", (2, 2))
    calls = [lambda **kw: app.connectivity(magnetic, points, **kw),
             lambda **kw: app.surface_diagnostics(magnetic, points, **kw),
             lambda **kw: app.bottom_diagnostics(magnetic, (2,2), **kw),
             lambda **kw: next(app.iter_connectivity(magnetic, points, **kw))]
    for call in calls:
        named, compatibility = call(quantities="q"), call(twist=False)
        assert compatibility.quantities == ("q",)
        np.testing.assert_array_equal(named.data.q, compatibility.data.q)
        with pytest.raises(ValueError, match="quantities alone"):
            call(quantities="q", twist=False)
        with pytest.raises(ValueError, match="boolean"):
            call(twist="false")


def test_named_los_follows_component_after_reordering(magnetic):
    plane = sm.Plane([-.5,-.5,-1], [1,0,0], [0,1,0], (2,2))
    rays = sm.RaySet.from_plane(plane, [0,0,1])
    expected = app.los(magnetic, rays, component="b3")
    reordered = sm.select_fields(magnetic, ("b3", "b1", "b2"))
    actual = app.los(reordered, rays, component="b3")
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.status, expected.status)
    raw = sm.integrate_los(reordered, plane, [0,0,1], component="b3")
    np.testing.assert_allclose(raw.values, actual.image)
    for name in ("missing", True, -1):
        with pytest.raises(ValueError):
            app.los(reordered, rays, component=name)


def test_named_thermal_inputs_preserve_interpretation_after_composition(magnetic):
    state = sm.derive_many(magnetic, {"temperature":"K", "density":"g cm^-3"},
                          lambda ctx: {"temperature":ctx.field("b3")*1e6,
                                       "density":ctx.field("b3")*1e-15})
    thermal = sm.thermal_fields(state, state, density_component="density", temperature_component="temperature",
                                density_unit_g_cm3=1., temperature_label="analytic state")
    reordered = sm.select_fields(state, ("density", "temperature"))
    expected = sm.thermal_fields(reordered, reordered, density_component=0, temperature_component=1,
                                 density_unit_g_cm3=1., temperature_label="analytic state")
    np.testing.assert_array_equal(thermal.values, expected.values)
    for name in ("missing", True):
        with pytest.raises(ValueError):
            sm.thermal_fields(state, state, density_component=name, temperature_component="temperature",
                              density_unit_g_cm3=1., temperature_label="analytic state")
    with pytest.raises(ValueError, match="kelvin"):
        sm.thermal_fields(state, state, density_component="density", temperature_component="density",
                          density_unit_g_cm3=1., temperature_label="invalid units")
