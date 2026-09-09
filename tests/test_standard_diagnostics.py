"""Known derivatives, physical normalization and selectable surface diagnostics."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from fixtures import mixed_source
from test_connectivity import magnetic_source


def test_standard_derivatives_and_physical_current_on_amr():
    with mixed_source()[0] as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet([[.25,.25,.25],[1.5,.5,.5]])
    gradient = sm.gradient(fields,"b1",workers=2)
    divergence = sm.divergence(fields,("b1","b2","b3"),workers=2)
    np.testing.assert_allclose(app.sample(gradient,points).values,[[0.,1.,2.]]*2,atol=1e-13)
    np.testing.assert_allclose(app.sample(divergence,points).values,0.,atol=1e-13)
    units = sm.MagneticUnits(field_tesla=.002,length_m=5e5,permeability_h_m=2e-6)
    current = sm.current_density(fields,units=units,workers=2)
    # The analytic curl is (3,-3,3), with SI multiplier 0.002 A/m^2.
    np.testing.assert_allclose(app.sample(current,points).values,[[.006,-.006,.006]]*2,atol=1e-14)
    assert current.valid_halo == 1 and current.fields[0].units == "A m^-2"
    norm = sm.magnitude(current)
    assert norm.valid_halo == 1 and norm.fields[0].units == "A m^-2"
    np.testing.assert_allclose(app.sample(norm,points).values,np.sqrt(3)*.006,atol=1e-14)
    product = sm.dot(fields,fields)
    result = sm.gradient(product,workers=2)
    assert result.valid_halo == 1
    with pytest.raises(TypeError):
        sm.current_density(fields,units="SI")
    with pytest.raises(MemoryError):
        sm.magnitude(fields,memory_limit=1)


def test_magnetic_pressure_energy_and_uniform_product():
    with magnetic_source(lambda x,y,z: np.array([np.full_like(x,3.),np.full_like(x,4.),np.zeros_like(x)]),mixed=True) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    units = sm.MagneticUnits(field_tesla=2.,length_m=3.,permeability_h_m=5.)
    pressure = sm.magnetic_pressure(fields,units=units)
    energy = sm.magnetic_energy_density(fields,units=units)
    plane = sm.Plane([-.5,-.5,.5],[1,0,0],[0,1,0],(5,4))
    np.testing.assert_allclose(app.field_map(pressure,plane).image,10.)
    np.testing.assert_allclose(app.field_map(energy,plane).image,10.)
    assert pressure.fields[0].units == "Pa" and energy.fields[0].units == "J m^-3"
    grid = app.uniform_grid(pressure,(12,10,8),workers=3)
    assert grid.values.shape == (12,10,8,1) and grid.valid.all()
    np.testing.assert_allclose(grid.values,10.)
    np.testing.assert_allclose(grid.axes[2],(np.arange(8)+.5)/8)
    np.testing.assert_allclose(grid.spacing,[2/12,2/10,1/8])
    with pytest.raises(MemoryError):
        app.uniform_grid(pressure,(100,100,100),memory_limit=1000)


def test_twist_only_is_independent_of_q_work(monkeypatch):
    import simesh.connectivity as implementation
    with magnetic_source(lambda x,y,z: np.array([-.5*y,.5*x,np.ones_like(x)]),mixed=True) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet.boundary(fields.mesh,"zmin",(4,3))
    combined = app.surface_diagnostics(fields,points,workers=2)
    curl = sm.curl(fields)
    def unavailable(*args,**kwargs):
        raise AssertionError("Q computation was requested during twist-only analysis")
    monkeypatch.setattr(implementation,"_stencil",unavailable)
    monkeypatch.setattr(implementation,"_squashing",unavailable)
    only_twist = app.surface_diagnostics(replace(fields,valid_halo=1),points,
        quantities=("twist",),curl_field=curl,workers=2)
    assert only_twist.quantities == ("twist",) and only_twist.data.q is None
    np.testing.assert_array_equal(only_twist.data.twist,combined.data.twist)
    np.testing.assert_array_equal(only_twist.data.footpoints,combined.data.footpoints)
    np.testing.assert_array_equal(only_twist.image("valid"),only_twist.image("twist_valid"))
    with pytest.raises(ValueError,match="not computed"):
        only_twist.image("q")
    with pytest.raises(ValueError,match="not computed"):
        only_twist.threshold(q_min=2.)
    limited = app.connectivity(fields,points,quantities="twist",max_steps=1)
    assert not limited.twist_valid.any()
    assert len(limited.threshold(abs_twist_min=0.)) == 0
    batches = list(app.iter_connectivity(fields,points,quantities="twist",seed_batch=5))
    np.testing.assert_array_equal(np.concatenate([b.points.ids for b in batches]),points.ids)
    empty = sm.line_diagnostics(fields,np.empty((0,3)),quantities="twist")
    assert empty.q is None and empty.twist.shape == (0,)


def test_q_only_bottom_and_invalid_quantity_selection():
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)])) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    result = app.bottom_diagnostics(fields,(3,4),quantities="q",workers=2)
    assert result.quantities == ("q",) and result.data.twist is None
    np.testing.assert_allclose(result.image("q"),2.)
    np.testing.assert_array_equal(result.points.positions[:,2],fields.mesh.lower[2])
    for quantities in ((),("q","q"),("current",)):
        with pytest.raises(ValueError):
            app.bottom_diagnostics(fields,(2,2),quantities=quantities)
