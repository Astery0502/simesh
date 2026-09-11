"""Physical transfer limits, AMR ordering and explicit response conventions."""

from dataclasses import replace
import numpy as np
import pytest

import simesh as sm
from simesh import applications as app
from simesh.fields import publish
from simesh.physics._euv_tables import RESPONSES
from simesh.physics.radiation import _saha_log_ratios, _saha_populations
from fixtures import mixed_source


@pytest.mark.parametrize("wavelength", tuple(RESPONSES))
def test_response_tables_and_native_amr_integration(wavelength):
    grid, response, logarithmic = RESPONSES[wavelength]
    model = sm.EUV(wavelength=wavelength)
    temperature = 10.**np.asarray(grid) if logarithmic else np.asarray(grid)
    np.testing.assert_allclose(model.response(temperature), response, rtol=2e-13, atol=1e-98)
    assert np.array_equal(model.response([temperature[0]*.9, temperature[-1]*1.1]), [0., 0.])
    i = len(grid)//2
    midpoint = (grid[i]+grid[i+1])/2
    expected = np.sqrt(response[i]*response[i+1]) if logarithmic else (response[i]+response[i+1])/2
    np.testing.assert_allclose(model.response(10.**midpoint if logarithmic else midpoint), expected,
                               rtol=2e-13, atol=1e-98)
    source, _ = mixed_source(lambda x, y, z: np.array([1+.2*x, 1+x*0, 1+x*0]))
    density = sm.prepare(source, ['b1'], scheme='exact-phase')
    thermal = sm.thermal_fields(density, temperature[i], density_unit_g_cm3=1e-15,
                                temperature_label='response test', model=model)
    plane = sm.orthographic_plane(source.mesh.lower, source.mesh.upper, [.3,.2,1], (3,3))
    native = sm.integrate_thermal_los(thermal, plane, [.3,.2,1], model=model, length_unit_cm=1e8)
    reference = sm.integrate_thermal_los(thermal, plane, [.3,.2,1], model=model,
                                       length_unit_cm=1e8, implementation='reference')
    rays = app.thermal_los(thermal, sm.RaySet.from_plane(plane, [.3,.2,1]),
                           model=model, length_unit_cm=1e8, workers=2, ray_batch=2)
    assert native.complete and reference.complete and rays.complete
    np.testing.assert_allclose(native.values, reference.values, rtol=1e-11, atol=1e-90)
    np.testing.assert_allclose(native.values, rays.image, rtol=1e-13, atol=1e-90)


def test_new_emission_measure_preserves_explicit_historical_choices():
    rho, t = 1e-15, 1e6
    composition = sm.CoronalComposition(.2)
    ne = composition.number_density(rho)
    nh = composition.number_density(rho, convention='amrvac-hydrogen')
    new = sm.EUV(composition=composition)
    old = sm.AIA171(composition=composition)
    np.testing.assert_allclose(new.emissivity(rho, t), ne*nh*new.response(t))
    np.testing.assert_allclose(old.emissivity(rho, t), ne*ne*old.response(t))
    assert old.identity != new.identity


def constant_slabs(j, kappa):
    """Independently prescribed constant coefficients in each physical slab."""
    mesh = sm.mesh_from_forest((len(j),1,1), np.ones(len(j),bool), lower=(0,0,0),
                              upper=(len(j),1,1), block_shape=(8,8,8))
    values = np.empty((len(j),10,10,10,2))
    values[...,0] = np.asarray(j)[:,None,None,None]
    values[...,1] = np.asarray(kappa)[:,None,None,None]
    return publish(mesh, values, sm.select_region(mesh, [mesh.lower, mesh.upper]),
        (sm.FieldDefinition('emissivity','K cm^-1','prepared-node'),
         sm.FieldDefinition('opacity','cm^-1','prepared-node')), 1, 1, 'analytic-slabs', 'test',
        {'radiation_units':'K','model':'analytic-slabs','temperature':'prescribed',
         'density_unit_g_cm3':1.,'absorption':'prescribed'})


@pytest.mark.parametrize('tau', [0., 1e-16, 1e-5, 1., 1000.])
def test_exact_slab_transparent_and_opaque_limits(tau):
    coefficients = constant_slabs([3.], [tau/2])
    rays = sm.RaySet(sm.PointSet([[0,.5,.5],[-1,2,.5]]), [1,0,0])
    result = sm.radiative_los(coefficients, rays, length_unit_cm=2., background=7., subdivisions=1)
    factor = -np.expm1(-tau)/tau if tau else 1.
    expected = 6*factor+7*np.exp(-tau)
    assert result.complete
    np.testing.assert_allclose(result.intensity.values, [expected,7], rtol=3e-14)
    np.testing.assert_allclose(result.optical_depth.values, [tau,0], rtol=3e-14)
    np.testing.assert_allclose(result.thin_intensity.values, [6.,0], rtol=3e-14)
    np.testing.assert_allclose(result.absorption_fraction, [1-factor,0], atol=3e-15)


def test_front_to_back_order_and_far_boundary_background():
    fields = constant_slabs([2.,8.], [1.,3.])
    rays = sm.RaySet(sm.PointSet([[0,.5,.5],[2,.5,.5]]), [[1,0,0],[-1,0,0]])
    a, b = 2*-np.expm1(-1.), 8/3*-np.expm1(-3.)
    result = sm.radiative_los(fields, rays, length_unit_cm=1., background=4., subdivisions=2)
    reference = sm.radiative_los(fields, rays, length_unit_cm=1., background=4., subdivisions=2,
                                implementation='reference')
    assert result.complete and reference.complete
    np.testing.assert_allclose(result.intensity.values,
        [a+np.exp(-1)*b+4*np.exp(-4), b+np.exp(-3)*a+4*np.exp(-4)], rtol=3e-14)
    np.testing.assert_allclose(result.intensity.values, reference.intensity.values, rtol=3e-14)


def test_absorption_diagnostic_survives_bright_background():
    fields = constant_slabs([3.], [1.])
    rays = sm.RaySet(sm.PointSet([[0,.5,.5]]), [1,0,0])
    result = sm.radiative_los(fields,rays,length_unit_cm=1.,background=1e30)
    np.testing.assert_allclose(result.absorption_fraction, np.exp(-1.), rtol=1e-13)


def test_radio_isothermal_slab_and_saved_products(tmp_path):
    source, _ = mixed_source(lambda x,y,z:np.ones((3,*x.shape)))
    density = sm.prepare(source,['b1'],scheme='exact-phase')
    model = sm.RadioFreeFree(frequency_hz=1e9)
    thermal = sm.thermal_fields(density, 1e6, density_unit_g_cm3=1e-14,
                                temperature_label='isothermal', model=model)
    coefficients = sm.radiation_fields(thermal, model=model)
    plane = sm.orthographic_plane(source.mesh.lower,source.mesh.upper,[1,.3,.2],(6,5))
    rays = sm.RaySet.from_plane(plane,[1,.3,.2])
    result = sm.radiative_los(coefficients,rays,length_unit_cm=1e9,workers=3,ray_batch=7)
    reference = sm.radiative_los(coefficients,rays,length_unit_cm=1e9,implementation='reference')
    assert result.complete and reference.complete
    expected = 1e6*-np.expm1(-result.optical_depth.values)
    np.testing.assert_allclose(result.intensity.values,expected,rtol=3e-13)
    np.testing.assert_allclose(result.intensity.values,reference.intensity.values,rtol=3e-13)
    for label in ('intensity','optical_depth','thin_intensity'):
        item = getattr(result,label)
        path = tmp_path/(label+'.result.npz')
        sm.save_result(path,item)
        loaded = sm.load_result(path).result
        np.testing.assert_array_equal(loaded.values,item.values)
        assert loaded.metadata['model'] == model.identity


def test_absorber_cold_limit_edges_hot_plasma_and_charge_balance():
    absorber = sm.HHeAbsorption()
    n = 1e11
    expected = n*(5.16e-20+.1*9.25e-19)
    np.testing.assert_allclose(absorber.opacity(n,100.,wavelength=171),expected,rtol=2e-9)
    assert absorber.opacity(n,1e6,wavelength=171) < 1e-10*expected
    assert absorber.opacity(n,1e6,wavelength=171) > 0.
    assert absorber.opacity(n,1e4,wavelength=913) == 0
    assert absorber.opacity(0.,1e4,wavelength=171) == 0
    # Independently bisect charge neutrality and use all three populations.
    for t in (6000.,10000.,20000.,60000.,1e6):
        lo, hi = 1e-100, 1.2*n
        log_ratios = _saha_log_ratios(t)
        for _ in range(160):
            ne = (lo+hi)/2
            h0,h1,he0,he1,he2 = _saha_populations(log_ratios,ne)
            if ne/n > h1+.1*(he1+2*he2):
                hi = ne
            else:
                lo = ne
        expected = n*(h0*5.16e-20+.1*(he0*9.25e-19+he1*7.17e-19))
        np.testing.assert_allclose(absorber.opacity(n,t,wavelength=171),expected,rtol=2e-8)


def test_partial_coverage_sample_limits_and_units_are_not_hidden():
    fields = constant_slabs([2.,8.], [1000.,3.])
    rays = sm.RaySet(sm.PointSet([[0,.5,.5]]),[1,0,0])
    selection = sm.select_region(fields.mesh, [[0,0,0],[.9,1,1]])
    slots = np.array([0,-1],np.int64)
    partial = replace(fields, selection=selection, slot_of_leaf=slots, _values=fields.values[:1].copy())
    for implementation in ('native','reference'):
        missing = sm.radiative_los(partial,rays,length_unit_cm=1.,implementation=implementation)
        assert missing.intensity.status[0] == sm.LOSStatus.MISSING_COVERAGE
        assert np.isnan(missing.intensity.values[0])
        limited = sm.radiative_los(fields,rays,length_unit_cm=1.,max_samples=1,implementation=implementation)
        assert limited.intensity.status[0] == sm.LOSStatus.SAMPLE_LIMIT
        assert np.isnan(limited.optical_depth.values[0])
    with pytest.raises(ValueError,match='valid halo'):
        sm.radiative_los(replace(fields,valid_halo=0),rays,length_unit_cm=1.)
    with pytest.raises(MemoryError):
        sm.radiative_los(fields,rays,length_unit_cm=1.,memory_limit=1)


def test_euv_cold_screen_attenuates_hot_background_emission():
    # A resolved cool screen exercises H/He absorption that a 1 MK image cannot.
    source, _ = mixed_source(lambda x,y,z:np.array([1+x*0, 1+x*0, 1+x*0]))
    density = sm.prepare(source,['b1'],scheme='exact-phase')
    model = sm.EUV(wavelength=193)
    thermal = sm.thermal_fields(density, 1e6, density_unit_g_cm3=1e-13,
                                temperature_label='hot with cold screen', model=model)
    data = thermal.values.copy()
    for slot, leaf in enumerate(thermal.leaf_ids):
        x = (thermal.mesh.bounds[leaf,0,0]+(np.arange(data.shape[1])+.5-thermal.storage_halo)*
             thermal.mesh.spacing[leaf,0])
        data[slot,...,1] = np.where(x[:,None,None] < .5, 8000., 1e6)
    thermal = replace(thermal,_values=data)
    coefficients = sm.radiation_fields(thermal,model=model,absorption=sm.HHeAbsorption())
    rays = sm.RaySet(sm.PointSet([[0,.5,.5],[2,.5,.5]]),[[1,0,0],[-1,0,0]])
    result = sm.radiative_los(coefficients,rays,length_unit_cm=1e9,subdivisions=16)
    reference = sm.radiative_los(coefficients,rays,length_unit_cm=1e9,subdivisions=16,
                                implementation='reference')
    assert result.complete and reference.complete
    assert result.intensity.values[0] < .1*result.intensity.values[1]
    assert np.all(result.optical_depth.values > 1.)
    np.testing.assert_allclose(result.intensity.values,reference.intensity.values,rtol=1e-11)
