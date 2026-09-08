"""Independent physical normalization and nonlinear thermal LOS checks."""
from dataclasses import replace
import unittest
import numpy as np

from simesh.analysis import (AIA171, CoronalComposition, FieldDefinition, prepare,
    thermal_fields, emissivity_fields, integrate_thermal_los, Plane, LOSStatus)
from simesh.analysis.thermal import PROTON_MASS_G, BOLTZMANN_ERG_K
from simesh.analysis._aia171_table import LOG_T, RESPONSE
from test_prepared import source_fixture


class ThermalTests(unittest.TestCase):
    def test_table_eos_normalization_and_endpoints(self):
        model = AIA171()
        np.testing.assert_allclose(model.response(10.**np.array(LOG_T)), RESPONSE, rtol=2e-14)
        self.assertAlmostEqual(float(model.response(1e9))/RESPONSE[-1], 1., places=13)
        np.testing.assert_array_equal(model.response([1e3, 1e10]), [0., 0.])
        i = 36
        self.assertAlmostEqual(float(model.response(10**((LOG_T[i]+LOG_T[i+1])/2)))/
                               np.sqrt(RESPONSE[i]*RESPONSE[i+1]), 1., places=13)
        rho = 1.4*PROTON_MASS_G*1e9
        p = 2.3*1e9*BOLTZMANN_ERG_K*1e6
        self.assertAlmostEqual(CoronalComposition().temperature(rho,p)/1e6, 1.)
        compatibility = AIA171("amrvac-hydrogen")
        self.assertAlmostEqual(float(model.emissivity(rho,1e6)/compatibility.emissivity(rho,1e6)),1.44)
        for t in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError): model.response(t)

    def test_isothermal_full_domain_and_external_temperature(self):
        source,_ = source_fixture(lambda x,y,z: np.array([np.ones_like(x),np.full_like(x,1e6),0*x]))
        source = replace(source, fields=(FieldDefinition("rho","code"),FieldDefinition("T","K"),FieldDefinition("unused","code")))
        ids = np.arange(source.mesh.leaf_count)[::-1]
        density = prepare(source,ids,[0])
        temperature = prepare(source,ids[::-1],[1])
        rho_unit = 1.4*PROTON_MASS_G*1e9
        state = thermal_fields(density,temperature,density_unit_g_cm3=rho_unit,temperature_label="manufactured constant K")
        iso = thermal_fields(density,1e6,density_unit_g_cm3=rho_unit,temperature_label="isothermal 1 MK")
        np.testing.assert_array_equal(state.values,iso.values)
        plane = Plane([-5.,1.,-2.],[0.,8.,0.],[0.,0.,6.],(3,4))
        for direction in ([1.,0.,0.],[1.,.3,.2]):
            for order in ("emissivity-first","thermodynamics-first"):
                r = integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,order=order)
                self.assertTrue(r.complete)
                expected = AIA171().emissivity(rho_unit,1e6)*r.depth*1e8
                np.testing.assert_allclose(r.values,expected,rtol=3e-13,atol=1e-10)
                self.assertEqual(r.scalar_units,"DN s^-1 pixel^-1")
        with self.assertRaises(ValueError):
            thermal_fields(density,None,density_unit_g_cm3=rho_unit,temperature_label="missing")
        missing = replace(state,slot_of_leaf=np.full_like(state.slot_of_leaf,-1))
        r = integrate_thermal_los(missing,plane,[1.,0.,0.],length_unit_cm=1e8)
        self.assertTrue(np.all(r.status==LOSStatus.MISSING_COVERAGE))
        r = integrate_thermal_los(state,plane,[1.,0.,0.],length_unit_cm=1e8,max_samples=1)
        self.assertFalse(r.complete)
        self.assertTrue(np.isnan(r.values).all())

    def test_nonlinear_response_orders_against_continuum(self):
        # Entire ray remains away from physical boundary extension and AMR
        # limiter extrema. An affine T crosses several tabular power-law pieces.
        def fields(x,y,z):
            return np.array([1+.12*x, 5.5e5+1.1e5*x, 0*x])
        source,_ = source_fixture(fields)
        source = replace(source, fields=(FieldDefinition("rho","code"),FieldDefinition("T","K"),FieldDefinition("unused","code")))
        ready = prepare(source,np.arange(source.mesh.leaf_count),[0,1])
        state = thermal_fields(ready,ready,density_unit_g_cm3=1.4*PROTON_MASS_G*1e9,
                               temperature_component=1,temperature_label="manufactured affine rho,T")
        plane = Plane([0.,2.5,-.5],[0.,3.,0.],[0.,0.,1.],(2,3))
        d = np.array([1.,.3,.1]); d /= np.linalg.norm(d)
        # Independent dense midpoint quadrature of analytic thermodynamics;
        # response is evaluated as explicit tabular power laws, not API calls.
        s = .2+(np.arange(262144)+.5)*(2.3/262144)
        x = d[0]*s
        t = 5.5e5+1.1e5*x
        ii = np.searchsorted(10.**np.array(LOG_T),t)-1
        exponent = np.log(np.array(RESPONSE)[ii+1]/np.array(RESPONSE)[ii])/np.log(10.**(np.array(LOG_T)[ii+1]-np.array(LOG_T)[ii]))
        response = np.array(RESPONSE)[ii]*(t/10.**np.array(LOG_T)[ii])**exponent
        expected = np.mean((1.2e9*(1+.12*x))**2*response)*2.3*1e8
        errors=[]
        for subdivisions in (1,4,16):
            r = integrate_thermal_los(state,plane,d,length_unit_cm=1e8,near=.2,far=2.5,subdivisions=subdivisions)
            self.assertTrue(r.complete)
            errors.append(float(np.max(np.abs(r.values/expected-1))))
        pre = integrate_thermal_los(state,plane,d,length_unit_cm=1e8,near=.2,far=2.5,order="emissivity-first")
        pre_error=float(np.max(np.abs(pre.values/expected-1)))
        self.assertLess(errors[-1],1e-5)
        self.assertLess(errors[-1],errors[0])
        self.assertGreater(pre_error,errors[-1]*10)
        print("thermal continuum relative errors",errors,"node response first",pre_error)

    def test_native_reference_parallel_and_failure_conformance(self):
        from simesh.analysis import orthographic_plane
        source,_ = source_fixture(lambda x,y,z:np.array([2+.03*x*y, 8e5+1e4*x*y+2e4*z, 0*x]))
        source = replace(source,fields=(FieldDefinition('rho','code'),FieldDefinition('T','K'),FieldDefinition('unused','code')))
        ready = prepare(source,np.arange(source.mesh.leaf_count),[0,1])
        state = thermal_fields(ready,ready,density_unit_g_cm3=1.4*PROTON_MASS_G*1e9,
                               temperature_component=1,temperature_label='manufactured nonaffine')
        mesh = state.mesh
        for direction in ([1.,0.,0.],[1.,.3,.2],[-1.,-.8,.6],[0.,0.,-1.]):
            plane = orthographic_plane(mesh.lower,mesh.upper,direction,(7,9))
            near = np.linspace(0.,.5,63).reshape(7,9)
            reference = integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,
                near=near,implementation='reference')
            native = integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,near=near)
            np.testing.assert_array_equal(native.status,reference.status)
            np.testing.assert_allclose(native.values,reference.values,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(native.depth,reference.depth,rtol=2e-13,atol=2e-12)
            for workers in (2,4):
                parallel = integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,near=near,workers=workers)
                np.testing.assert_array_equal(parallel.values,native.values)
                np.testing.assert_array_equal(parallel.samples,native.samples)
                np.testing.assert_array_equal(parallel.status,native.status)
        plane = Plane([0.,2.5,-.5],[0.,3.,0.],[0.,0.,1.],(3,4))
        for data in (np.full_like(state.values,np.nan),np.full_like(state.values,-1.)):
            invalid = replace(state,values=data)
            reference = integrate_thermal_los(invalid,plane,[1.,0.,0.],length_unit_cm=1e8,implementation='reference')
            native = integrate_thermal_los(invalid,plane,[1.,0.,0.],length_unit_cm=1e8,workers=4)
            np.testing.assert_array_equal(native.status,reference.status)
            self.assertTrue(np.isnan(native.values).all())
        with self.assertRaises(ValueError):
            integrate_thermal_los(state,plane,[1,0,0],length_unit_cm=1e8,implementation='reference',workers=2)
        with self.assertRaises(MemoryError):
            integrate_thermal_los(state,plane,[1,0,0],length_unit_cm=1e8,budget_bytes=1)


if __name__ == "__main__": unittest.main()
