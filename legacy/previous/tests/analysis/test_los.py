"""Independent ray depth, AMR coverage and scalar quadrature evidence."""

import unittest
from dataclasses import replace
import numpy as np

from simesh.analysis import prepare, PreparedPool, Plane, integrate_los, orthographic_plane, LOSStatus
from test_prepared import source_fixture


def origins_for(plane):
    nx,ny = plane.shape
    u = (np.arange(nx)+.5)/nx
    v = (np.arange(ny)+.5)/ny
    return plane.origin+u[:,None,None]*plane.u+v[None,:,None]*plane.v


def independent_depth(mesh,origins,direction,near,far):
    direction = np.asarray(direction,dtype=float)
    direction /= np.linalg.norm(direction)
    near,far = np.broadcast_to(near,origins.shape[:2]),np.broadcast_to(far,origins.shape[:2])
    output = np.zeros(origins.shape[:2])
    for index in np.ndindex(output.shape):
        point = origins[index]
        first,last = float(near[index]),float(far[index])
        for a in range(3):
            if direction[a] == 0:
                if not mesh.lower[a]<=point[a]<mesh.upper[a]:
                    first,last = 1.,0.
                    break
            else:
                pair = sorted(((mesh.lower[a]-point[a])/direction[a],(mesh.upper[a]-point[a])/direction[a]))
                first,last = max(first,pair[0]),min(last,pair[1])
        output[index] = max(0.,last-first)
    return output


class LOSTests(unittest.TestCase):
    def test_constant_views_depths_and_parallel_bounded_coverage(self):
        source,_ = source_fixture(lambda x,y,z: np.array([np.full_like(x,2.),0*x,0*x]))
        mesh = source.mesh
        fields = prepare(source,np.arange(mesh.leaf_count),[0])
        cases = ((Plane([-5.,1.,-2.],[0.,8.,0.],[0.,0.,6.],(8,9)),[1.,0.,0.],0.,np.inf),
                 (Plane([11.,9.,4.],[0.,-8.,0.],[0.,0.,-6.],(8,9)),[-1.,-.2,-.1],0.,np.inf),
                 (Plane([-5.,0.,-3.],[0.,10.,0.],[0.,0.,8.],(8,9)),[1.,.4,.3],
                  np.linspace(0,3,72).reshape(8,9),np.linspace(3,18,72).reshape(8,9)))
        for plane,direction,near,far in cases:
            expected = independent_depth(mesh,origins_for(plane),direction,near,far)
            result = integrate_los(fields,plane,direction,near=near,far=far,tile_shape=(3,4))
            self.assertTrue(result.complete)
            np.testing.assert_allclose(result.depth,expected,atol=2e-12,rtol=2e-13)
            np.testing.assert_allclose(result.values,2*expected,atol=2e-12,rtol=2e-13)
            parallel = integrate_los(fields,plane,direction,near=near,far=far,workers=4,tile_shape=(5,7))
            np.testing.assert_array_equal(result.values,parallel.values)
        plane,direction,near,far = cases[1]
        reference = integrate_los(fields,plane,direction)
        pool = PreparedPool(source,[0],4)
        bounded = integrate_los(pool,plane,direction,workers=4,tile_shape=(2,2))
        self.assertTrue(bounded.complete)
        np.testing.assert_array_equal(bounded.values,reference.values)
        self.assertGreater(pool.prepared_count,4)
        pool.close()
        upper = Plane([9.,1.,-2.],[0.,8.,0.],[0.,0.,6.],(3,4))
        empty = integrate_los(fields,upper,[0.,1.,0.])
        self.assertTrue(empty.complete)
        self.assertTrue(np.all(empty.status==LOSStatus.EMPTY))
        self.assertTrue(np.all(empty.values==0.))

    def test_affine_quadrature_and_explicit_incomplete_results(self):
        source,_ = source_fixture(lambda x,y,z: np.array([x+4.,0*x,0*x]))
        mesh = source.mesh
        fields = prepare(source,np.arange(mesh.leaf_count),[0])
        plane = Plane([0.,2.5,-.5],[0.,3.,0.],[0.,0.,1.],(7,8))
        direction = np.array([1.,.3,.1])
        direction /= np.linalg.norm(direction)
        near,far = .2,2.5
        midpoints = origins_for(plane)+direction*((near+far)/2)
        expected = (midpoints[...,0]+4)*(far-near)
        result = integrate_los(fields,plane,direction,near=near,far=far)
        self.assertTrue(result.complete)
        np.testing.assert_allclose(result.values,expected,atol=2e-12,rtol=2e-13)
        limited = integrate_los(fields,plane,direction,near=near,far=far,max_samples=1)
        self.assertFalse(limited.complete)
        self.assertTrue(np.isnan(limited.values).all())
        missing = integrate_los(prepare(source,[0],[0]),plane,direction,near=near,far=far)
        self.assertTrue(np.all(missing.status==LOSStatus.MISSING_COVERAGE))
        invalid = replace(fields,values=np.full_like(fields.values,np.nan))
        bad = integrate_los(invalid,plane,direction,near=near,far=far)
        self.assertTrue(np.all(bad.status==LOSStatus.NONFINITE_SCALAR))

    def test_gaussian_reconstruction_reference(self):
        from analysis_core.los_reference import ray_integral
        def field(x,y,z):
            return np.array([2+.05*x*y+.03*y*z+.01*x*y*z,0*x,0*x])
        source,_ = source_fixture(field)
        mesh = source.mesh
        fields = prepare(source,np.arange(mesh.leaf_count),[0])
        direction = [1.,.4,.3]
        plane = orthographic_plane(mesh.lower,mesh.upper,direction,(7,8))
        exact = integrate_los(fields,plane,direction,quadrature="gauss2")
        self.assertTrue(exact.complete)
        origins = origins_for(plane)
        reference = np.array([ray_integral(fields,p,exact.direction) for p in origins.reshape(-1,3)])
        np.testing.assert_allclose(exact.values.ravel(),reference[:,0],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(exact.depth.ravel(),reference[:,1],atol=2e-12,rtol=2e-13)
        parallel = integrate_los(fields,plane,direction,quadrature="gauss2",workers=4,tile_shape=(3,5))
        np.testing.assert_array_equal(exact.values,parallel.values)
        pool = PreparedPool(source,[0],8)
        bounded = integrate_los(pool,plane,direction,quadrature="gauss2",workers=4)
        np.testing.assert_array_equal(exact.values,bounded.values)
        pool.close()
        coarse = integrate_los(fields,plane,direction,quadrature="midpoint",step_fraction=.5)
        fine = integrate_los(fields,plane,direction,quadrature="midpoint",step_fraction=.125)
        coarse_error = np.linalg.norm(coarse.values-exact.values)
        fine_error = np.linalg.norm(fine.values-exact.values)
        self.assertLess(fine_error,coarse_error)
        print(f"LOS reconstruction L2 errors: midpoint .5={coarse_error:.6g}, .125={fine_error:.6g}")


if __name__ == '__main__':
    unittest.main()
