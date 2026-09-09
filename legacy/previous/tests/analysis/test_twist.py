"""Independent accepted-segment twist and explicit trajectory/retrace evidence."""

import unittest
import numpy as np

from simesh.analysis import prepare, trace, retrace, with_curl, PreparedPool, CurlPool, sample
from test_prepared import source_fixture


class TwistTests(unittest.TestCase):
    def test_helical_stage_diagnostic_and_selected_retracing(self):
        def field(x,y,z):
            return np.array([-(y-5.),x-3.,np.full_like(x,.4)])
        source,_ = source_fixture(field)
        primary = prepare(source,np.arange(source.mesh.leaf_count),[0,1,2])
        bundle = with_curl(primary)
        phases = np.linspace(0,2*np.pi,16,endpoint=False)
        radii = np.linspace(.7,1.3,16)
        seeds = np.ascontiguousarray(np.column_stack((3+radii*np.cos(phases),5+radii*np.sin(phases),np.zeros(16))))
        kwargs = dict(step=.01,max_steps=300,twist=True)
        one = trace(bundle,seeds,**kwargs)
        four = trace(bundle,seeds,workers=4,trajectories=True,**kwargs)
        np.testing.assert_array_equal(one.positions,four.positions)
        np.testing.assert_array_equal(one.twist,four.twist)
        # curl(B)=(0,0,2), radius is constant along each exact field line.
        density = .8/(4*np.pi*(radii*radii+.16))
        expected = density*one.length
        error = np.max(np.abs(one.twist-expected))
        self.assertLess(error,2e-10)
        from test_field_lines import scalar_rk4
        def augmented_rhs(p):
            b = np.array([-(p[1]-5),p[0]-3,.4])
            norm2 = np.dot(b,b)
            return np.r_[b/np.sqrt(norm2),.8/(4*np.pi*norm2)]
        for row in (0,7,15):
            independent = scalar_rk4(np.r_[seeds[row],0.],augmented_rhs,.01,300)
            np.testing.assert_allclose(np.r_[one.positions[row],one.twist[row]],independent,
                                       atol=2e-12,rtol=2e-12)
        refined = trace(bundle,seeds,step=.005,max_steps=600,twist=True)
        refined_error = np.max(np.abs(refined.twist-density*refined.length))
        self.assertLess(refined_error,error/10)
        np.testing.assert_array_equal(four.trajectories[:,0],seeds)
        np.testing.assert_array_equal(four.trajectories[np.arange(16),four.steps],one.positions)
        values,_,valid = sample(bundle.curl,seeds)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(values,np.broadcast_to([0.,0.,2.],values.shape),atol=2e-13,rtol=2e-14)
        pool = PreparedPool(source,[0,1,2],4)
        coupled = CurlPool(pool)
        bounded = trace(coupled,seeds[:4],workers=4,seed_batch=4,**kwargs)
        np.testing.assert_array_equal(bounded.positions,one.positions[:4])
        np.testing.assert_array_equal(bounded.twist,one.twist[:4])
        self.assertGreater(coupled.derived_count,4)
        self.assertLessEqual(coupled.derived_count,pool.prepared_count)
        coupled.close()
        pool.close()
        # Selection follows a completed diagnostic; recomputation is explicit.
        chosen = one.seed_ids[np.argsort(one.twist)[-3:]]
        again = retrace(bundle,one,chosen,workers=4,**kwargs)
        np.testing.assert_array_equal(again.seed_ids,chosen)
        np.testing.assert_array_equal(again.positions,one.positions[chosen])
        np.testing.assert_array_equal(again.trajectories,four.trajectories[chosen])
        self.assertIsNone(one.trajectories)
        with self.assertRaises(MemoryError):
            trace(bundle,seeds,step=.01,max_steps=10**7,trajectories=True,budget_bytes=1024**2)
        print(f"helical twist max errors: h={error:.6g}, h/2={refined_error:.6g}")


if __name__ == "__main__":
    unittest.main()
