"""Scientific and ownership evidence for independent native RK4 seeds."""

import unittest
import numpy as np

from simesh.analysis import prepare, trace, sample, PreparedPool, Termination
from test_prepared import source_fixture


def scalar_rk4(seed, rhs, step, count):
    x = seed.copy()
    for _ in range(count):
        k1 = rhs(x)
        k2 = rhs(x+step*.5*k1)
        k3 = rhs(x+step*.5*k2)
        k4 = rhs(x+step*k3)
        x = x+(step/6.)*(((k1+2.*k2)+2.*k3)+k4)
    return x


class FieldLineTests(unittest.TestCase):
    def test_leaf_face_crossings_match_unhinted_sampling(self):
        source,_ = source_fixture(lambda x,y,z: np.array([1.+.1*np.sin(y),.3+.1*np.cos(z),.2+.05*np.sin(x)]))
        fields = prepare(source,np.arange(source.mesh.leaf_count),[0,1,2])
        mesh = source.mesh
        faces = mesh.bounds.mean(axis=1)
        faces[:,0] = mesh.bounds[:,0,0]
        safe = np.all((faces>mesh.lower+.5)&(faces<mesh.upper-.5),axis=1)
        faces = faces[safe][::max(1,np.count_nonzero(safe)//12)]
        seeds = np.ascontiguousarray(np.concatenate((faces,np.nextafter(faces,mesh.lower),
                                                    np.nextafter(faces,mesh.upper))))
        step,count = .03,8
        for direction in (-1,1):
            def rhs(point):
                b = sample(fields,point[None,:])[0][0]
                norm = np.hypot(np.hypot(b[0],b[1]),b[2])
                return direction*b/norm
            reference = np.array([scalar_rk4(seed,rhs,step,count) for seed in seeds])
            result = trace(fields,seeds,step=step,max_steps=count,direction=direction,workers=4)
            np.testing.assert_array_equal(result.steps,np.full(len(seeds),count))
            np.testing.assert_allclose(result.positions,reference,rtol=2e-12,atol=2e-12)

    def test_constant_boundary_null_and_chunk_identity(self):
        source,_ = source_fixture(lambda x,y,z: np.array([np.ones_like(x),0*x,0*x]))
        fields = prepare(source, np.arange(source.mesh.leaf_count), [0,1,2])
        seeds = np.array([[-3.,4.,0.], [2.,4.,0.], [8.9,4.,0.], [9.,4.,0.]])
        ids = np.array([101,17,35,-2], dtype=np.int64)
        result = trace(fields, seeds, seed_ids=ids, step=.125, max_steps=20, workers=4, seed_batch=2)
        np.testing.assert_array_equal(result.seed_ids,ids)
        np.testing.assert_array_equal(result.termination,[1,1,3,6])
        np.testing.assert_allclose(result.positions[:2], seeds[:2]+[2.5,0,0], atol=1e-14)
        self.assertFalse(result.localized_endpoint)
        self.assertEqual(result.length[2],0.)
        capped = trace(fields, seeds[:2], step=.2, max_steps=100, max_length=.55, direction=-1)
        self.assertEqual(capped.termination[0],Termination.DOMAIN_EXIT)
        self.assertEqual(capped.termination[1],Termination.MAX_LENGTH)
        self.assertEqual(capped.length[1],.55)
        limited = trace(prepare(source,[0],[0,1,2]), seeds[1:2],step=.1,max_steps=1)
        self.assertEqual(limited.termination[0],Termination.MISSING_COVERAGE)
        null_source,_ = source_fixture(lambda x,y,z: np.array([0*x,0*x,0*x]))
        null = prepare(null_source, [0], [0,1,2])
        seed = null.mesh.bounds[[0]].mean(axis=1)
        self.assertEqual(trace(null,seed,step=.1).termination[0],Termination.NULL_FIELD)
        # Nonfinite input and an overflowing norm have distinct numerical status;
        # neither silently turns into a zero tangent or an accepted step.
        from dataclasses import replace
        for value, reason in ((np.nan,Termination.NONFINITE_FIELD),
                              (np.finfo(float).max,Termination.UNREPRESENTABLE_NORM)):
            payload = np.full_like(null.values,value)
            poisoned = replace(null,values=payload)
            output = trace(poisoned,seed,step=.1)
            self.assertEqual(output.termination[0],reason)
            self.assertEqual(output.steps[0],0)

    def test_helical_accuracy_parallel_and_bounded_stage_continuation(self):
        # Centered well inside the domain; native affine B reconstruction is
        # independent of the RK reference, including interior AMR interfaces.
        def field(x,y,z):
            return np.array([-(y-5.),x-3.,np.full_like(x,.4)])
        source,_ = source_fixture(field)
        fields = prepare(source,np.arange(source.mesh.leaf_count),[0,1,2])
        phases = np.linspace(0,2*np.pi,32,endpoint=False)
        seeds = np.ascontiguousarray(np.column_stack((3.+np.cos(phases),5.+np.sin(phases),np.zeros(32))))
        h, count = .02, 150
        kwargs = dict(step=h,max_steps=count,seed_batch=16)
        serial = trace(fields,seeds,**kwargs)
        parallel = trace(fields,seeds,workers=4,**kwargs)
        for key in ("positions","length","steps","termination"):
            np.testing.assert_array_equal(getattr(serial,key),getattr(parallel,key))
        self.assertTrue(np.all(serial.termination==Termination.MAX_STEPS))
        norm = np.sqrt(1.+.4**2)
        angle = phases+(h*count)/norm
        expected = np.column_stack((3.+np.cos(angle),5.+np.sin(angle),np.full(32,h*count*.4/norm)))
        error = np.max(np.abs(serial.positions-expected))
        def rhs(p):
            b = np.array([-(p[1]-5.),p[0]-3.,.4])
            return b/np.sqrt(np.dot(b,b))
        reference = np.array([scalar_rk4(seed,rhs,h,count) for seed in seeds])
        np.testing.assert_allclose(serial.positions,reference,atol=2e-12,rtol=2e-12)
        refined = trace(fields,seeds,step=h/2,max_steps=count*2,workers=1)
        refined_error = np.max(np.abs(refined.positions-expected))
        self.assertLess(refined_error,error/10)
        pool = PreparedPool(source,[0,1,2],4)
        bounded = trace(pool,seeds[:8],step=h,max_steps=count,workers=4,seed_batch=4)
        np.testing.assert_array_equal(bounded.positions,serial.positions[:8])
        np.testing.assert_array_equal(bounded.steps,serial.steps[:8])
        self.assertGreater(bounded.misses.sum(),0)
        self.assertGreater(pool.prepared_count,pool.capacity)
        pool.close()
        print(f"helical endpoint max errors: h={error:.6g}, h/2={refined_error:.6g}")


if __name__ == "__main__":
    unittest.main()
