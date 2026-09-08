"""Optional native dispatch preserves independent row arithmetic and miss state."""
import unittest
import numpy as np
from simesh.analysis import prepare, PreparedPool, trace, integrate_los, orthographic_plane
from simesh.utils.lib.analysis.native import openmp_build_info
from test_prepared import source_fixture


class ExecutionTests(unittest.TestCase):
    def test_native_backend_conformance_or_explicit_unavailability(self):
        source,_ = source_fixture(lambda x,y,z: np.array([-y,x,np.ones_like(z)]))
        mesh=source.mesh
        seeds=np.ascontiguousarray(mesh.lower+np.random.default_rng(8).random((41,3))*(mesh.upper-mesh.lower))
        seeds[0]=mesh.upper+1
        ready=prepare(source,range(mesh.leaf_count),[0,1,2])
        opts=dict(step=.15,max_steps=32,seed_batch=8,twist=True,trajectories=True)
        reference=trace(ready,seeds,**opts)
        direction=[.3,.2,1.]
        plane=orthographic_plane(mesh.lower,mesh.upper,direction,(12,12))
        expected=integrate_los(ready,plane,direction,tile_shape=(2,4))
        pool=PreparedPool(source,[0,1,2],8)
        try:
            threaded=trace(pool,seeds,workers=4,schedule='dynamic',**opts)
            for name in ('positions','length','steps','termination','samples','twist','trajectories'):
                np.testing.assert_array_equal(getattr(threaded,name),getattr(reference,name))
            image=integrate_los(pool,plane,direction,tile_shape=(2,4),workers=4,schedule='dynamic')
            for name in ('values','entry','exit','status','samples'):
                np.testing.assert_array_equal(getattr(image,name),getattr(expected,name))
        finally:
            pool.close()
        if not openmp_build_info()['enabled']:
            with self.assertRaises(RuntimeError):
                trace(ready,seeds,backend='openmp',**opts)
            return
        for schedule in ('static','dynamic'):
            for workers in (1,2,4):
                pool=PreparedPool(source,[0,1,2],8)
                try:
                    actual=trace(pool,seeds,workers=workers,backend='openmp',schedule=schedule,**opts)
                    for name in ('seed_ids','seeds','positions','length','steps','termination','samples',
                                 'twist','trajectories'):
                        np.testing.assert_array_equal(getattr(actual,name),getattr(reference,name))
                    image=integrate_los(pool,plane,direction,tile_shape=(2,4),
                                        workers=workers,backend='openmp',schedule=schedule)
                    for name in ('values','entry','exit','status','samples'):
                        np.testing.assert_array_equal(getattr(image,name),getattr(expected,name))
                finally:
                    pool.close()


if __name__=='__main__':
    unittest.main()
