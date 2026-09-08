"""View/tile scheduling preserves each requested image and numerical sum."""
import unittest
import numpy as np
from simesh.analysis import (prepare, PreparedPool, integrate_los,integrate_los_views,
                             orthographic_plane)
from test_prepared import source_fixture


class LOSViewTests(unittest.TestCase):
    def test_equal_images_across_borrowed_tile_order_and_workers(self):
        source,_=source_fixture()
        ready=prepare(source,range(source.mesh.leaf_count),[0,1,2])
        directions=[[.3,.2,1.],[.31,.2,1.],[.3,.21,1.]]
        planes=[orthographic_plane(ready.mesh.lower,ready.mesh.upper,d,(12,12)) for d in directions]
        near=np.zeros((12,12));near[0]=.2
        expected=[integrate_los(ready,p,d,near=near) for p,d in zip(planes,directions)]
        for order in ('view','tile'):
            pool=PreparedPool(source,[0,1,2],4)
            try:
                images=integrate_los_views(pool,planes,directions,near=near,view_order=order,
                                           workers=2,tile_shape=(2,2))
                for image,reference in zip(images,expected):
                    for name in ('values','entry','exit','status','samples'):
                        np.testing.assert_array_equal(getattr(image,name),getattr(reference,name))
                    self.assertFalse(image.direction.flags.writeable)
                with self.assertRaises(MemoryError):
                    integrate_los_views(pool,planes,directions,budget_bytes=pool.controlled_bytes)
            finally:
                pool.close()


if __name__=='__main__':
    unittest.main()
