"""Global native derivative ownership and repeated sampled-plane composition."""

import gc
import tempfile
from pathlib import Path
import unittest
import numpy as np

from simesh.analysis import prepare, global_curl, curl, Plane, sample_plane
from simesh.amrvac.analysis import prepare_resident
from test_prepared import source_fixture


class GlobalSliceTests(unittest.TestCase):
    def test_global_bounded_resident_memmap_and_planes(self):
        source,fixture = source_fixture()
        mesh = source.mesh
        primary = prepare(source,np.arange(mesh.leaf_count),[0,1,2])
        expected = curl(primary)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"curl.npy"
            output = np.lib.format.open_memmap(path,mode="w+",dtype=float,shape=expected.values.shape)
            global_field = global_curl(source,batch_size=13,output=output)
            np.testing.assert_array_equal(global_field.values,expected.values)
            self.assertEqual(len(global_field.leaf_ids),mesh.leaf_count)
            # Independent whole-field centered stencil on the known valid input.
            b = primary.values
            d = [(b[:,2:,1:-1,1:-1]-b[:,:-2,1:-1,1:-1])/(2*mesh.spacing[:,0,None,None,None,None]),
                 (b[:,1:-1,2:,1:-1]-b[:,1:-1,:-2,1:-1])/(2*mesh.spacing[:,1,None,None,None,None]),
                 (b[:,1:-1,1:-1,2:]-b[:,1:-1,1:-1,:-2])/(2*mesh.spacing[:,2,None,None,None,None])]
            reference = np.stack((d[1][...,2]-d[2][...,1],d[2][...,0]-d[0][...,2],
                                  d[0][...,1]-d[1][...,0]),axis=-1)
            np.testing.assert_array_equal(global_field.values,reference)
            plane = Plane([0.,3.,0.],[6.,0.,0.],[0.,4.,0.],(31,25))
            oblique = Plane([1.,3.,-.5],[4.,0.,1.],[0.,4.,.5],(17,23))
            for geometry in (plane,oblique):
                sliced = sample_plane(global_field,geometry,tile_rows=7)
                self.assertTrue(sliced.valid.all())
                np.testing.assert_allclose(sliced.values,np.broadcast_to([3.,-3.,3.],sliced.values.shape),
                                           atol=2e-13,rtol=2e-14)
                again = sample_plane(global_field,geometry,tile_rows=11)
                np.testing.assert_array_equal(sliced.values,again.values)
            outside = sample_plane(global_field,Plane([-20.,0.,0.],[1.,0.,0.],[0.,1.,0.],(3,4)))
            self.assertFalse(outside.valid.any())
            self.assertTrue(np.isnan(outside.values).all())
            # File remains reusable as a complete native derived product.
            output.flush()
            np.testing.assert_array_equal(np.load(path,mmap_mode="r"),global_field.values)

        resident = prepare_resident(mesh,fixture.root_shape,fixture.forest_flags,fixture.backing,source.fields)
        resident_curl = curl(resident)
        np.testing.assert_allclose(resident_curl.values,expected.values,atol=2e-13,rtol=2e-14)
        view = resident.values[0]
        preserved = view.copy()
        del resident
        gc.collect()
        # NumPy views keep C-owned canonical backing alive beyond the descriptor.
        np.testing.assert_array_equal(view,preserved)
        self.assertGreater(resident_curl.nbytes,0)


if __name__ == "__main__":
    unittest.main()
