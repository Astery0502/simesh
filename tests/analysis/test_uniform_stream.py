"""Bounded uniform delivery through native point/plane consumption."""

import unittest
import numpy as np
from simesh.analysis import PreparedPool,iter_uniform
from simesh.amrvac.analysis import prepare_resident
from test_prepared import source_fixture


class UniformStreamTests(unittest.TestCase):
    def test_bounded_slabs_match_canonical_uniform_and_keep_outputs(self):
        source,fixture=source_fixture()
        mesh=source.mesh
        resident=prepare_resident(mesh,fixture.root_shape,fixture.forest_flags,fixture.backing,source.fields)
        shape=np.array([17,13,9],dtype=np.uint32)
        expected=np.empty((3,*shape))
        resident.owner.uniform_grid_linear(expected,shape,mesh.lower.copy(),mesh.upper.copy(),np.arange(3,dtype=np.uint32))
        output=np.empty((*shape,3))
        pool=PreparedPool(source,[0,1,2],4)
        first_slab=preserved=None
        for iz,slab in iter_uniform(pool,shape,tile_rows=3):
            self.assertTrue(slab.valid.all())
            output[:,:,iz]=slab.values
            if iz==0:
                first_slab=slab
                preserved=slab.values.copy()
        pool.close()
        np.testing.assert_allclose(output,np.moveaxis(expected,0,-1),atol=2e-12,rtol=2e-12)
        np.testing.assert_array_equal(first_slab.values,preserved)


if __name__=='__main__':
    unittest.main()
