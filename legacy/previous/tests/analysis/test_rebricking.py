"""Coverage and mapped F/D support protections for the bounded E1 probe."""
import unittest
import numpy as np
from simesh.analysis import prepare, sample, curl
from analysis_core.rebricking import build_bricks,pack_bricks,sample_bricks,curl_bricks
from test_prepared import source_fixture


class RebrickTests(unittest.TestCase):
    def test_coverage_two_halo_values_and_mapped_consumption(self):
        source,_ = source_fixture()
        ids = np.arange(source.mesh.leaf_count)
        native = prepare(source,ids,[0,1,2])
        points = source.mesh.bounds.mean(axis=1)
        expected = curl(native).interior()
        for shape in ((1,1,1),(2,1,1),(2,2,2)):
            p = build_bricks(source.mesh,shape)
            np.testing.assert_array_equal(np.sort(p.members),ids)
            self.assertEqual(int(np.prod(p.shapes,axis=1).sum()),len(ids)*int(np.prod(source.mesh.block_shape)))
            fields = pack_bricks(p,native)
            for leaf in ids:
                np.testing.assert_array_equal(fields.window(leaf,halo=2),native.values[leaf])
            np.testing.assert_allclose(sample_bricks(fields,points)[0],sample(native,points)[0],rtol=1e-13,atol=1e-13)
            np.testing.assert_array_equal(curl_bricks(fields,ids),expected)


if __name__=='__main__': unittest.main()
