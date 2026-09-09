"""Actual v5 file-source composition, field mapping and detached lifetime."""

import tempfile
from pathlib import Path
import unittest
import numpy as np

from simesh.analysis import open_source,open_prepared,prepare,prepare_region,sample,PreparedPool,trace,Termination
from simesh.amrvac.datio import get_metadata,write_header,write_forest_tree
from lfe_001 import write_fixture_dat
from test_prepared import source_fixture


def write_staggered_fixture(path,fixture):
    ordinary=path.with_name('ordinary.dat')
    write_fixture_dat(ordinary,fixture)
    header,flags,tree=get_metadata(str(ordinary))
    header['staggered']=True
    block=np.asarray(fixture.block_counts)
    record_bytes=24+fixture.backing.shape[1]*int(np.prod(block))*8+3*int(np.prod(block+1))*8
    tree=(tree[0],tree[1],header['offset_blocks']+np.arange(len(tree[0]),dtype=np.int64)*record_bytes)
    with path.open('wb') as stream:
        write_header(stream,header)
        write_forest_tree(stream,header,flags.astype(np.int32),tree)
        stream.seek(header['offset_blocks'])
        for block_values in fixture.backing:
            stream.write(np.zeros(6,dtype=np.int32).tobytes())
            stream.write(block_values.transpose(0,3,2,1).tobytes())
            stream.write(np.full((3,*(block+1)),9e99).tobytes())


class FileSourceTests(unittest.TestCase):
    def test_region_preserves_original_support_and_detaches(self):
        _,fixture=source_fixture(lambda x,y,z: np.array([np.sin(x)+y,y*y+z,np.cos(z)+x]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'region.dat'
            write_staggered_fixture(path,fixture)
            with open_source(path,field_names=['b1','b2','b3']) as source:
                mesh=source.mesh
                leaf=int(np.argmax(fixture.node_levels[fixture.leaf_node_ids]))
                point=mesh.bounds[leaf].mean(axis=0)
                point[0]=mesh.bounds[leaf,1,0]
                bounds=np.array([point-.01*mesh.spacing[leaf],point+.01*mesh.spacing[leaf]])
                full=prepare(source,np.arange(mesh.leaf_count),[0,1,2])
                regional=prepare_region(source,bounds,[0,1,2])
                self.assertLess(len(regional.leaf_ids),mesh.leaf_count)
                np.testing.assert_array_equal(regional.values,full.values[regional.leaf_ids])
                # A thin box crossing a leaf face must include both owners,
                # even if it contains no original cell centers.
                points=np.array([point,np.nextafter(point,mesh.lower),np.nextafter(point,mesh.upper)])
                self.assertTrue(sample(regional,points)[2].all())
                np.testing.assert_array_equal(sample(regional,points)[0],sample(full,points)[0])
                outside=np.flatnonzero(regional.slot_of_leaf<0)[0]
                outside_seed=np.ascontiguousarray(mesh.bounds[[outside]].mean(axis=1))
                for invalid in ((point,point),([100.,100.,100.],[101.,101.,101.])):
                    with self.assertRaises(ValueError):
                        prepare_region(source,invalid,[0,1,2])
            loaded=open_prepared(path,field_names=['b1','b2','b3'],bounds=bounds)
            path.unlink()
            np.testing.assert_array_equal(loaded.values,regional.values)
            np.testing.assert_array_equal(loaded.leaf_ids,regional.leaf_ids)
            self.assertTrue(sample(loaded,points)[2].all())
            result=trace(loaded,outside_seed,step=.01,max_steps=2)
            self.assertEqual(result.termination[0],Termination.MISSING_COVERAGE)
            self.assertEqual(result.steps[0],0)

    def test_staggered_file_native_selection_and_resident_workflow(self):
        _,fixture=source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'staggered.dat'
            write_staggered_fixture(path,fixture)
            ids=np.array([15,2,37],dtype=np.int64)
            with open_source(path,field_names=['b3','b1'],field_units='code_B') as source:
                interiors=prepare(source,ids,[1,0],halo=0)
                np.testing.assert_array_equal(interiors.values,
                    np.moveaxis(fixture.backing[ids][:,[0,2]],1,-1))
                product=prepare(source,ids,[1,0])
                pool=PreparedPool(source,[0,1],2)
                with pool.borrow([1]):
                    pass
                points=source.mesh.bounds[ids].mean(axis=1)
                expected=sample(product,points)[0]
            np.testing.assert_array_equal(sample(product,points)[0],expected)
            with self.assertRaises(OSError):
                with pool.borrow([5]):
                    pass
            pool.close()
            resident=open_prepared(path,field_names=['b1','b2','b3'])
            np.testing.assert_array_equal(resident.interior(),np.moveaxis(fixture.backing,1,-1))
            self.assertTrue(sample(resident,points)[2].all())
            with self.assertRaises(MemoryError):
                open_prepared(path,field_names=['b1','b2','b3'],budget_bytes=1)


if __name__=='__main__':
    unittest.main()
