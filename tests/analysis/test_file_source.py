"""Actual v5 file-source composition, field mapping and detached lifetime."""

import tempfile
from pathlib import Path
import unittest
import numpy as np

from simesh.analysis import open_source,open_prepared,prepare,sample,PreparedPool
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
