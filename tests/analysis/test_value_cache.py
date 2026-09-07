"""Numerical cache parity, failure publication and source-lifecycle boundaries."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np

from simesh.analysis import open_source, prepare, PreparedPool
from simesh.analysis.value_cache import InteriorValueCache
from simesh_rewrite.blockio import array_block_reader, read_blocks_into
from lfe_001 import write_fixture_dat
from test_prepared import source_fixture


class ValueCacheTests(unittest.TestCase):
    def test_field_selection_snapshot_and_closed_or_changed_source(self):
        _, fixture = source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.dat'
            write_fixture_dat(path,fixture)
            with open_source(path,value_cache_capacity=128) as source:
                first = prepare(source,[3,7],[0,1,2])
                again = prepare(source,[7,3],[0,1,2])
                np.testing.assert_array_equal(first.values,again.values[::-1])
                self.assertGreater(again.preparation_stats['value_cache_hits'],0)
                self.assertEqual(again.preparation_stats['value_cache_misses'],0)
                selected = prepare(source,[3,7],[2,0])
                np.testing.assert_array_equal(selected.values,first.values[...,[2,0]])
                self.assertGreater(selected.preparation_stats['value_cache_misses'],0)
                pool = PreparedPool(source,[0,1,2],2)
                with pool.borrow([3,7]):
                    pass
                # Same geometry, changed values: cached complete ghosts cannot
                # bypass the immutable file lifecycle check, even on all hits.
                fixture.backing[:] *= 2
                write_fixture_dat(path,fixture)
                with self.assertRaises(OSError):
                    with pool.borrow([3,7]):
                        pass
                with self.assertRaises(OSError):
                    prepare(source,[3,7],[0,1,2])
            with self.assertRaises(OSError):
                with pool.borrow([3,7]):
                    pass
            pool.close()
            with open_source(path,value_cache_capacity=128) as new:
                updated = prepare(new,[3,7],[0,1,2])
                self.assertIsNot(updated.source,first.source)
                np.testing.assert_array_equal(updated.values,2*first.values)

    def test_bounded_eviction_and_failed_read_does_not_publish(self):
        backing = np.arange(6*2*4**3,dtype=float).reshape(6,2,4,4,4)
        original = array_block_reader(backing)
        broken = False
        def read(*args):
            if broken:
                args[-2].fill(-123)
                raise OSError('injected failed miss')
            original.read_into(*args)
        cache = InteriorValueCache(replace(original,read_into=read),2)
        def load(ids):
            out = np.empty((len(ids),2,4,4,4))
            read_blocks_into(cache.reader,np.zeros(3,dtype=np.int64),np.full(3,4,dtype=np.int64),
                np.array(ids,dtype=np.int64),np.array([0,1],dtype=np.int64),out,np.zeros(3,dtype=np.int64))
            return out
        np.testing.assert_array_equal(load([0,1]),backing[[0,1]])
        broken = True
        with self.assertRaises(OSError):
            load([1,2])
        self.assertEqual(cache.directory[2],-1)
        broken = False
        np.testing.assert_array_equal(load([1,2,3,4]),backing[[1,2,3,4]])
        self.assertEqual(np.count_nonzero(cache.directory>=0),2)
        np.testing.assert_array_equal(load([4,0]),backing[[4,0]])


if __name__=='__main__':
    unittest.main()
