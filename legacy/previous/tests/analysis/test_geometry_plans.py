"""Retained geometric actions must preserve values and independence from fields."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from simesh.analysis import open_source, prepare, sample, curl
from simesh.analysis.geometry_plans import build_fill_plan
from test_prepared import source_fixture
from lfe_001 import write_fixture_dat


class GeometryPlanTests(unittest.TestCase):
    def test_plan_with_value_cache_and_file_lifecycle(self):
        _, fixture = source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.dat"
            write_fixture_dat(path,fixture)
            with open_source(path,value_cache_capacity=128) as source:
                plan = build_fill_plan(source,[3,7])
                first = plan.prepare(source,[2,0])
                warm = plan.prepare(source,[2,0])
                np.testing.assert_array_equal(first.values,warm.values)
                self.assertGreater(first.preparation_stats["read_value_bytes"],0)
                self.assertEqual(warm.preparation_stats["read_value_bytes"],0)
                self.assertGreater(warm.preparation_stats["requested_value_bytes"],0)
                selected = plan.prepare(source,[0])
                expected = prepare(source,plan.leaf_ids,[0])
                np.testing.assert_array_equal(selected.values,first.values[...,[1]])
                np.testing.assert_array_equal(selected.values,expected.values)
                fixture.backing[:] *= 2
                write_fixture_dat(path,fixture)
                with self.assertRaises(OSError): plan.prepare(source,[0])
            with self.assertRaises(OSError): plan.prepare(source,[0])
            # A detached result remains valid after its source context ends.
            points = first.mesh.bounds[plan.leaf_ids].mean(axis=1)
            self.assertTrue(sample(first,points)[2].all())

    def test_mixed_transfer_reuse_fields_values_and_consumers(self):
        source, fixture = source_fixture()
        ids = np.arange(source.mesh.leaf_count)
        plan = build_fill_plan(source,ids,capacity=max(57,len(ids)))
        for fields in ([2,0,1],[1], [0,1]):
            expected = prepare(source,ids,fields)
            actual = plan.prepare(source,fields)
            np.testing.assert_array_equal(actual.values,expected.values)
        # Same geometry with fresh values: changing extrema changes minmod.
        # Fixture owner updates backing explicitly between source lifecycles.
        fixture.backing[:] = np.sin(fixture.backing*3.7)
        fresh = replace(source,identity=object())
        actual = plan.prepare(fresh,[0,1,2])
        expected = prepare(fresh,ids,[0,1,2])
        np.testing.assert_array_equal(actual.values,expected.values)
        np.testing.assert_array_equal(curl(actual).values,curl(expected).values)
        points = source.mesh.bounds.mean(axis=1)
        np.testing.assert_array_equal(sample(actual,points)[0],sample(expected,points)[0])
        unrelated,_ = source_fixture()
        with self.assertRaises(ValueError): plan.prepare(unrelated,[0])
        with self.assertRaises(MemoryError): plan.prepare(fresh,[0],budget_bytes=1)
        def fail(*args,**kwargs): raise OSError("reader failed")
        with self.assertRaises(OSError): plan.prepare(replace(fresh,read_interiors=fail),[0])
        # A failed execution has no retained mutable numeric state to poison.
        np.testing.assert_array_equal(plan.prepare(fresh,[0,1,2]).values,expected.values)


if __name__ == "__main__": unittest.main()
