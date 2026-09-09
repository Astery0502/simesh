"""Focused native-field composition; run with explicit development providers."""

from dataclasses import replace
import unittest

import numpy as np

from simesh.analysis import FieldDefinition, PreparedPool, prepare, sample, curl, iter_prepared
from simesh_rewrite.blockio import array_block_reader
from simesh_rewrite.forest import RefinedForest
from analysis_core.rewrite_provider import make_source
from lfe_001 import make_fixture


def source_fixture(field=None):
    f = make_fixture()
    # Reuse the fixture's validated provider forest; parent IDs are not consumed
    # by this adapter but preserve explicit metadata ownership for accounting.
    forest = RefinedForest(f.node_levels, f.node_coords,
        np.full(len(f.node_levels), -1, dtype=np.int64), f.child_node_ids,
        f.node_leaf_ids, f.leaf_node_ids, f.root_node_ids, f.max_level)
    if field is not None:
        from simesh_rewrite.refined_geometry import refined_leaf_geometry
        bounds, spacing = refined_leaf_geometry(f.domain_lower, f.domain_upper,
            f.root_shape, f.domain_counts, f.block_counts, f.node_levels, f.node_coords,
            f.leaf_node_ids, np.arange(len(f.leaf_node_ids), dtype=np.int64))
        local = np.indices(tuple(f.block_counts))+.5
        for leaf in range(len(bounds)):
            xyz = bounds[leaf,0,:,None,None,None]+local*spacing[leaf,:,None,None,None]
            f.backing[leaf] = field(*xyz)
    source = make_source(f.root_shape, f.coord_to_rank, forest, f.domain_lower,
        f.domain_upper, f.block_counts, array_block_reader(f.backing),
        tuple(FieldDefinition(name, "code") for name in ("b1", "b2", "b3")))
    return source, f


def scalar_sample(product, points):
    output = []
    for point in points:
        inside = np.all((point >= product.mesh.bounds[:, 0]) &
                        (point < product.mesh.bounds[:, 1]), axis=1)
        leaf = np.flatnonzero(inside)[0]
        q = (point-product.mesh.bounds[leaf, 0])/product.mesh.spacing[leaf]-.5
        base = np.floor(q).astype(int)
        t = q-base
        base += product.halo
        value = np.zeros(len(product.fields))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    weight = ((t[0] if x else 1-t[0]) * (t[1] if y else 1-t[1]) *
                              (t[2] if z else 1-t[2]))
                    value += weight*product.values[product.slot_of_leaf[leaf],
                                                  base[0]+x,base[1]+y,base[2]+z]
        output.append(value)
    return np.array(output)


class PreparedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.fixture = source_fixture()
        cls.ids = np.arange(cls.source.mesh.leaf_count-1, -1, -1, dtype=np.int64)
        cls.product = prepare(cls.source, cls.ids, [2, 0, 1])

    def test_direct_preparation_matches_checked_provider(self):
        from simesh_rewrite.completed_primary import (
            make_completed_primary_consumer, execute_selected_refined_halos_with_consumer,
        )
        source, f = source_fixture(lambda x,y,z: np.array([
            np.sin(3*x)+y*y, np.cos(2*y)+z*z, np.sin(x*z)-y]))
        ids = np.random.default_rng(83).permutation(source.mesh.leaf_count).astype(np.int64)
        selected_fields = np.array([2,0,1],dtype=np.int64)
        actual = prepare(source,ids,selected_fields)
        expected = np.empty_like(actual.values)
        order = np.argsort(ids)
        def consume(state, offset, primary_ids, payload, *windows):
            expected[order[offset:offset+len(primary_ids)]] = np.moveaxis(payload[:len(primary_ids)],1,-1)
        consumer = make_completed_primary_consumer(None,consume,output_arrays=(expected,))
        execute_selected_refined_halos_with_consumer(
            array_block_reader(f.backing),consumer,np.ascontiguousarray(ids[order]),selected_fields,
            f.root_shape,f.coord_to_rank,f.root_node_ids,f.node_levels,f.node_coords,
            f.child_node_ids,f.node_leaf_ids,f.leaf_node_ids,np.full(3,2,dtype=np.int64),
            np.full(3,2,dtype=np.int64),np.zeros((3,6),dtype=np.uint8),
            np.full(3,-1,dtype=np.int64),min(128,source.mesh.leaf_count))
        # Oscillatory values exercise changing minmod decisions on mixed-level
        # faces, edges, corners and physical boundaries, not just affine fields.
        np.testing.assert_array_equal(actual.values,expected)

    def test_selected_mapping_halos_and_direct_sampling(self):
        p, f = self.product, self.fixture
        np.testing.assert_array_equal(p.interior(), np.moveaxis(f.backing[self.ids][:, [2,0,1]], 1, -1))
        rng = np.random.default_rng(12)
        points = p.mesh.lower + rng.random((1500, 3))*(p.mesh.upper-p.mesh.lower)
        # Include owner faces and representable neighbors on mixed-level edges.
        faces = p.mesh.bounds[:, 0] + .25*p.mesh.spacing
        faces[:, 0] = p.mesh.bounds[:, 0, 0]
        points = np.concatenate((points, faces, np.nextafter(faces, p.mesh.upper)))
        actual, owners, valid = sample(p, points)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(actual, scalar_sample(p, points), atol=2e-13, rtol=2e-14)
        # Affine values are exact away from continuous physical boundary influence.
        safe = np.all((points > p.mesh.lower+1.) & (points < p.mesh.upper-1.), axis=1)
        x,y,z = points[safe].T
        expected = np.column_stack((5*x+6*y, y+2*z, 3*z+4*x))
        np.testing.assert_allclose(actual[safe], expected, atol=2e-13, rtol=2e-14)
        self.assertTrue(np.all(p.mesh.locate([p.mesh.upper, [np.nan,0,0]]) == -1))
        missing = prepare(self.source, [0], [0])
        values, _, okay = sample(missing, [p.mesh.bounds[-1].mean(axis=0)])
        self.assertFalse(okay[0])
        self.assertTrue(np.isnan(values).all())

    def test_coverage_lease_preserves_actual_request_recency(self):
        pool=PreparedPool(self.source,[0],3)
        try:
            with pool.borrow([1,2,3]):
                pass
            with pool.borrow([1]):
                pass
            with pool.borrow(pool.resident_leaf_ids,touch=False):
                pass
            with pool.borrow([4]):
                pass
            before=pool.prepared_count
            with pool.borrow([1]) as hot:
                expected=prepare(self.source,[1],[0])
                np.testing.assert_array_equal(hot.window(1,[0,0,0],self.source.mesh.block_shape),
                                              expected.interior()[0])
            self.assertEqual(pool.prepared_count,before)
        finally:
            pool.close()


    def test_borrow_eviction_failure_and_lifetime(self):
        selected_fields = np.array([2,0,1],dtype=np.int64)
        pool = PreparedPool(self.source, selected_fields, 2)
        selected_fields[:] = [0,1,2]  # A retained selector must be owned by the pool.
        with pool.borrow([5, 1]) as borrowed:
            saved = borrowed.values.copy()
            with self.assertRaises(RuntimeError):
                pool.clear()
            with self.assertRaises(RuntimeError):
                with pool.borrow([7]):
                    pass
            with self.assertRaises(RuntimeError):
                with pool.borrow([1]):
                    pass
            np.testing.assert_array_equal(saved, borrowed.values)
        with self.assertRaises(RuntimeError):
            _ = borrowed.values
        with pool.borrow([7, 9]) as current:
            points = self.source.mesh.bounds[[7,9]].mean(axis=1)
            np.testing.assert_array_equal(sample(current, points)[0], sample(self.product, points)[0])
        pool.close()
        # A detached product remains usable independently of any cache lifetime.
        self.assertTrue(sample(self.product, points)[2].all())
        with self.assertRaises(MemoryError):
            prepare(self.source, [0], [0], budget_bytes=1)

        def fail(ids, fields, halo, out):
            out.fill(123)
            raise OSError("injected provider failure")

        broken = PreparedPool(replace(self.source, fill=fail), [0], 2)
        with self.assertRaises(OSError):
            with broken.borrow([0]):
                pass
        self.assertTrue(np.all(broken._directory == -1))

    def test_bounded_native_file_to_sink(self):
        import os
        import tempfile
        from pathlib import Path
        from lfe_001 import write_fixture_dat
        from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
        from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.dat"
            write_fixture_dat(path, self.fixture)
            fd = os.open(path, os.O_RDONLY)
            try:
                index = read_amrvac_v5_index(fd)
                binding = bind_amrvac_v5_forest(index)
                reader = make_amrvac_v5_block_reader(fd, index, binding)
                source = make_source(binding.root_shape, binding.coord_to_rank, binding.forest,
                    index.domain_lower, index.domain_upper, index.block_cell_counts,
                    reader, self.source.fields)
                selected = np.array([32, 5, 18, 67, 94])
                sink = np.lib.format.open_memmap(Path(directory)/"result.npy", mode="w+",
                    dtype=float, shape=(len(selected), 4,4,4,2))
                position = {leaf:i for i,leaf in enumerate(selected)}
                for batch in iter_prepared(source, selected, [2,0], capacity=2):
                    for leaf in batch.leaf_ids:
                        sink[position[leaf]] = batch.window(leaf, [0,0,0], [4,4,4])
                np.testing.assert_array_equal(sink, np.moveaxis(self.fixture.backing[selected][:,[2,0]],1,-1))
                detached = prepare(source, selected, [2,0])
            finally:
                os.close(fd)
            np.testing.assert_array_equal(detached.interior(), sink)

    def test_parent_and_box_coverage(self):
        mesh = self.source.mesh
        ids = np.concatenate([mesh.descendants(int(root)) for root in mesh.roots.ravel()])
        np.testing.assert_array_equal(np.sort(ids), np.arange(mesh.leaf_count))
        selected, lower, upper = mesh.select_box(mesh.lower, mesh.upper)
        np.testing.assert_array_equal(selected, np.arange(mesh.leaf_count))
        self.assertTrue(np.all(lower == 0))
        self.assertTrue(np.all(upper == mesh.block_shape))

    def test_extended_derivative_validity_and_interface_samples(self):
        derived = curl(self.product, components=(1,2,0))
        self.assertEqual(derived.halo, 1)
        self.assertFalse(np.shares_memory(derived.values, self.product.values))
        mesh = self.source.mesh
        # Sample near leaf faces: derivative interpolation consumes the outer
        # primary layer. Keep physical constant-extension effects separate.
        points = mesh.bounds[:, 0] + .1*mesh.spacing
        safe = np.all((points > mesh.lower+1.5) & (points < mesh.upper-1.5), axis=1)
        values, _, valid = sample(derived, points[safe])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(values, np.broadcast_to([3.,-3.,3.], values.shape),
                                   atol=2e-13, rtol=2e-14)
        twice = curl(derived)
        self.assertEqual(twice.halo, 0)
        with self.assertRaises(ValueError):
            sample(twice, points[:1])


if __name__ == "__main__":
    unittest.main()
