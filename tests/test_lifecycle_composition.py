"""Numerical workflows across recursive AMR layouts and explicit lifetimes."""

import numpy as np
import pytest

import simesh as sm
from simesh import applications as app
from fixtures import nested_source


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("scheme", ["exact-phase", "coordinate-phase"])
def test_recursive_layout_preserves_composed_field_calculus(depth, scheme):
    with nested_source(depth) as source:
        mesh = source.mesh
        centers = mesh.bounds.mean(axis=1)
        np.testing.assert_array_equal(mesh.locate(centers), np.arange(mesh.leaf_count))
        ready = sm.prepare(source, scheme=scheme)
    x, y, z = centers.T
    expected = np.column_stack((y+2*z, 3*z+4*x, 5*x+6*y))
    sampled = app.sample(ready, sm.PointSet(centers))
    assert sampled.valid.all()
    np.testing.assert_allclose(sampled.values, expected, rtol=0, atol=2e-14)
    curl = sm.curl(ready)
    assert curl.valid_halo == ready.valid_halo-1
    values, owners, valid = sm.sample(curl, centers)
    assert valid.all()
    np.testing.assert_array_equal(owners, np.arange(mesh.leaf_count))
    np.testing.assert_allclose(values, np.broadcast_to([3., -3., 3.], values.shape),
                               rtol=0, atol=2e-13)
    points, areas = sm.native_bottom_seeds(mesh)
    np.testing.assert_allclose(areas.sum(), 1., rtol=0, atol=2e-15)
    assert np.all(points.positions[:, 2] == mesh.lower[2])


@pytest.mark.parametrize("interpolation", ["zero", "native"])
def test_uniform_slabs_match_volume_with_shifted_bounds_and_sparse_coverage(interpolation):
    mesh = sm.mesh_from_forest((2, 2, 3), np.ones(12, dtype=bool),
        lower=(-.3, .1, -.7), upper=(1.1, 1.9, 2.3), block_shape=(6, 4, 8))
    backing = np.arange(12*2*6*4*8, dtype=float).reshape(12, 2, 6, 4, 8)
    # Sparse, reversed storage must retain original owners in missing slabs too.
    with sm.source_from_arrays(mesh, backing, ("a", "b"), copy=False) as source:
        fields = sm.read_fields(source, leaf_ids=[10, 8, 7, 3, 1])
    bounds = np.array([mesh.lower, mesh.upper])+np.array([0., 0., .5])
    resolution = (12, 8, 24) if interpolation == "native" else (9, 7, 19)
    volume = app.uniform_grid(fields, resolution, bounds=bounds,
                              components=("b", "a"), interpolation=interpolation)
    for iz, slab in sm.iter_uniform(fields, resolution, bounds=bounds,
                                   components=("b", "a"), interpolation=interpolation, workers=2):
        np.testing.assert_array_equal(slab.values, volume.values[:, :, iz])
        np.testing.assert_array_equal(slab.valid, volume.valid[:, :, iz])
        # Use the volume lattice arithmetic at half-open block faces.
        x, y, z = volume.axes
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z[iz])))
        np.testing.assert_array_equal(slab.owners, mesh.locate(points).reshape(slab.valid.shape))


def test_zero_slabs_keep_kernel_ownership_at_rounded_block_face():
    mesh = sm.mesh_from_forest((1, 1, 2), np.ones(2, dtype=bool),
        lower=(0., 0., -.7), upper=(1., 1., 1.3), block_shape=(2, 2, 2))
    backing = np.empty((2, 1, 2, 2, 2))
    backing[0], backing[1] = 10., 20.
    with sm.source_from_arrays(mesh, backing, ("value",), copy=False) as source:
        fields = sm.read_fields(source)
    resolution = (1, 1, 3)
    volume = app.uniform_grid(fields, resolution, interpolation="zero")
    for iz, slab in sm.iter_uniform(fields, resolution, interpolation="zero"):
        np.testing.assert_array_equal(slab.values, volume.values[:, :, iz])
        assert slab.valid.all()
