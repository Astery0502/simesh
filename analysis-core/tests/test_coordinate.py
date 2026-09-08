"""Whole-domain transfer, inherited boundary correction and independent ownership."""

import gc
import weakref
import numpy as np
import pytest

import simesh as sm
from fixtures import mixed_source
from simesh.preparation import coordinate


def test_constant_across_refinement_with_two_physical_sides():
    source, _ = mixed_source(lambda x, y, z: np.array([np.ones_like(x)]*3))
    serial = sm.prepare(source, scheme="coordinate-phase")
    # The legacy three-row prolong tables produced 3456 zeros for this case.
    np.testing.assert_array_equal(serial.values, np.ones_like(serial.values))
    for workers in (2, 4):
        parallel = sm.prepare(source, scheme="coordinate-phase", workers=workers)
        np.testing.assert_array_equal(parallel.values, serial.values)
    derived = sm.curl(serial, workers=4)
    np.testing.assert_array_equal(derived.values, np.zeros_like(derived.values))
    points = np.array([[.999, .125, .125], [1.001, .125, .125]])
    assert sm.sample(serial, points)[2].all()
    np.testing.assert_array_equal(sm.sample(serial, points)[0], np.ones((2, 3)))


def test_uniform_non_cubic_boundaries_and_no_coarse_workspace():
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]), lower=(-2.7, .125, -1.1),
                              upper=(4.9, 2.5, 3.7), block_shape=(8, 6, 4))
    values = np.random.default_rng(14).normal(size=(1, 2, 8, 6, 4))
    source = sm.source_from_arrays(mesh, values, ("a", "b"))
    ready = sm.prepare(source, fields=("b", "a"), scheme="coordinate-phase", workers=4)
    expected = np.pad(np.moveaxis(values[:, [1, 0]], 1, -1),
                      ((0, 0), (2, 2), (2, 2), (2, 2), (0, 0)), mode="edge")
    np.testing.assert_array_equal(ready.values, expected)
    assert "local_coarsen_seconds" not in ready.preparation_stats["stages"]


def test_field_order_and_nonpacked_full_domain_selection():
    source, raw = mixed_source()
    serial = sm.prepare(source, fields=("b3", "b1", "b2"), scheme="coordinate-phase")
    ids = np.arange(source.mesh.leaf_count-1, -1, -1, dtype=np.int64)
    selected = sm.prepare(source, fields=("b3", "b1", "b2"), leaf_ids=ids,
                          scheme="coordinate-phase", workers=4)
    np.testing.assert_array_equal(selected.leaf_ids, ids)
    np.testing.assert_array_equal(selected.values, serial.values)
    with pytest.raises(ValueError, match="nonpacked"):
        selected.interior()
    for leaf in ids:
        np.testing.assert_array_equal(selected.window(leaf, [0, 0, 0], source.mesh.block_shape),
                                      np.moveaxis(raw[leaf, [2, 0, 1]], 0, -1))
    physical_curl = sm.curl(serial, components=(1, 2, 0))
    reversed_curl = sm.curl(selected, components=(1, 2, 0), workers=4)
    np.testing.assert_array_equal(reversed_curl.values, physical_curl.values[ids])
    sample, _, valid = sm.sample(physical_curl, [[1.5, .5, .5], [.25, .25, .25]])
    assert valid.all()
    np.testing.assert_allclose(sample, [[3., -3., 3.]]*2, rtol=0, atol=2e-13)


def test_final_backing_outlives_geometry_workspace_and_source(monkeypatch):
    source, _ = mixed_source()
    references = []
    build = coordinate.build_geometry
    allocate = coordinate.allocate_workspace

    def capture_geometry(mesh):
        geometry = build(mesh)
        references.extend([weakref.ref(geometry), weakref.ref(geometry.neighbor_children)])
        return geometry

    def capture_workspace(geometry, fields):
        workspace = allocate(geometry, fields)
        references.extend([weakref.ref(workspace), weakref.ref(workspace.coarse)])
        return workspace

    monkeypatch.setattr(coordinate, "build_geometry", capture_geometry)
    monkeypatch.setattr(coordinate, "allocate_workspace", capture_workspace)
    ready = sm.prepare(source, scheme="coordinate-phase", workers=4)
    values = ready.values
    expected = values.copy()
    assert values.flags.owndata and values.base is None
    source.close()
    del ready, source
    gc.collect()
    assert all(reference() is None for reference in references)
    np.testing.assert_array_equal(values, expected)


def test_rejected_scope_budget_and_failed_exchange(monkeypatch):
    source, raw = mixed_source()
    expected = raw.copy()
    with pytest.raises(ValueError, match="full-domain"):
        sm.prepare(source, scheme="coordinate-phase", leaf_ids=[0])
    with pytest.raises(MemoryError):
        sm.prepare(source, scheme="coordinate-phase", memory_limit=1)
    from simesh._kernels.coordinate import openmp_build_info
    if not openmp_build_info()["enabled"]:
        with pytest.raises(RuntimeError, match="OpenMP build"):
            sm.prepare(source, scheme="coordinate-phase", backend="openmp")

    def failed(geometry, workspace, values, **kwargs):
        values.fill(999.)
        raise RuntimeError("injected exchange failure")

    monkeypatch.setattr(coordinate, "execute", failed)
    with pytest.raises(RuntimeError, match="injected"):
        sm.prepare(source, scheme="coordinate-phase")
    np.testing.assert_array_equal(raw, expected)


def test_explicit_openmp_if_built():
    from simesh._kernels.coordinate import openmp_build_info
    if not openmp_build_info()["enabled"]:
        pytest.skip("ordinary build")
    source, _ = mixed_source(lambda x, y, z: np.array([np.sin(x)+y*y, np.cos(y)+z, x*z-y]))
    serial = sm.prepare(source, scheme="coordinate-phase")
    for workers in (1, 2, 4):
        result = sm.prepare(source, scheme="coordinate-phase", workers=workers, backend="openmp")
        np.testing.assert_array_equal(result.values, serial.values)


def test_unrepresentable_coordinate_stencil_is_rejected_before_exchange():
    # The block faces remain distinct, but rounding at this offset can map the
    # last fine halo center to the coarse buffer's final (non-stencil) cell.
    mesh = sm.mesh_from_forest((2,1,1), np.array([False]+[True]*9),
                              lower=(1e16,0,0), upper=(1e16+16,1,1), block_shape=(4,4,4))
    source = sm.source_from_arrays(mesh, np.ones((9,1,4,4,4)), ("value",))
    with pytest.raises(ValueError, match="not representable"):
        sm.prepare(source, scheme="coordinate-phase")
