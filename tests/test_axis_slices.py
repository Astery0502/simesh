"""Native slice extraction, interface ownership and independent output lifetime."""

from dataclasses import replace

import numpy as np
import pytest

import simesh as sm
from simesh.fields import _Lease


def source():
    mesh = sm.mesh_from_forest((2, 1, 1), np.array([False] + [True]*9),
                              lower=(0, 0, 0), upper=(2, 1, 1), block_shape=(4, 6, 8))
    cells = np.indices(mesh.block_shape)
    values = np.empty((mesh.leaf_count, 2, *mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        values[leaf, 0] = leaf*10000 + cells[0]*100 + cells[1]*10 + cells[2]
        values[leaf, 1] = -values[leaf, 0]
    return sm.source_from_arrays(mesh, values, ("code", "negative")), values


@pytest.mark.parametrize("axis", range(3))
def test_mixed_slice_matches_original_cells_and_tiles_domain(axis):
    src, original = source()
    with src:
        fields = sm.read_fields(src)
    result = sm.slice_axis(fields, axis, .37, components=("negative", "code"))
    g = result.geometry
    assert fields.valid_halo == fields.storage_halo == 0
    assert result.valid.all() and result.usable.all()
    assert not np.shares_memory(result.values, fields.values)
    assert not result.values.flags.writeable and not g.leaf_ids.flags.writeable
    area = 0.
    points = []
    bounds = g.bounds
    for row, leaf in enumerate(g.leaf_ids):
        edges = src.mesh.bounds[leaf, 0, axis] + np.arange(src.mesh.block_shape[axis]+1)*src.mesh.spacing[leaf, axis]
        candidates = np.flatnonzero((edges[:-1] <= .37) & (.37 < edges[1:]))
        assert len(candidates) == 1
        expected = np.take(original[leaf], candidates[0], axis=axis+1)
        np.testing.assert_array_equal(result.values[row], np.moveaxis(expected[::-1], 0, -1))
        assert g.cell_indices[row] == candidates[0]
        u, v = g.cell_edges(row)
        np.testing.assert_array_equal([u[0], v[0]], bounds[row, 0])
        np.testing.assert_array_equal([u[-1], v[-1]], bounds[row, 1])
        area += (u[-1]-u[0])*(v[-1]-v[0])
        point = src.mesh.bounds[leaf].mean(axis=0)
        point[axis] = .37
        points.append(point)
    np.testing.assert_array_equal(src.mesh.locate(points), g.leaf_ids)
    assert area == pytest.approx(np.prod((src.mesh.upper-src.mesh.lower)[list(g.axes)]))
    if axis != 0:
        assert set(g.levels) == {1, 2}


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", ["positive", "negative"])
def test_interfaces_and_domain_faces_agree_with_surface_reductions(axis, side):
    src, _ = source()
    with src:
        fields = sm.read_fields(src, "code")
    mesh = fields.mesh
    transverse = [a for a in range(3) if a != axis]
    bounds = np.array([mesh.lower[transverse], mesh.upper[transverse]])
    # Internal cell face, coarse/fine block face, and both domain faces.
    for coordinate in (.25, 1. if axis == 0 else .5, mesh.lower[axis], mesh.upper[axis]):
        result = sm.slice_axis(fields, axis, coordinate, side=side)
        surface = sm.AxisAlignedSurface(axis, coordinate, bounds, side=side)
        reference = sm.surface_flux(fields, surface, "code")
        integral = np.sum(result.values[..., 0]*np.prod(result.geometry.spacing, axis=1)[:, None, None])
        assert integral == pytest.approx(reference.value)


def test_coarse_fine_interface_owns_only_one_side():
    src, _ = source()
    with src:
        fields = sm.read_fields(src)
    coarse = sm.slice_axis(fields, "x", 1., side="positive")
    fine = sm.slice_axis(fields, "x", 1., side="negative")
    assert len(coarse.geometry.leaf_ids) == 1
    assert len(fine.geometry.leaf_ids) == 4
    assert np.all(coarse.geometry.cell_indices == 0)
    assert np.all(fine.geometry.cell_indices == fields.mesh.block_shape[0]-1)


def test_partial_reordered_slots_nonfinite_and_borrow_lifetime():
    src, _ = source()
    g = sm.AxisSlice(src.mesh, "z", .37)
    ids = g.leaf_ids[::2][::-1]
    with src:
        fields = sm.read_fields(src, region=sm.Selection(src.mesh, ids))
    backing = fields.values.copy()
    backing[0, 0, 0, g.cell_indices[np.flatnonzero(g.leaf_ids == ids[0])[0]], 0] = np.nan
    lease = _Lease()
    fields = replace(fields, _values=backing, _lease=lease,
                     fields=tuple(replace(f, interpretation="categorical") for f in fields.fields))
    result = sm.slice_axis(fields, "z", .37)
    assert np.all(np.isnan(result.values[~result.valid]))
    np.testing.assert_array_equal(result.valid, np.isin(g.leaf_ids, ids))
    for row, leaf in enumerate(g.leaf_ids):
        if result.valid[row]:
            np.testing.assert_array_equal(result.values[row], backing[fields.slot_of_leaf[leaf], :, :, g.cell_indices[row]])
    assert result.usable.sum() == len(ids)*np.prod(g.block_shape)-1
    saved = result.values.copy()
    lease.active = False
    np.testing.assert_array_equal(result.values, saved)
    with pytest.raises(RuntimeError, match="expired"):
        sm.slice_axis(fields, "z", .37)


@pytest.mark.parametrize("kwargs", [dict(axis=True), dict(axis="q"), dict(coordinate=np.nan),
                                     dict(coordinate=3.), dict(side="both"), dict(components=[]),
                                     dict(components=(0, 0))])
def test_invalid_requests(kwargs):
    src, _ = source()
    with src:
        fields = sm.read_fields(src)
    request = dict(axis="z", coordinate=.37)
    request.update(kwargs)
    with pytest.raises(ValueError):
        sm.slice_axis(fields, **request)


def test_budget_rejects_before_output_allocation():
    src, _ = source()
    with src:
        fields = sm.read_fields(src)
    with pytest.raises(MemoryError):
        sm.slice_axis(fields, "z", .37, memory_limit=1)
