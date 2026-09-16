"""Stored-curve profiles preserve geometry, validity, provenance and bounded work."""

from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

import simesh as sm
from simesh import applications as app
from simesh.line_profiles import sample_line_profiles
import simesh.line_profiles as profiles_module


def source_fixture(*, mixed=False):
    mesh = sm.mesh_from_forest((2, 1, 1) if mixed else (1, 1, 1),
        np.array([False]+[True]*9) if mixed else np.array([True]),
        lower=(0., 0., 0.), upper=(2., 1., 1.), block_shape=(8, 8, 8))
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count, 5, *mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        x, y, z = (mesh.bounds[leaf, 0, :, None, None, None] +
                   local*mesh.spacing[leaf, :, None, None, None])
        values[leaf] = [np.zeros_like(x), np.zeros_like(x), np.ones_like(x),
                        1e6+100*x+200*y+300*z, 2.+x+2*y+3*z]
    return sm.source_from_arrays(mesh, values, ("bx", "by", "bz", "temperature", "density"),
        units={"bx": "T", "by": "T", "bz": "T", "temperature": "K", "density": "kg/m^3"})


def given_lines(seeds, branches, *, ids=None, termination=None):
    points = sm.PointSet(seeds, ids=ids)
    counts = [len(branch) for branch in branches]
    packed = np.concatenate([np.asarray(branch, dtype=float).reshape(-1, 3) for branch in branches])
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    if termination is None:
        termination = np.full((len(points), 2), sm.Termination.MAX_STEPS, dtype=np.int64)
    return sm.LineSet(points, packed, offsets, np.asarray(termination, dtype=np.int64), object())


@pytest.mark.parametrize("mixed", [False, True])
def test_given_curves_values_lengths_units_ids_and_joined_association(mixed):
    with source_fixture(mixed=mixed) as source:
        fields = sm.prepare(source, ("temperature", "density"), scheme="exact-phase")
    a, b = [.7, .4, .4], [1.3, .6, .5]
    negative = [a, [.4, .4, .4], [.4, .3, .4]]
    positive = [a, [1., .4, .4], [1., .6, .4], [1., .6, .7]]
    lines = given_lines([a, b], [negative, positive, [], [b, [1.6, .6, .5]]],
        ids=np.array([901, -33]), termination=[[sm.Termination.MAX_LENGTH, sm.Termination.MAX_STEPS],
                                             [sm.LineSet.NOT_REQUESTED, sm.Termination.DOMAIN_EXIT]])
    result = sample_line_profiles(fields, lines, ("density", "temperature"),
                                  point_batch=2, length_units=sm.LengthUnits(10., "m"))
    xyz = lines.positions
    np.testing.assert_allclose(result.values[:, 0], 2+xyz[:, 0]+2*xyz[:, 1]+3*xyz[:, 2], atol=1e-13)
    np.testing.assert_allclose(result.values[:, 1], 1e6+100*xyz[:, 0]+200*xyz[:, 1]+300*xyz[:, 2])
    np.testing.assert_allclose(result.arclength, [0, 3, 4, 0, 3, 5, 8, 0, 3], atol=1e-14)
    assert result.valid.all() and result.finite.all() and result.usable.all()
    assert result.lines is lines and result.offsets is lines.offsets
    np.testing.assert_array_equal(result.seed_ids, [901, -33])
    assert [d.units for d in result.definitions] == ["kg/m^3", "K"]
    assert result.component_indices == (1, 0) and result.length_units.unit == "m"
    assert result.source_identity is fields.value_identity
    assert result.line_source_identity is lines.source_identity
    joined = result.line(901)
    assert joined.source_identity is result.source_identity
    assert joined.line_source_identity is result.line_source_identity
    assert joined.scheme == fields.scheme and joined.boundary == "interior"
    np.testing.assert_array_equal(joined.point_indices, [2, 1, 0, 3, 4, 5, 6])
    np.testing.assert_array_equal(joined.directions, [-1, -1, -1, 1, 1, 1, 1])
    np.testing.assert_allclose(joined.arclength, [-4, -3, 0, 0, 3, 5, 8])
    for name in ("values", "owners", "valid", "finite", "boundary_adjusted"):
        np.testing.assert_array_equal(getattr(joined, name), getattr(result, name)[joined.point_indices])
    np.testing.assert_array_equal(joined.positions, lines.positions[joined.point_indices])
    np.testing.assert_array_equal(joined.positions[2], joined.positions[3])
    branch = result.branch(901, -1)
    np.testing.assert_allclose(branch.arclength, [0, 3, 4])
    assert np.shares_memory(branch.values, result.values)
    assert not np.shares_memory(joined.values, result.values)
    empty = result.branch(-33, -1)
    assert empty.values.shape == (0, 2) and empty.termination[0] == sm.LineSet.NOT_REQUESTED
    np.testing.assert_array_equal(result.line(-33).point_indices, [7, 8])
    for obj in (result, joined, branch):
        for name in ("values", "arclength", "owners", "valid", "finite", "boundary_adjusted"):
            assert not getattr(obj, name).flags.writeable


@pytest.mark.parametrize("point_batch,workers", [(1, 1), (2, 3), (17, 2), (10000, 4)])
def test_batch_and_worker_results_are_identical(point_batch, workers):
    with source_fixture(mixed=True) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    z = np.linspace(.2, .8, 71)
    path = np.column_stack((.7+.03*np.sin(z*40), .4+.02*np.cos(z*20), z))
    lines = given_lines([path[0]], [[], path], ids=np.array([600]))
    reference = sample_line_profiles(fields, lines, [4, "temperature"], point_batch=1000)
    result = sample_line_profiles(fields, lines, [4, "temperature"],
                                  point_batch=point_batch, workers=workers)
    for name in ("arclength", "values", "owners", "valid", "finite", "boundary_adjusted"):
        np.testing.assert_array_equal(getattr(result, name), getattr(reference, name))


def test_diagnose_select_trace_then_sample_other_physics():
    with source_fixture() as source:
        magnetic = sm.prepare(source, ("bx", "by", "bz"), scheme="exact-phase")
        thermal = sm.prepare(source, ("temperature", "density"), scheme="exact-phase")
    points = sm.PointSet([[.6, .4, .5], [1.4, .6, .5]], ids=np.array([19, 402]))
    diagnostics = app.connectivity(magnetic, points, quantities=("q", "twist"), step=.04, max_steps=100)
    selected = diagnostics.threshold(q_min=1.9, abs_twist_min=0., mode="all")
    assert selected.ids.tolist() == [19, 402]
    lines = app.trace(magnetic, selected, step=.04, step_fraction=None, max_steps=4)
    assert np.all(lines.termination == sm.Termination.MAX_STEPS)
    assert np.all(np.diff(lines.offsets) == 5)
    result = sample_line_profiles(thermal, lines, ("temperature", "density"), point_batch=3, workers=2)
    assert result.line_source_identity is magnetic.value_identity
    assert result.source_identity is thermal.value_identity
    assert result.source_identity != result.line_source_identity
    assert result.valid.all() and result.finite.all()
    np.testing.assert_array_equal(result.termination, lines.termination)
    for seed_id in selected.ids:
        for direction in (-1, 1):
            np.testing.assert_allclose(result.branch(seed_id, direction).arclength, np.arange(5)*.04)
    # A caller may also deliberately sample another source on reusable geometry.
    with source_fixture() as other_source:
        other = sm.prepare(other_source, ("temperature",), scheme="exact-phase")
    alternate = sample_line_profiles(other, lines, "temperature")
    np.testing.assert_array_equal(alternate.values[:, 0], result.values[:, 0])


def test_boundary_policy_and_true_outside_points():
    with source_fixture() as source:
        fields = sm.prepare(source, ("temperature", "density"), scheme="exact-phase")
    above = np.nextafter(1., np.inf)
    below = np.nextafter(0., -np.inf)
    path = np.array([[.7, .4, .5], [.7, .4, 1.], [2., 1., 1.], [0., 0., 0.],
                     [.7, .4, above], [below, .4, 1.]])
    lines = given_lines([path[0]], [[], path])
    result = sample_line_profiles(fields, lines, point_batch=2)
    np.testing.assert_array_equal(result.valid, [1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(result.boundary_adjusted, [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(lines.positions, path)
    sampling_positions = path.copy()
    for row in (1, 2):
        for axis in range(3):
            if sampling_positions[row, axis] == fields.mesh.upper[axis]:
                sampling_positions[row, axis] = np.nextafter(fields.mesh.upper[axis], fields.mesh.lower[axis])
    expected = sm.sample(fields, sampling_positions)
    np.testing.assert_array_equal(result.values, expected[0])
    np.testing.assert_array_equal(result.owners, expected[1])
    native = sample_line_profiles(fields, lines, boundary="native")
    np.testing.assert_array_equal(native.values, sm.sample(fields, path)[0])
    assert not native.boundary_adjusted.any()
    assert not native.valid[1:3].any()


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_missing_coverage_and_component_finiteness_stay_distinct(nonfinite):
    with source_fixture(mixed=True) as source:
        fields = sm.prepare(source, ("temperature", "density"), leaf_ids=[0], scheme="exact-phase")
    # Test a finite and a nonfinite component on the same supplied coverage.
    backing = fields.values.copy()
    backing[..., 0] = nonfinite
    fields = replace(fields, _values=backing, value_identity=object())
    path = [[.2, .2, .2], [1.6, .6, .6], [2., .6, .6], [3., .6, .6]]
    lines = given_lines([path[0]], [path, []])
    result = sample_line_profiles(fields, lines)
    np.testing.assert_array_equal(result.valid, [True, False, False, False])
    np.testing.assert_array_equal(result.finite, [[False, True], [False, False], [False, False], [False, False]])
    np.testing.assert_array_equal(result.usable, result.finite)
    assert result.owners[1] >= 0 and result.owners[2] >= 0 and result.owners[3] == -1
    np.testing.assert_array_equal(result.boundary_adjusted, [False, False, True, False])
    assert len(result.arclength) == 4 and result.arclength[-1] > result.arclength[1]
    density = sample_line_profiles(fields, lines, "density")
    assert density.values.shape == (4, 1) and density.usable[0, 0]


def test_duplicate_seed_coordinates_remain_distinct_and_derived_current_is_sampleable():
    with source_fixture() as source:
        magnetic = sm.prepare(source, ("bx", "by", "bz"), scheme="exact-phase")
    seeds = sm.PointSet([[.8, .4, .5], [.8, .4, .5]], ids=np.array([-50, 700]))
    lines = app.trace(magnetic, seeds, step=.04, max_steps=3)
    current = sm.current_density(magnetic, units=sm.MagneticUnits(field_tesla=1., length_m=1e6))
    assert current.valid_halo == 1
    result = sample_line_profiles(current, lines)
    assert result.valid.all() and result.usable.all()
    np.testing.assert_allclose(result.values, 0., atol=1e-15)
    assert all(d.units == "A m^-2" for d in result.definitions)
    a, b = result.line(-50), result.line(700)
    assert not np.intersect1d(a.point_indices, b.point_indices).size
    np.testing.assert_array_equal(a.positions, b.positions)
    assert result.source_identity is current.value_identity


def test_empty_singleton_unrequested_and_tangent_branches():
    with source_fixture() as source:
        fields = sm.prepare(source, scheme="exact-phase")
    for direction in ("along", "against", "both"):
        magnetic = replace(fields, _values=np.ascontiguousarray(fields.values[..., :3]), fields=fields.fields[:3])
        lines = app.trace(magnetic, sm.PointSet([[.8, .4, .5]], ids=np.array([42])),
                          direction=direction, step=.01, max_steps=1)
        result = sample_line_profiles(fields, lines, 3)
        assert result.offsets.tolist() == lines.offsets.tolist()
        assert len(result.line(42).values) == len(lines.positions)
    lines = given_lines([[.8, .4, .5], [.9, .5, .6]], [[], [], [], [[.9, .5, .6]]],
        ids=np.array([44, 999]), termination=[[-2, -2], [-1, sm.Termination.NULL_FIELD]])
    result = sample_line_profiles(fields, lines)
    assert result.line(44).values.shape == (0, 5)
    np.testing.assert_array_equal(result.line(999).arclength, [0.])
    np.testing.assert_array_equal(result.termination, lines.termination)
    empty = sm.LineSet(sm.PointSet(np.empty((0, 3))), np.empty((0, 3)),
                       np.zeros(1, dtype=np.int64), np.empty((0, 2), dtype=np.int64), object())
    result = sample_line_profiles(fields, empty, "temperature")
    assert result.values.shape == (0, 1) and result.offsets.tolist() == [0]


def test_batch_memory_admission_and_view_budgets(monkeypatch):
    with source_fixture() as source:
        fields = sm.prepare(source, scheme="exact-phase")
    path = np.column_stack((np.full(100, .8), np.full(100, .4), np.linspace(.2, .8, 100)))
    lines = given_lines([path[0]], [[], path])
    calls = []
    native = profiles_module._sample
    def observed(fields, points, workers, executor, *args):
        calls.append(len(points))
        return native(fields, points, workers, executor, *args)
    monkeypatch.setattr(profiles_module, "_sample", observed)
    base = fields.mesh.nbytes+fields.nbytes+lines.nbytes
    retained = len(path)*(18+9*2)
    scratch = 7*(128+16*2)
    with pytest.raises(MemoryError, match="line profiles"):
        sample_line_profiles(fields, lines, [3, 4], point_batch=7, memory_limit=base+retained-1)
    assert not calls
    result = sample_line_profiles(fields, lines, [3, 4], point_batch=7, memory_limit=base+retained+scratch)
    assert max(calls) == 7 and sum(calls) == len(path)
    assert result.nbytes == lines.nbytes+retained
    for method in (lambda: result.line(0, memory_limit=1),
                   lambda: result.branch(0, 1, memory_limit=1)):
        with pytest.raises(MemoryError):
            method()


def test_lifetimes_and_valid_halo_not_allocated_padding():
    with source_fixture(mixed=True) as source:
        batches = sm.iter_prepared(source, ("temperature", "density"), scheme="exact-phase", batch_size=1)
        borrowed = next(batches)
        position = borrowed.mesh.bounds[borrowed.leaf_ids[0]].mean(axis=0)
        lines = given_lines([position], [[position], []])
        result = sample_line_profiles(borrowed, lines)
        saved = result.values.copy()
        ref = weakref.ref(borrowed)
        batches.close()
        with pytest.raises(RuntimeError, match="expired"):
            sample_line_profiles(borrowed, lines)
        del borrowed, batches
        gc.collect()
        assert ref() is None
    np.testing.assert_array_equal(result.values, saved)
    with source_fixture() as source:
        fields = sm.prepare(source, scheme="exact-phase")
    with pytest.raises(ValueError, match="valid halo"):
        sample_line_profiles(replace(fields, valid_halo=0), lines)


def test_joined_profile_does_not_retain_parent_arrays():
    with source_fixture() as source:
        fields = sm.prepare(source, ("temperature",), scheme="exact-phase")
    a, b = [.8, .4, .5], [.9, .5, .6]
    lines = given_lines([a, b], [[a], [a, b], [], [b]], ids=np.array([42, 99]))
    profiles = sample_line_profiles(fields, lines)
    references = [weakref.ref(array) for array in (lines.positions, lines.termination,
        profiles.values, profiles.arclength, profiles.owners, profiles.valid,
        profiles.finite, profiles.boundary_adjusted)]
    joined = profiles.line(42)
    assert not np.shares_memory(joined.termination, lines.termination)
    del profiles, lines
    gc.collect()
    assert all(reference() is None for reference in references)
    np.testing.assert_array_equal(joined.termination, [sm.Termination.MAX_STEPS]*2)
    assert not joined.termination.flags.writeable


@pytest.mark.parametrize("kwargs", [{"components": []}, {"components": [3, "temperature"]},
    {"components": True}, {"components": [99]}, {"components": "absent"},
    {"components": [1.5]}, {"workers": 0}, {"point_batch": 0}, {"point_batch": True},
    {"length_units": "m"}, {"boundary": "clip"}, {"memory_limit": 0}])
def test_invalid_controls(kwargs):
    with source_fixture() as source:
        fields = sm.prepare(source, scheme="exact-phase")
    lines = given_lines([[.8, .4, .5]], [[[.8, .4, .5]], []])
    with pytest.raises((ValueError, TypeError)):
        sample_line_profiles(fields, lines, **kwargs)


def test_invalid_seed_geometry_and_unrepresentable_distance():
    with source_fixture() as source:
        fields = sm.prepare(source, scheme="exact-phase")
    malformed = given_lines([[.8, .4, .5]], [[[.9, .4, .5]], []])
    with pytest.raises(ValueError, match="start at its seed"):
        sample_line_profiles(fields, malformed)
    huge = given_lines([[-1e308, 0., 0.]], [[[-1e308, 0., 0.], [1e308, 0., 0.]], []])
    with pytest.raises(ValueError, match="arclength is not representable"):
        sample_line_profiles(fields, huge)
    normal = given_lines([[0., 0., 0.]], [[[0., 0., 0.], [2., 0., 0.]], []])
    with pytest.raises(ValueError, match="arclength is not representable"):
        sample_line_profiles(fields, normal, length_units=sm.LengthUnits(1e308, "m"))
    result = sample_line_profiles(fields, normal)
    with pytest.raises(ValueError, match="existing seed ID"):
        result.line(10)
    with pytest.raises(ValueError, match="direction"):
        result.branch(0, True)
