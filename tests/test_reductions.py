"""Analytic native AMR reductions, geometric clipping and explicit failure policy."""

from dataclasses import replace
import numpy as np
import pytest

import simesh as sm
from simesh.fields import _Lease
from simesh.reductions import (AxisAlignedSurface, LengthUnits, extrema, histogram,
                               surface_flux, volume_integral, weighted_mean)
from fixtures import write_dat


def source():
    mesh = sm.mesh_from_forest((2, 1, 1), np.array([False] + [True] * 9),
                              lower=(0, 0, 0), upper=(2, 1, 1), block_shape=(4, 4, 4))
    values = np.empty((mesh.leaf_count, 4, *mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        x, y, z = (mesh.bounds[leaf, 0, :, None, None, None] +
                   (np.indices(mesh.block_shape) + .5) * mesh.spacing[leaf, :, None, None, None])
        values[leaf] = [x, np.ones_like(x), np.where(x < 1, 2., 4.), x + 2*y + 3*z]
    return sm.source_from_arrays(mesh, values, ("x", "one", "weight", "linear"), units="U"), values


@pytest.fixture
def raw():
    src, _ = source()
    with src:
        return sm.read_fields(src)


def test_mixed_leaves_integrate_constant_and_linear_without_preparation(raw):
    assert raw.storage_halo == raw.valid_halo == 0
    assert raw.mesh.leaf_count == 9  # The refined parent must not be counted.
    constant = volume_integral(raw, "one")
    assert constant.value == pytest.approx(2.)
    assert constant.coverage.requested_measure == 2.
    assert constant.coverage.complete
    assert constant.coverage.cell_count == 9*4**3
    assert constant.coverage.valid_measure == 2.
    assert volume_integral(raw, "linear").value == pytest.approx(7.)
    assert weighted_mean(raw, "linear").value == pytest.approx(3.5)
    assert constant.field.interpretation == "cell-average"


def test_physical_measure_conversion_is_explicit(raw):
    units = LengthUnits(3., "m")
    result = volume_integral(raw, "one", units=units)
    assert result.value == pytest.approx(54.)
    assert result.units == "(U) * (m^3)"
    assert result.coverage.units == "m^3"
    assert result.coverage.available_measure == pytest.approx(54.)
    mean = weighted_mean(raw, "x", units=units)
    assert mean.value == pytest.approx(1.)
    assert mean.weight_sum == pytest.approx(54.)
    assert mean.units == "U"
    changed_label = replace(raw, fields=tuple(replace(f, units="cm") for f in raw.fields))
    assert volume_integral(changed_label, "one").value == 2.
    assert volume_integral(changed_label, "one").coverage.units == "coordinate-length^3"


def test_partial_cells_use_piecewise_constant_values(raw):
    box = np.array([[.13, .11, .17], [1.7, .89, .92]])
    volume = np.prod(box[1] - box[0])
    assert volume_integral(raw, "one", region=box).value == pytest.approx(volume)
    # Independent 1D quadrature for x: fine cells on [0, 1], coarse on [1, 2].
    edges = np.r_[np.linspace(0., 1., 9), np.linspace(1., 2., 5)[1:]]
    lengths = np.maximum(0., np.minimum(edges[1:], box[1, 0]) - np.maximum(edges[:-1], box[0, 0]))
    expected = np.dot(lengths, .5*(edges[:-1] + edges[1:])) * np.prod((box[1] - box[0])[1:])
    result = volume_integral(raw, "x", region=box)
    assert result.value == pytest.approx(expected)
    continuous_linear = .5 * (box[0, 0] + box[1, 0]) * volume
    assert abs(result.value - continuous_linear) > 1.e-4
    selected = sm.select_region(raw.mesh, box)
    assert volume_integral(raw, "x", region=selected).value == result.value
    src, _ = source()
    with src:
        regional = sm.read_fields(src, region=box)
    assert volume_integral(regional, "x").value == result.value


def test_explicit_density_and_cell_total_weights(raw):
    mean = weighted_mean(raw, "x", weights=raw, weight_component="weight")
    assert mean.value == pytest.approx(7/6)
    assert mean.weight_sum == 6.
    assert mean.weight_mode == "density"
    assert mean.weight_units == "(U) * (coordinate-length^3)"
    # Each fine cell carries equal total weight to each coarse cell here.
    cell_mean = weighted_mean(raw, "x", weights=raw, weight_component="one", weight_mode="cell-total")
    assert cell_mean.value == pytest.approx((8*.5 + 1.5)/9)
    assert cell_mean.weight_sum == 9*4**3
    assert cell_mean.weight_units == "U"
    partial = weighted_mean(raw, "x", weights=raw, weight_component="one", weight_mode="cell-total",
                            region=[[.01, 0, 0], [.02, .02, .02]], units=LengthUnits(100., "cm"))
    assert partial.value == pytest.approx(.0625)
    assert partial.weight_sum == pytest.approx(.01*.02*.02 / .125**3)


def test_ghosts_slot_order_and_weight_order_are_independent(raw):
    h = 2
    padded = np.full((10, 8, 8, 8, 4), np.nan)
    slots = np.arange(9, dtype=np.int64)[::-1] + 1
    for leaf in range(9):
        padded[slots[leaf], h:-h, h:-h, h:-h] = raw.values[leaf]
    changed = replace(raw, _values=padded, storage_halo=h, valid_halo=0,
                      selection=sm.Selection(raw.mesh, raw.leaf_ids[::-1]), slot_of_leaf=slots)
    with pytest.raises(ValueError, match="nonpacked"):
        changed.interior()
    assert volume_integral(changed, "linear") == volume_integral(raw, "linear")
    assert weighted_mean(changed, "x", weights=raw, weight_component="weight") == weighted_mean(
        raw, "x", weights=changed, weight_component="weight")
    assert extrema(changed, "one") == extrema(raw, "one")
    a, b = histogram(changed, [0, .5, 1, 2], "x"), histogram(raw, [0, .5, 1, 2], "x")
    np.testing.assert_array_equal(a.bin_weights, b.bin_weights)
    face = AxisAlignedSurface("x", 1., [[0, 0], [1, 1]])
    assert surface_flux(changed, face, "weight").value == surface_flux(raw, face, "weight").value


def test_explicit_reverse_read_order(raw):
    src, _ = source()
    with src:
        reverse = sm.read_fields(src, leaf_ids=np.arange(9)[::-1])
    assert volume_integral(reverse, "linear").value == volume_integral(raw, "linear").value
    assert extrema(reverse, "x").minimum == extrema(raw, "x").minimum


def test_extrema_positions_are_inside_clipped_cell_and_ties_are_stable(raw):
    result = extrema(raw, "x")
    assert result.minimum.value == .0625
    assert result.minimum.leaf_id == 0
    assert result.minimum.cell_index == (0, 0, 0)
    assert result.maximum.value == 1.875
    assert result.maximum.leaf_id == 8
    clipped = extrema(raw, "one", region=[[.12, .12, .12], [.13, .13, .13]], units=LengthUnits(2., "m"))
    np.testing.assert_allclose(clipped.minimum.position, [.1225]*3)
    assert clipped.minimum == clipped.maximum
    assert clipped.position_units == "coordinate-length"
    assert clipped.coverage.valid_measure == pytest.approx(.01**3*8)


def test_histogram_volume_explicit_weights_and_outside_bins(raw):
    result = histogram(raw, [.25, 1., 1.875], "x")
    np.testing.assert_allclose(result.bin_weights, [.75, 1.])
    assert result.underflow == .25 and result.overflow == 0.
    assert result.total_weight == 2.
    assert result.coverage.valid_measure == 2.
    assert result.coverage.complete
    assert not result.edges.flags.writeable and not result.bin_weights.flags.writeable
    weighted = histogram(raw, [0, 1, 2], "x", weights=raw, weight_component="weight")
    np.testing.assert_allclose(weighted.bin_weights, [2, 4])
    totals = histogram(raw, [0, 1, 2], "x", weights=raw, weight_component="one", weight_mode="cell-total")
    np.testing.assert_allclose(totals.bin_weights, [8*4**3, 4**3])
    overflow = histogram(raw, [0, .5], "x")
    assert overflow.overflow == 1.5


def test_histogram_retains_small_weights_in_later_bins(raw):
    values = raw.values.copy()
    values[..., 2] = np.where(values[..., 0] < 1., 1.e20, 1.)
    weighted = replace(raw, _values=values)
    result = histogram(raw, [0, 1, 2], "x", weights=weighted, weight_component="weight")
    assert result.bin_weights[1] == 1.


def test_missing_domain_and_field_coverage_are_separate(raw):
    src, _ = source()
    with src:
        left = sm.read_fields(src, leaf_ids=list(range(8)))
    assert volume_integral(left, "one").value == 1.
    # An explicit full box asks for more than the default supplied leaf union.
    with pytest.raises(ValueError, match="missing field coverage"):
        volume_integral(left, "one", region=[left.mesh.lower, left.mesh.upper])
    result = volume_integral(left, "one", region=[[-1, 0, 0], [2, 1, 1]], missing="omit")
    assert result.value == 1.
    assert result.coverage.requested_measure == 3.
    assert result.coverage.domain_measure == 2.
    assert result.coverage.available_measure == 1.
    assert result.coverage.outside_measure == result.coverage.missing_measure == 1.
    assert result.coverage.fraction == pytest.approx(1/3)
    assert not result.coverage.complete
    subset = replace(left, mesh=raw.mesh, selection=sm.Selection(raw.mesh, left.leaf_ids))
    with pytest.raises(ValueError, match="missing field coverage"):
        weighted_mean(raw, "x", weights=subset, weight_component="one")
    mean = weighted_mean(raw, "x", weights=subset, weight_component="one", missing="omit")
    assert mean.value == pytest.approx(.5)
    assert mean.coverage.available_measure == 1.
    restricted = sm.Selection(raw.mesh, [8], requested_bounds=[raw.mesh.lower, raw.mesh.upper])
    with pytest.raises(ValueError, match="missing field coverage"):
        volume_integral(raw, "one", region=restricted)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_policies_and_empty_outcomes(raw, invalid):
    values = raw.values.copy()
    values[8, 0, 0, 0, 1] = invalid
    bad = replace(raw, _values=values)
    with pytest.raises(ValueError, match="nonfinite"):
        volume_integral(bad, "one")
    result = volume_integral(bad, "one", nonfinite="omit")
    assert result.value == pytest.approx(2 - .25**3)
    assert result.coverage.invalid_measure == .25**3
    assert result.coverage.valid_cell_count == result.coverage.cell_count - 1
    assert not result.coverage.complete
    mean = weighted_mean(raw, "x", weights=bad, weight_component="one", nonfinite="omit")
    assert mean.weight_sum == pytest.approx(2 - .25**3)
    box = [[1, 0, 0], [1.25, .25, .25]]
    assert volume_integral(bad, "one", region=box, nonfinite="omit").value == 0.
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_mean(bad, "one", region=box, nonfinite="omit")
    with pytest.raises(ValueError, match="finite covered cell"):
        extrema(bad, "one", region=box, nonfinite="omit")
    hist = histogram(bad, [0, 2], "one", region=box, nonfinite="omit")
    assert hist.total_weight == 0. and hist.coverage.valid_measure == 0.
    face = AxisAlignedSurface("x", 1., [[0, 0], [.25, .25]])
    assert surface_flux(bad, face, "one", nonfinite="omit").coverage.invalid_measure == .25**2


def test_zero_and_negative_weights(raw):
    for value in (-1., 0.):
        data = raw.values.copy()
        data[..., 2] = value
        weights = replace(raw, _values=data)
        with pytest.raises(ValueError, match="nonnegative" if value < 0 else "positive total weight"):
            weighted_mean(raw, "x", weights=weights, weight_component="weight", nonfinite="omit")
        if value == 0:
            hist = histogram(raw, [0, 1, 2], "x", weights=weights, weight_component="weight")
            assert hist.total_weight == 0 and hist.coverage.complete


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_surface_area_orientation_and_partial_cells(raw, axis):
    surface = AxisAlignedSurface(axis, .37, [[.13, .17], [.83, .91]])
    result = surface_flux(raw, surface, "one", units=LengthUnits(3., "m"))
    assert result.value == pytest.approx(.70*.74*9)
    assert result.coverage.valid_measure == pytest.approx(.70*.74*9)
    assert result.coverage.units == "m^2" and result.units == "(U) * (m^2)"
    assert surface_flux(raw, replace(surface, normal=-1), "one").value == pytest.approx(-.70*.74)


@pytest.mark.parametrize("side,expected", [("positive", 4.), ("negative", 2.)])
def test_surface_coarse_fine_interface_has_single_owner(raw, side, expected):
    face = AxisAlignedSurface("x", 1., [[0, 0], [1, 1]], side=side)
    result = surface_flux(raw, face, "weight")
    assert result.value == expected
    assert result.coverage.valid_measure == 1.
    assert result.coverage.complete
    assert result.coverage.cell_count == (4**2 if side == "positive" else 4*4**2)


@pytest.mark.parametrize("coordinate,positive,negative", [(.25, .3125, .1875), (0., .0625, .0625),
                                                          (2., 1.875, 1.875)])
def test_surface_internal_cell_and_domain_face_sides(raw, coordinate, positive, negative):
    face = AxisAlignedSurface("x", coordinate, [[0, 0], [1, 1]])
    assert surface_flux(raw, face, "x").value == positive
    assert surface_flux(raw, replace(face, side="negative"), "x").value == negative


def test_surface_piecewise_linear_trace_and_partial_missing_coverage(raw):
    face = AxisAlignedSurface("z", .3, [[0, 0], [2, 1]])
    # Fine z uses center .3125, coarse z uses .375. Plane integration is not
    # evaluation of the continuous linear polynomial at z=.3.
    assert surface_flux(raw, face, "linear").value == pytest.approx(4 + 3*(.3125 + .375))
    partial = replace(raw, selection=sm.Selection(raw.mesh, range(8)),
                      slot_of_leaf=np.array(list(range(8)) + [-1], dtype=np.int64))
    result = surface_flux(partial, face, "one", missing="omit")
    assert result.value == result.coverage.available_measure == 1.
    assert result.coverage.missing_measure == 1. and not result.coverage.complete
    outside = AxisAlignedSurface("z", -1., [[0, 0], [2, 1]])
    with pytest.raises(ValueError, match="outside"):
        surface_flux(raw, outside, "one")
    assert surface_flux(raw, outside, "one", missing="omit").coverage.domain_measure == 0.
    clipped = AxisAlignedSurface("z", 0., [[-1, 0], [2, 1]])
    assert surface_flux(raw, clipped, "one", missing="omit").coverage.outside_measure == 1.


def test_invalid_arguments_and_expired_borrow(raw):
    for component in (-1, 99, True, "absent", 1.5):
        with pytest.raises(ValueError):
            volume_integral(raw, component)
    for edges in ([0, 0], [2, 1], [0, np.inf], [[0, 1]], [0]):
        with pytest.raises(ValueError, match="edges"):
            histogram(raw, edges)
    for region in ([[0, 0, 0], [0, 1, 1]], [[0, 0], [1, 1]], [[0, 0, 0], [np.nan, 1, 1]]):
        with pytest.raises(ValueError, match="bounds"):
            volume_integral(raw, region=region)
    with pytest.raises(ValueError, match="outside"):
        volume_integral(raw, region=[[-1, 0, 0], [2, 1, 1]])
    with pytest.raises(ValueError, match="raise or omit"):
        volume_integral(raw, missing="ignore")
    with pytest.raises(ValueError, match="weight_mode"):
        weighted_mean(raw, weights=raw, weight_mode="auto")
    with pytest.raises(TypeError, match="LengthUnits"):
        volume_integral(raw, units="m")
    src, _ = source()
    with src:
        other = sm.read_fields(src)
    with pytest.raises(ValueError, match="same Mesh"):
        weighted_mean(raw, weights=other)
    with pytest.raises(ValueError, match="different mesh"):
        volume_integral(raw, region=other.selection)
    duplicate = replace(raw, fields=(raw.fields[0],) + raw.fields[:3])
    with pytest.raises(ValueError, match="ambiguous"):
        volume_integral(duplicate, "x")
    lease = _Lease()
    borrowed = replace(raw, _lease=lease)
    result = volume_integral(borrowed, "one")
    lease.active = False
    assert result.value == 2.
    with pytest.raises(RuntimeError, match="expired"):
        volume_integral(borrowed)


@pytest.mark.parametrize("scale", [0., -1., np.inf, np.nan, True])
def test_invalid_length_units(scale):
    with pytest.raises(ValueError):
        LengthUnits(scale, "m")


def test_invalid_surface_and_overflow(raw):
    for kwargs in ({"axis": "xy"}, {"axis": True}, {"coordinate": np.nan}, {"normal": 0},
                   {"side": "both"}, {"bounds": [[0, 0], [0, 1]]}):
        arguments = dict(axis="x", coordinate=1., bounds=[[0, 0], [1, 1]])
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            AxisAlignedSurface(**arguments)
    for scale in (1.e200, 1.e-200):
        with pytest.raises(ValueError, match="representable"):
            volume_integral(raw, units=LengthUnits(scale, "m"))
    data = raw.values.copy()
    data[..., 1] = 1.e308
    with pytest.raises(FloatingPointError):
        volume_integral(replace(raw, _values=data), "one")


def test_saved_ghost_file_to_regional_integral_workflow(tmp_path):
    src, values = source()
    path = tmp_path / "snapshot.dat"
    with src:
        write_dat(path, src.mesh, values, saved_ghosts=True, staggered=True)
    box = [[.13, .11, .17], [1.7, .89, .92]]
    with sm.open_amrvac(path) as reopened:
        interiors = sm.read_fields(reopened, fields=("b2", "b1"), region=box)
    assert interiors.valid_halo == 0
    mass = volume_integral(interiors, "b2", units=LengthUnits(2., "m"))
    assert mass.value == pytest.approx(np.prod(np.subtract(box[1], box[0])) * 8)
    assert mass.coverage.complete
    distribution = histogram(interiors, [0, 1, 2], "b1")
    assert distribution.total_weight == pytest.approx(mass.value / 8)


def test_shifted_anisotropic_domain_and_rectangular_blocks():
    mesh = sm.mesh_from_forest((2, 1, 1), np.array([False] + [True]*9),
                              lower=(10., -2., 3.), upper=(14., 4., 13.), block_shape=(2, 4, 8))
    with sm.source_from_arrays(mesh, np.ones((9, 1, 2, 4, 8)), ["one"]) as src:
        fields = sm.read_fields(src)
    assert volume_integral(fields).value == pytest.approx(240.)
    assert volume_integral(fields, region=[[10.1, -.8, 4.2], [13.7, 2.3, 11.9]]).value == pytest.approx(3.6*3.1*7.7)
    for axis, expected in enumerate((60., 40., 24.)):
        transverse = [a for a in range(3) if a != axis]
        rectangle = AxisAlignedSurface(axis, .5*(mesh.lower[axis] + mesh.upper[axis]),
                                       [mesh.lower[transverse], mesh.upper[transverse]])
        assert surface_flux(fields, rectangle).value == pytest.approx(expected)


def test_empty_target_and_ignored_values_outside_region(raw):
    empty = sm.Selection(raw.mesh, [])
    integral = volume_integral(raw, region=empty)
    assert integral.value == integral.coverage.requested_measure == 0.
    assert integral.coverage.complete
    assert histogram(raw, [0, 2], region=empty).total_weight == 0.
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_mean(raw, region=empty)
    with pytest.raises(ValueError, match="finite covered cell"):
        extrema(raw, region=empty)
    outside = volume_integral(raw, region=[[3, 3, 3], [4, 4, 4]], missing="omit")
    assert outside.value == outside.coverage.valid_measure == 0.
    assert outside.coverage.outside_measure == 1.
    values = raw.values.copy()
    values[..., 0] = np.nan
    values[0, 0, 0, 0, 0] = 2.
    bad = replace(raw, _values=values)
    assert weighted_mean(bad, region=[[0, 0, 0], [.1, .1, .1]]).value == 2.


def test_prepared_values_are_reduced_as_interiors_and_derived_metadata_survives(raw):
    src, _ = source()
    with src:
        prepared = sm.prepare(src, scheme="exact-phase")
    assert prepared.valid_halo == 2
    assert volume_integral(prepared, "linear").value == volume_integral(raw, "linear").value
    squared = sm.derive(raw, "x_squared", lambda ctx: ctx.field("x")**2, units="U^2")
    result = volume_integral(squared)
    assert result.field.interpretation == "pointwise-derived"
    assert result.representation == "piecewise-constant leaf interiors"
    # The sum of squared cell means differs from the integral of continuous x^2.
    assert result.value < 8/3
