"""Persistent native sections and quantitative reductions retain their scientific contracts."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from test_reductions import source
from test_results_io import roundtrip, rewrite


@pytest.fixture
def raw():
    snapshot, _ = source()
    with snapshot:
        return sm.read_fields(snapshot)


@pytest.mark.parametrize("side", ["positive", "negative"])
def test_native_slice_roundtrip_retains_original_mesh_and_cell_geometry(tmp_path, raw, side):
    section = sm.slice_axis(raw, "x", 1., components=("linear", "one"), side=side)
    restored = roundtrip(tmp_path, section).result
    assert restored.geometry.mesh is not raw.mesh
    for name in ("bounds", "spacing", "levels", "leaf_ids", "cell_indices"):
        np.testing.assert_array_equal(getattr(restored.geometry,name), getattr(section.geometry,name))
    for row in range(len(section.geometry.leaf_ids)):
        for before, after in zip(section.geometry.cell_edges(row), restored.geometry.cell_edges(row)):
            np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(restored.usable, section.usable)


def test_partial_slice_preserves_missing_blocks_and_nonfinite_cells(tmp_path):
    snapshot, _ = source()
    with snapshot:
        fields = sm.read_fields(snapshot, region=sm.select_roots(snapshot.mesh, ((0,0,0),(1,1,1))))
    values = fields.values.copy()
    values[0,0,0,0,0] = np.inf
    fields = replace(fields, _values=values)
    section = sm.slice_axis(fields, "z", .01)
    restored = roundtrip(tmp_path, section).result
    assert not restored.valid.all() and np.isinf(restored.values).any()
    np.testing.assert_array_equal(restored.usable, section.usable)


def test_reductions_roundtrip_preserves_units_weights_surface_and_locations(tmp_path, raw):
    lengths = sm.LengthUnits(3., "m")
    results = [sm.volume_integral(raw, "linear", units=lengths),
               sm.weighted_mean(raw, "linear", units=lengths),
               sm.extrema(raw, "linear", units=lengths),
               sm.surface_flux(raw, sm.AxisAlignedSurface("x",1.,[[0,0],[1,1]], normal=-1,side="negative"),
                               "weight",units=lengths)]
    for mode in ("density", "cell-total"):
        results += [sm.weighted_mean(raw,"x",weights=raw,weight_component="weight",weight_mode=mode),
                    sm.histogram(raw,[.25,1,1.75],"x",weights=raw,weight_component="weight",weight_mode=mode)]
    results.append(sm.histogram(raw,[.25,1,1.75],"x",units=lengths))
    for index, result in enumerate(results):
        roundtrip(tmp_path, result, f"reduction-{index}.npz")


def test_omitted_coverage_and_empty_integral_survive_roundtrip(tmp_path, raw):
    values = raw.values.copy()
    values[0,0,0,0,0] = np.nan
    fields = replace(raw, _values=values)
    integral = sm.volume_integral(fields,"x",region=[[-1,0,0],[2,1,1]],missing="omit",nonfinite="omit")
    restored = roundtrip(tmp_path,integral).result
    assert not restored.coverage.complete
    assert restored.coverage.outside_measure > 0 and restored.coverage.invalid_measure > 0
    empty = sm.volume_integral(fields,"x",region=[[3,0,0],[4,1,1]],missing="omit")
    assert roundtrip(tmp_path,empty,"empty.npz").result.value == 0


@pytest.mark.parametrize("damage", ["forest", "axis", "coverage", "representation"])
def test_corrupt_native_slice_is_rejected(tmp_path, raw, damage):
    path = sm.save_result(tmp_path/"slice.npz",sm.slice_axis(raw,"z",.25))
    def change(manifest, arrays):
        node = manifest["result"]
        if damage == "forest": arrays[node["geometry"]["mesh"]["leaf_flags"]][:] = False
        elif damage == "axis": node["geometry"]["axis"] = 3
        elif damage == "coverage": arrays[node["valid"]][0] = False
        else: node["representation"] = "linear samples"
    rewrite(path,change)
    with pytest.raises(sm.ResultFileError):
        sm.load_result(path)


@pytest.mark.parametrize("damage", ["coverage", "weights", "edges", "field"])
def test_corrupt_reduction_is_rejected(tmp_path, raw, damage):
    path = sm.save_result(tmp_path/"histogram.npz", sm.histogram(raw,[0,1,2],"x"))
    def change(manifest, arrays):
        node = manifest["result"]
        if damage == "coverage": node["coverage"]["valid_measure"] = -1.
        elif damage == "weights": arrays[node["bin_weights"]][0] = -1.
        elif damage == "edges": arrays[node["edges"]][1] = 3.
        else: node["field"]["type"] = "Coverage"
    rewrite(path,change)
    with pytest.raises(sm.ResultFileError):
        sm.load_result(path)
