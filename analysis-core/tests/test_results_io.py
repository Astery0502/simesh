"""Durable associations, scientific state, and strict file-boundary validation."""

from dataclasses import fields, is_dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from simesh import results_io as rio
from test_connectivity import magnetic_source


@pytest.fixture(scope="module")
def magnetic():
    with magnetic_source(lambda x, y, z: np.array([np.zeros_like(x), np.zeros_like(x), np.ones_like(x)]),
                         cells=8) as source:
        return sm.prepare(source, scheme="exact-phase")


@pytest.fixture
def points():
    plane = sm.Plane([-.6, -.5, .4], [1.2, .1, 0], [.1, 1., 0], (2, 3))
    return sm.PointSet.from_plane(plane, ids=np.array([-7, 51, 2**63-1, 19, 200, -2**63], np.int64))


def assert_equivalent(original, restored):
    if isinstance(original, np.ndarray):
        assert original.dtype == restored.dtype and original.shape == restored.shape
        np.testing.assert_array_equal(original, restored)
        assert not restored.flags.writeable
    elif is_dataclass(original):
        assert type(restored) is type(original)
        for member in fields(original):
            if member.name == "source_identity":
                assert restored.source_identity is None
            else:
                assert_equivalent(getattr(original, member.name), getattr(restored, member.name))
    else:
        assert original == restored


def roundtrip(tmp_path, original, name="result.npz", **kwargs):
    path = tmp_path / name
    assert rio.save_result(path, original, **kwargs) == path
    loaded = rio.load_result(path)
    assert loaded.schema_version == rio.SCHEMA_VERSION == 1
    assert_equivalent(original, loaded.result)
    return loaded


def test_points_full_sparse_and_empty_keep_layout_and_parent_plane(tmp_path, points):
    full = roundtrip(tmp_path, points).result
    assert full.reshape(np.arange(6)).shape == (2, 3)
    selected = points.select(np.array([[True, False, False], [False, True, False]]))
    sparse = roundtrip(tmp_path, selected, "sparse.npz").result
    assert sparse.shape == (2,) and sparse.plane.shape == (2, 3)
    np.testing.assert_array_equal(sparse.ids, [-7, 200])
    roundtrip(tmp_path, points.select(np.zeros(6, bool)), "empty.npz")
    normals = np.broadcast_to([.3, .4, 1.], (6, 3))
    roundtrip(tmp_path, sm.PointSet(points.positions, points.ids, normals=normals), "normals.npz")
    roundtrip(tmp_path, sm.PointSet(np.empty((0, 3))), "no-plane.npz")


@pytest.mark.parametrize("per_ray", [False, True])
def test_rays_preserve_exact_normalized_vectors_and_intervals(tmp_path, points, per_ray):
    directions = np.array([[.13*i, .21, 1] for i in range(6)]) if per_ray else [.13, .21, 1.]
    rays = sm.RaySet(points, directions, near=np.arange(6)/10, far=[np.inf, 1, 2, 3, 4, 5])
    restored = roundtrip(tmp_path, rays).result
    chosen = np.array([True, False, True, False, False, False])
    assert_equivalent(rays.select(chosen), restored.select(chosen))


def test_sample_and_uniform_keep_definitions_geometry_and_nonfinite_validity(tmp_path, magnetic, points):
    definitions = tuple(sm.FieldDefinition(name, "T", "prepared-node") for name in ("Bx", "By", "Bz"))
    sample = app.sample(magnetic, points)
    values = sample.values.copy()
    values[0, 0], values[1, 2] = np.nan, np.inf
    sample = replace(sample, values=values, definitions=definitions)
    loaded = roundtrip(tmp_path, sample).result
    np.testing.assert_array_equal(loaded.image, sample.image)
    np.testing.assert_array_equal(loaded.usable, sample.usable)
    assert loaded.valid.all() and not loaded.usable[:2].any()
    volume = app.uniform_grid(magnetic, (3, 4, 2), bounds=([-.6, -.5, .2], [.6, .5, .8]))
    values = volume.values.copy()
    values[0, 0, 0] = np.nan
    volume = replace(volume, values=values, definitions=definitions)
    restored = roundtrip(tmp_path, volume, "volume.npz").result
    for axis, original in zip(restored.axes, volume.axes):
        np.testing.assert_array_equal(axis, original)
    np.testing.assert_array_equal(restored.spacing, volume.spacing)
    np.testing.assert_array_equal(restored.usable, volume.usable)
    outside = app.sample(magnetic, sm.PointSet([[5., 5., 5.]]))
    assert not roundtrip(tmp_path, outside, "outside.npz").result.valid.any()


@pytest.mark.parametrize("quantities", [("q",), ("twist",), ("q", "twist")])
def test_q_only_twist_only_and_combined_maps(tmp_path, magnetic, points, quantities):
    result = app.connectivity(magnetic, points, quantities=quantities, local_radius=.15)
    loaded = roundtrip(tmp_path, result).result
    roundtrip(tmp_path, result.data, "raw-qsl.npz")
    assert loaded.quantities == quantities
    np.testing.assert_array_equal(loaded.image("q_valid"), result.image("q_valid"))
    np.testing.assert_array_equal(loaded.image("twist_valid"), result.image("twist_valid"))
    if "q" in quantities:
        np.testing.assert_allclose(loaded.data.q, 2., atol=1e-10)
        np.testing.assert_array_equal(loaded.data.q_local, loaded.data.q)
        assert_equivalent(result.threshold(q_min=1.9), loaded.threshold(q_min=1.9))
    else:
        assert loaded.data.q is None and loaded.data.log10_q_perp is None
        with pytest.raises(ValueError, match="not computed"):
            loaded.threshold(q_min=2)
    if "twist" in quantities:
        assert_equivalent(result.threshold(abs_twist_min=0), loaded.threshold(abs_twist_min=0))
    else:
        assert loaded.data.twist is None
    limited = app.connectivity(magnetic, points, quantities=quantities, max_steps=1)
    restored = roundtrip(tmp_path, limited, "limited.npz").result
    assert not restored.data.complete.any() and not restored.data.valid.any()


@pytest.mark.parametrize("direction", ["both", "along", "against", "inward"])
def test_compact_line_branches_are_associated_by_id(tmp_path, magnetic, direction):
    seeds = sm.PointSet.boundary(magnetic.mesh, "zmax", (2, 2), ids=np.array([99, -4, 81, 5]))
    original = app.trace(magnetic, seeds, direction=direction, step=.03, max_steps=50, seed_batch=2)
    restored = roundtrip(tmp_path, original).result
    assert restored.offsets.shape == (2*len(seeds)+1,)
    for seed_id in seeds.ids:
        np.testing.assert_array_equal(restored.line(seed_id), original.line(seed_id))
        for side in (-1, 1):
            np.testing.assert_array_equal(restored.branch(seed_id, side), original.branch(seed_id, side))
    assert len(restored.positions) < 2*len(seeds)*51


def test_empty_outside_and_tangent_traces(tmp_path, magnetic):
    for name, seeds, direction in (
            ("empty", sm.PointSet(np.empty((0, 3))), "both"),
            ("outside", sm.PointSet([[3., 3., 3.]]), "both"),
            ("tangent", sm.PointSet.boundary(magnetic.mesh, "xmin", (1, 2)), "inward")):
        result = app.trace(magnetic, seeds, direction=direction, step=.02)
        assert len(roundtrip(tmp_path, result, name+".npz").result.positions) == 0


@pytest.mark.parametrize("thermal", [False, True])
def test_ray_results_keep_status_units_and_physical_model(tmp_path, magnetic, thermal):
    rays = sm.RaySet(sm.PointSet([[.1, .1, -1], [3., 3., -1.]], ids=np.array([101, 99]), shape=(1, 2)),
                     [.01, .02, 1])
    if thermal:
        density = sm.derive(magnetic, "density", lambda ctx: ctx.field("b3"), units="code")
        ready = sm.thermal_fields(density, 1e6, density_unit_g_cm3=1e-15, temperature_label="isothermal test")
        result = app.thermal_los(ready, rays, length_unit_cm=1e8)
        assert result.metadata["model"] and result.metadata["length_unit_cm"] == 1e8
    else:
        result = app.los(magnetic, rays, component=2)
    loaded = roundtrip(tmp_path, result).result
    assert loaded.complete
    np.testing.assert_array_equal(loaded.image, result.image)
    np.testing.assert_array_equal(loaded.valid, result.valid)
    assert_equivalent(loaded.select(loaded.valid), result.select(result.valid))
    failed = app.los(magnetic, rays, component=2, max_samples=1)
    restored = roundtrip(tmp_path, failed, "failed.npz").result
    assert not restored.complete
    np.testing.assert_array_equal(restored.status, failed.status)


def test_cross_process_diagnose_reload_select_trace_workflow(tmp_path, magnetic, points):
    controls = {"quantities": ["q", "twist"], "step_fraction": .25, "max_steps": 1000}
    diagnostic = app.connectivity(magnetic, points, **controls)
    path = tmp_path / "diagnostic.npz"
    source = {"description": "analytic constant Bz", "field_units": "code", "coordinate_units": "code"}
    metadata = {"calculation": controls, "physical_model": {"magnetic_components": ["b1", "b2", "b3"]}}
    rio.save_result(path, diagnostic, source=source, metadata=metadata)
    selected_path = tmp_path / "selected.npz"
    script = """
import sys
import numpy as np
from simesh.results_io import load_result, save_result
saved = load_result(sys.argv[1])
assert saved.source_verification == 'unverified'
assert saved.result.source_identity is None
assert saved.result.image('q').shape == (2, 3)
np.testing.assert_allclose(saved.result.image('q'), 2., atol=1e-10)
selected = saved.result.threshold(q_min=1.9)
save_result(sys.argv[2], selected, metadata={'selection': {'q_min': 1.9}}, source=saved.source)
"""
    subprocess.run([sys.executable, "-c", script, str(path), str(selected_path)], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    loaded = rio.load_result(path)
    assert loaded.metadata == metadata and loaded.source == source
    selected = rio.load_result(selected_path).result
    lines = app.trace(magnetic, selected, step=.03, max_steps=100)
    saved_lines = roundtrip(tmp_path, lines, "lines.npz", source=source,
                            metadata={"direction": "both", "step": .03, "max_steps": 100}).result
    np.testing.assert_array_equal(saved_lines.seeds.ids, diagnostic.points.ids)
    for seed_id in selected.ids:
        assert len(saved_lines.branch(seed_id, -1)) > 1 and len(saved_lines.branch(seed_id, 1)) > 1


def test_explicit_metadata_never_serializes_live_source_identity(tmp_path, magnetic, points):
    result = replace(app.sample(magnetic, points), source_identity=object())
    loaded = roundtrip(tmp_path, result, metadata={"units": {"length_cm": 1e8}, "components": ("x", "y")})
    assert loaded.source is None and loaded.source_verification == "unspecified"
    assert loaded.metadata["components"] == ["x", "y"]
    with np.load(tmp_path / "result.npz", allow_pickle=False) as archive:
        text = archive["__manifest__"].tobytes().decode()
        assert "source_identity" not in text and "object at" not in text
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def rewrite(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    manifest = json.loads(arrays["__manifest__"].tobytes())
    change(manifest, arrays)
    arrays["__manifest__"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


@pytest.mark.parametrize("damage", ["version", "bool-version", "type", "shape", "duplicate-ids", "id-dtype",
                                     "position-dtype", "missing-array", "extra-array", "extra-member",
                                     "object-array", "nonfinite", "unnormalized", "verified", "wrong-child"])
def test_invalid_or_unsupported_geometry_files_are_rejected(tmp_path, points, damage):
    path = rio.save_result(tmp_path / "bad.npz", points)
    def change(manifest, arrays):
        node = manifest["result"]
        if damage == "version": manifest["schema_version"] = 2
        elif damage == "bool-version": manifest["schema_version"] = True
        elif damage == "type": node["type"] = "os.system"
        elif damage == "shape": node["shape"] = [3, 3]
        elif damage == "duplicate-ids": arrays[node["ids"]][1] = arrays[node["ids"]][0]
        elif damage == "id-dtype": arrays[node["ids"]] = arrays[node["ids"]].astype(np.uint64)
        elif damage == "position-dtype": arrays[node["positions"]] = arrays[node["positions"]].astype(np.float32)
        elif damage == "missing-array": del arrays[node["positions"]]
        elif damage == "extra-array": arrays["extra"] = np.zeros(1)
        elif damage == "extra-member": node["extra"] = True
        elif damage == "object-array": arrays[node["positions"]] = np.empty((6, 3), object)
        elif damage == "nonfinite": arrays[node["positions"]][0, 0] = np.nan
        elif damage == "unnormalized": arrays[node["normals"]] *= 2
        elif damage == "verified": manifest["provenance"]["verification"] = "verified"
        elif damage == "wrong-child": node["plane"]["type"] = "PointSet"
    rewrite(path, change)
    with pytest.raises(rio.ResultFileError):
        rio.load_result(path)


@pytest.mark.parametrize("damage", ["offset-count", "offset-start", "offset-end", "offset-order", "offset-negative",
                                     "offset-overflow", "seed-mismatch", "bad-termination", "running"])
def test_corrupt_packed_offsets_and_associations_are_rejected(tmp_path, magnetic, points, damage):
    path = rio.save_result(tmp_path / "bad.npz", app.trace(magnetic, points, step=.1, max_steps=10))
    def change(manifest, arrays):
        node = manifest["result"]
        offsets = arrays[node["offsets"]]
        if damage == "offset-count": arrays[node["offsets"]] = offsets[:-1]
        elif damage == "offset-start": offsets[0] = 1
        elif damage == "offset-end": offsets[-1] += 1
        elif damage == "offset-order": offsets[1] = offsets[2]+1
        elif damage == "offset-negative": offsets[1] = -1
        elif damage == "offset-overflow": offsets[1] = np.iinfo(np.int64).max
        elif damage == "seed-mismatch": arrays[node["positions"]][0] += .1
        elif damage == "bad-termination": arrays[node["termination"]][0, 0] = 99
        elif damage == "running": arrays[node["termination"]][0, 0] = int(sm.Termination.RUNNING)
    rewrite(path, change)
    with pytest.raises(rio.ResultFileError):
        rio.load_result(path)


@pytest.mark.parametrize("damage", ["q-shape", "partial-q", "bad-valid", "bad-complete", "seed-mismatch", "bad-status", "nan-q", "zero-q"])
def test_corrupt_diagnostic_states_are_rejected(tmp_path, magnetic, points, damage):
    path = rio.save_result(tmp_path / "bad.npz", app.connectivity(magnetic, points))
    def change(manifest, arrays):
        node = manifest["result"]["data"]
        if damage == "q-shape": arrays[node["q"]] = np.zeros((2, 3))
        elif damage == "partial-q": del arrays[node["q"]]; node["q"] = None
        elif damage == "bad-valid": arrays[node["valid"]][0] = False
        elif damage == "bad-complete": arrays[node["complete"]][0] = False
        elif damage == "seed-mismatch": arrays[node["seeds"]][0, 0] += .1
        elif damage == "bad-status": arrays[node["termination"]][0, 0] = 99
        elif damage == "nan-q": arrays[node["q"]][0] = np.nan
        elif damage == "zero-q": arrays[node["q"]][0] = 0.
    rewrite(path, change)
    with pytest.raises(rio.ResultFileError):
        rio.load_result(path)


@pytest.mark.parametrize("damage", ["running", "nan-success", "nonzero-empty"])
def test_contradictory_los_states_are_rejected(tmp_path, magnetic, damage):
    rays = sm.RaySet(sm.PointSet([[0.,0.,-1.],[3.,3.,-1.]]),[0,0,1])
    result = app.los(magnetic,rays,component=2)
    path = rio.save_result(tmp_path/"ray.npz",result)
    def change(manifest,arrays):
        node = manifest["result"]
        if damage == "running": arrays[node["status"]][0] = int(sm.LOSStatus.RUNNING)
        elif damage == "nan-success": arrays[node["values"]][0] = np.nan
        else: arrays[node["values"]][1] = 1.
    rewrite(path,change)
    with pytest.raises(rio.ResultFileError):
        rio.load_result(path)
    invalid_values = result.values.copy()
    invalid_values[0] = np.nan
    with pytest.raises(rio.ResultFileError):
        rio.save_result(tmp_path/"invalid-output.npz",replace(result,values=invalid_values))


@pytest.mark.parametrize("payload", [b"not a file", b"PK\x03\x04truncated"])
def test_unreadable_files_and_filesystem_errors(tmp_path, payload):
    path = tmp_path / "bad.npz"
    path.write_bytes(payload)
    with pytest.raises(rio.ResultFileError): rio.load_result(path)
    with pytest.raises(FileNotFoundError): rio.load_result(tmp_path / "missing.npz")


@pytest.mark.parametrize("payload", [b'{"format":1,"format":2}', b'{"format":NaN}', b'\xff', b'[]'])
def test_malformed_json_is_rejected(tmp_path, payload):
    path = tmp_path / "bad.npz"
    np.savez(path, __manifest__=np.frombuffer(payload, dtype=np.uint8))
    with pytest.raises(rio.ResultFileError): rio.load_result(path)


def test_duplicate_zip_members_and_manifest_size_limit(tmp_path, points):
    path = rio.save_result(tmp_path / "bad.npz", points)
    with zipfile.ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning):
            archive.writestr("__manifest__.npy", b"duplicate")
    with pytest.raises(rio.ResultFileError): rio.load_result(path)
    np.savez(path, __manifest__=np.zeros(1024*1024+1, dtype=np.uint8))
    with pytest.raises(rio.ResultFileError): rio.load_result(path)


@pytest.mark.parametrize("metadata", [{"bad": object()}, {"bad": np.ones(1)}, {1: "integer key"}, {"bad": float("inf")}])
def test_non_json_metadata_is_rejected_before_publication(tmp_path, points, metadata):
    path = tmp_path / "result.npz"
    with pytest.raises(rio.ResultFileError): rio.save_result(path, points, metadata=metadata)
    assert not path.exists() and not list(tmp_path.glob(".simesh-result-*"))


def test_refuse_overwrite_and_preserve_original_on_serialization_failure(tmp_path, points, monkeypatch):
    path = rio.save_result(tmp_path / "existing.result", points)
    original = path.read_bytes()
    with pytest.raises(FileExistsError): rio.save_result(path, points)
    assert path.read_bytes() == original
    def fail(stream, **arrays):
        stream.write(b"incomplete")
        raise OSError("injected serialization failure")
    monkeypatch.setattr(rio.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="injected"):
        rio.save_result(path, points, overwrite=True)
    assert path.read_bytes() == original
    with pytest.raises(OSError): rio.save_result(tmp_path / "new.result", points)
    assert not (tmp_path / "new.result").exists()
    assert not list(tmp_path.glob(".simesh-result-*"))


def test_atomic_publication_refuses_races_and_cleans_up_on_failure(tmp_path, points, monkeypatch):
    path = tmp_path / "racing.npz"
    real_link = rio.os.link
    def race(source, destination):
        destination.write_bytes(b"concurrent writer")
        real_link(source, destination)
    monkeypatch.setattr(rio.os, "link", race)
    with pytest.raises(FileExistsError): rio.save_result(path, points)
    assert path.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob(".simesh-result-*"))
    def fail_replace(source, destination):
        raise OSError("injected publication failure")
    monkeypatch.setattr(rio.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication"):
        rio.save_result(path, points, overwrite=True)
    assert path.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob(".simesh-result-*"))


def test_successful_explicit_overwrite_and_dangling_symlink_refusal(tmp_path, points):
    path = rio.save_result(tmp_path / "result.npz", points)
    selected = points.select(np.arange(6) % 2 == 0)
    rio.save_result(path, selected, overwrite=True)
    assert_equivalent(selected, rio.load_result(path).result)
    link = tmp_path / "dangling.npz"
    link.symlink_to(tmp_path / "absent.npz")
    with pytest.raises(FileExistsError): rio.save_result(link, points)
    assert link.is_symlink() and not (tmp_path / "absent.npz").exists()


@pytest.mark.parametrize("quantities", [("q",), ("twist",), ("q", "twist")])
def test_empty_and_batched_connectivity_maps(tmp_path, magnetic, points, quantities):
    empty = sm.PointSet(np.empty((0, 3)))
    roundtrip(tmp_path, app.connectivity(magnetic, empty, quantities=quantities), "empty.npz")
    for index, batch in enumerate(app.iter_connectivity(magnetic, points, quantities=quantities, seed_batch=2)):
        restored = roundtrip(tmp_path, batch, f"batch-{index}.npz").result
        assert restored.points.shape == (2,) and restored.points.plane.shape == (2, 3)
        np.testing.assert_array_equal(restored.points.ids, points.ids[2*index:2*index+2])


def test_variational_flux_and_native_scalar_radius(tmp_path, magnetic, points):
    result = app.connectivity(magnetic, points, quantities=("q",), method="variational",
                              normalization="flux", local_radius=np.float64(.15))
    roundtrip(tmp_path, result)


@pytest.mark.parametrize("kind,damage", [
    ("sample", "values"), ("sample", "owners"), ("sample", "definitions"),
    ("uniform", "values"), ("uniform", "valid"), ("uniform", "upper"),
    ("ray", "values"), ("ray", "samples"), ("ray", "status"), ("ray", "near")])
def test_other_result_contracts_are_validated(tmp_path, magnetic, points, kind, damage):
    if kind == "sample":
        result = app.sample(magnetic, points)
    elif kind == "uniform":
        result = app.uniform_grid(magnetic, (2, 3, 4))
    else:
        result = app.los(magnetic, sm.RaySet(points, [0, 0, 1]), component=2)
    path = rio.save_result(tmp_path / "bad.npz", result)
    def change(manifest, arrays):
        node = manifest["result"]
        if damage == "values": arrays[node[damage]] = arrays[node[damage]][:-1]
        elif damage == "owners": arrays[node[damage]][0] = -2
        elif damage == "definitions": node[damage][0]["units"] = 10
        elif damage == "valid": arrays[node[damage]] = arrays[node[damage]].astype(np.int64)
        elif damage == "upper": arrays[node[damage]][0] = -1e6
        elif damage == "samples": arrays[node[damage]][0] = -1
        elif damage == "status": arrays[node[damage]][0] = 999
        elif damage == "near": arrays[node["rays"][damage]][0] = -1
    rewrite(path, change)
    with pytest.raises(rio.ResultFileError): rio.load_result(path)


def test_invalid_input_keeps_existing_file_and_no_array_aliases_are_loaded(tmp_path, magnetic, points):
    path = rio.save_result(tmp_path / "result.npz", points)
    original = path.read_bytes()
    sample = app.sample(magnetic, points)
    with pytest.raises(rio.ResultFileError):
        rio.save_result(path, replace(sample, values=sample.values[:-1]), overwrite=True)
    with pytest.raises(rio.ResultFileError):
        rio.save_result(path, object(), overwrite=True)
    assert path.read_bytes() == original
    cyclic = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(rio.ResultFileError, match="nesting"):
        rio.save_result(path, points, metadata=cyclic, overwrite=True)
    assert path.read_bytes() == original
    def change(manifest, arrays):
        node = manifest["result"]["plane"]
        node["v"] = node["u"]
    rewrite(path, change)
    with pytest.raises(rio.ResultFileError): rio.load_result(path)


@pytest.mark.parametrize("member", ["a0000.npy.npy", "../a0000.npy", "unexpected.txt"])
def test_ambiguous_and_unexpected_archive_names_are_rejected(tmp_path, points, member):
    path = rio.save_result(tmp_path / "bad.npz", points)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(member, b"unexpected")
    with pytest.raises(rio.ResultFileError): rio.load_result(path)


def test_corrupt_array_payload_crc_is_rejected(tmp_path, points):
    path = rio.save_result(tmp_path / "bad.npz", points)
    # Rewrite without compression so one altered payload byte tests ZIP CRC.
    rewrite(path, lambda manifest, arrays: None)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("a0000.npy")
        offset = info.header_offset
    raw = bytearray(path.read_bytes())
    name_length = int.from_bytes(raw[offset+26:offset+28], "little")
    extra_length = int.from_bytes(raw[offset+28:offset+30], "little")
    payload_end = offset+30+name_length+extra_length+info.compress_size
    raw[payload_end-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(rio.ResultFileError): rio.load_result(path)
