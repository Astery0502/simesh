"""Core behavior across new source, storage, preparation and consumer boundaries."""

from dataclasses import replace
import numpy as np
import pytest

import simesh as sm
from simesh._amr.blockio import make_block_reader
from fixtures import mixed_source, write_dat


def test_mixed_region_reads_real_support_and_preserves_order():
    source, raw = mixed_source(lambda x, y, z: np.array([np.sin(3*x)+y*y, np.cos(2*y)+z*z, np.sin(x*z)-y]))
    full = sm.prepare(source, ("b3", "b1", "b2"), scheme="exact-phase")
    ids = np.array([8, 2, 1, 5])
    selected = sm.prepare(source, ("b3", "b1", "b2"), leaf_ids=ids, scheme="exact-phase")
    np.testing.assert_array_equal(selected.values, full.values[ids])
    np.testing.assert_array_equal(selected.interior(), np.moveaxis(raw[ids][:, [2, 0, 1]], 1, -1))
    box = np.array([[.99, .20, .20], [1.01, .21, .21]])
    regional = sm.prepare(source, ("b3", "b1", "b2"), region=box, scheme="exact-phase")
    assert len(regional.leaf_ids) < source.mesh.leaf_count
    assert regional.preparation_stats["selected_load_count"] > len(regional.leaf_ids)
    np.testing.assert_array_equal(regional.values, full.values[regional.leaf_ids])
    escaped = regional.values
    source.close()
    np.testing.assert_array_equal(escaped, regional.values)
    assert sm.sample(regional, [box.mean(axis=0)])[2].all()
    with pytest.raises(OSError):
        sm.prepare(source, scheme="exact-phase")
    with pytest.raises(ValueError, match="full-domain"):
        sm.prepare(mixed_source()[0], scheme="coordinate-phase", region=box)


def test_analytic_curl_and_separate_validity_from_storage():
    source, _ = mixed_source()
    ready = sm.prepare(source, scheme="exact-phase")
    reference = sm.curl(ready)
    point = np.array([[1.5, .5, .5], [.25, .25, .25]])
    values, _, valid = sm.sample(reference, point)
    assert valid.all()
    np.testing.assert_allclose(values, [[3., -3., 3.]]*2, rtol=0, atol=2e-13)
    storage = np.full((source.mesh.leaf_count, 14, 14, 14, 3), np.nan)
    storage[:, 1:-1, 1:-1, 1:-1] = ready.values
    padded = replace(ready, _values=storage, storage_halo=3)
    np.testing.assert_array_equal(sm.sample(padded, point)[0], sm.sample(ready, point)[0])
    np.testing.assert_array_equal(sm.curl(padded, workers=4).values, reference.values)
    narrowed = sm.curl(replace(padded, valid_halo=1), workers=2)
    assert narrowed.valid_halo == 0
    np.testing.assert_array_equal(narrowed.values, reference.interior())
    with pytest.raises(ValueError, match="valid halo"):
        sm.sample(narrowed, point)
    assert not np.shares_memory(ready.values, reference.values)
    np.testing.assert_array_equal(sm.curl(ready, workers=4).values, reference.values)
    plane = sm.Plane([.15, .15, .25], [.2, 0, 0], [0, .2, 0], (12, 10))
    image = sm.sample_plane(reference, plane, workers=4)
    assert image.valid.all()
    np.testing.assert_allclose(image.values, np.broadcast_to([3., -3., 3.], image.values.shape),
                               rtol=0, atol=2e-13)


def test_borrow_lifetime_and_failed_fill_never_publishes():
    source, _ = mixed_source()
    ids = [8, 2, 1, 5, 0]
    expected = sm.prepare(source, ("b3", "b1"), leaf_ids=ids, scheme="exact-phase")
    batches = sm.iter_prepared(source, ("b3", "b1"), leaf_ids=ids, scheme="exact-phase", batch_size=2)
    first = next(batches)
    saved = first.values.copy()
    sm.sample(first, source.mesh.bounds[first.leaf_ids].mean(axis=1), workers=4)
    np.testing.assert_array_equal(first.values, saved)
    second = next(batches)
    with pytest.raises(RuntimeError, match="expired"):
        _ = first.values
    np.testing.assert_array_equal(second.values, expected.values[2:4])
    batches.close()
    with pytest.raises(RuntimeError, match="expired"):
        _ = second.values
    for part, batch in enumerate(sm.iter_prepared(source, ("b3", "b1"), leaf_ids=ids,
                                                scheme="exact-phase", batch_size=2)):
        np.testing.assert_array_equal(batch.values, expected.values[part*2:part*2+2])

    def fail(state, lower, upper, leaves, fields, output, offset):
        output.fill(123.)
        raise OSError("injected read failure")

    broken = sm.Source(source.mesh, source.fields,
                       make_block_reader(None, source._reader.shape, fail))
    with pytest.raises(OSError, match="injected"):
        sm.prepare(broken, scheme="exact-phase")
    iterator = sm.iter_prepared(broken, scheme="exact-phase", batch_size=2)
    with pytest.raises(OSError, match="injected"):
        next(iterator)
    with pytest.raises(MemoryError):
        sm.prepare(source, scheme="exact-phase", memory_limit=1)


def test_tracing_preserves_prefixes_parallel_results_and_missing_coverage():
    source, _ = mixed_source(lambda x, y, z: np.array([np.ones_like(x), np.zeros_like(y), np.zeros_like(z)]))
    ready = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.3, .3, .3], [.8, .6, .4], [1.5, .5, .5], [2., .5, .5]])
    serial = sm.trace(ready, seeds, step=.01, max_steps=32, trajectories=True, seed_batch=2)
    parallel = sm.trace(ready, seeds, step=.01, max_steps=32, trajectories=True, seed_batch=3, workers=4)
    for name in ("positions", "length", "steps", "termination", "samples", "trajectories"):
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
    np.testing.assert_allclose(serial.positions[:3], seeds[:3]+[.32, 0, 0], rtol=0, atol=2e-14)
    assert serial.termination[-1] == sm.Termination.OUTSIDE_SEED
    assert serial.point_counts[-1] == 0
    region = sm.prepare(source, leaf_ids=[0], scheme="exact-phase")
    missing = sm.trace(region, np.array([[.4, .25, .25]]), step=.02, max_steps=64)
    assert missing.termination[0] == sm.Termination.MISSING_COVERAGE
    assert missing.steps[0] < 64
    assert not missing.localized_endpoint
    empty = sm.trace(ready, np.empty((0, 3)), step=.01, max_steps=0)
    assert empty.positions.shape == (0, 3)


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("staggered,saved_ghosts", [(False, False), (True, True)])
def test_file_reader_bits_and_preparation_match_array_source(tmp_path, byte_order, staggered, saved_ghosts):
    source, values = mixed_source()
    path = tmp_path/"snapshot.dat"
    write_dat(path, source.mesh, values, byte_order=byte_order, staggered=staggered, saved_ghosts=saved_ghosts)
    with sm.open_amrvac(path) as file_source:
        metadata = file_source.metadata
        assert metadata.byte_order == byte_order and metadata.header['staggered'] == staggered
        assert metadata.path == str(path) and metadata.file_identity[2] == path.stat().st_size
        raw = sm.read_fields(file_source, ("b3", "b1"))
        np.testing.assert_array_equal(raw.interior().view(np.uint64),
                                      np.moveaxis(values[:, [2, 0]], 1, -1).view(np.uint64))
        ids = [8, 1, 5]
        actual = sm.prepare(file_source, ("b2", "b3"), leaf_ids=ids, scheme="exact-phase")
    expected = sm.prepare(source, ("b2", "b3"), leaf_ids=ids, scheme="exact-phase")
    np.testing.assert_array_equal(actual.values, expected.values)
    assert sm.sample(actual, actual.mesh.bounds[ids].mean(axis=1))[2].all()
    output = tmp_path/'ordinary.dat'
    sm.write_amrvac(output, raw, metadata=metadata)
    with sm.open_amrvac(output) as restored:
        assert not restored.metadata.header['staggered']
        assert restored.metadata.field_names == ('b3', 'b1')
        np.testing.assert_array_equal(sm.read_fields(restored).values.view(np.uint64), raw.values.view(np.uint64))


def test_file_change_truncated_tail_and_nonfinite_bits(tmp_path):
    source, values = mixed_source()
    patterns = np.array([0, 0x8000000000000000, 0x7FF0000000000000,
                         0xFFF0000000000000, 0x7FF8000000001234], dtype=np.uint64)
    values.view(np.uint64).reshape(-1)[:len(patterns)] = patterns
    path = tmp_path/"bits.dat"
    write_dat(path, source.mesh, values, byte_order=">", staggered=True, saved_ghosts=True)
    with sm.open_amrvac(path) as file_source:
        raw = sm.read_fields(file_source)
        np.testing.assert_array_equal(raw.interior().view(np.uint64), np.moveaxis(values, 1, -1).view(np.uint64))
        with path.open("ab") as stream:
            stream.write(b"x")
        with pytest.raises(OSError, match="changed"):
            sm.read_fields(file_source)
    write_dat(path, source.mesh, values, staggered=True)
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size-8)
    with sm.open_amrvac(path) as file_source:
        with pytest.raises(ValueError):
            sm.prepare(file_source, leaf_ids=[8], scheme="exact-phase")
