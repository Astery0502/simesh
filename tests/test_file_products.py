"""Native metadata, derived-field serialization and atomic file publication."""

import numpy as np
import pytest

import simesh as sm
from fixtures import mixed_source, write_dat


def test_derived_export_metadata_and_source_lifetime(tmp_path):
    source, raw = mixed_source()
    original = tmp_path/'original.dat'
    write_dat(original, source.mesh, raw)
    with sm.open_amrvac(original) as opened:
        header = opened.metadata.to_header()
    header.update(time=12.5, it=17, n_par=1, params=np.array([5./3]), param_names=['gamma'])
    metadata = sm.SnapshotMetadata(header)
    header['params'][0] = -1.
    header['xmin'][0] = -100.
    assert metadata.parameters['gamma'] == 5./3
    assert metadata.header['xmin'] == (0., 0., 0.)
    fields = sm.read_fields(source)
    derived = sm.derive(fields, 'p', lambda ctx: ctx.field('b2')-ctx.field('b1'))
    combined = sm.merge_fields((derived, sm.select_fields(fields, 'b1')))
    source.close()
    output = tmp_path/'derived.dat'
    sm.write_amrvac(output, combined, metadata=metadata)
    expected = np.stack((raw[:, 1]-raw[:, 0], raw[:, 0]), axis=-1)
    with sm.open_amrvac(output) as written:
        np.testing.assert_array_equal(sm.read_fields(written).values, expected)
        assert written.metadata.field_names == ('p', 'b1')
    with sm.open_amrvac(output, fields=('b1',)) as original_source:
        with sm.select_source(original_source, ('b1',)) as selected:
            with sm.cache_source(selected, capacity=2) as cached:
                info = original_source.metadata
                assert selected.metadata is info and cached.metadata is info
                density = sm.read_fields(cached)
        assert info.time == 12.5 and info.iteration == 17
        assert info.field_names == ('p', 'b1')
        assert tuple(f.name for f in original_source.fields) == ('b1',)
        with pytest.raises(AttributeError):
            original_source.metadata = None
    assert original_source.metadata is info
    with pytest.raises(TypeError):
        info.header['time'] = 1.
    with pytest.raises(TypeError):
        info.header['xmin'][0] = 1.
    mutable_header = info.to_header()
    mutable_header['params'][0] = -1.
    mutable_header['w_names'].clear()
    assert info.parameters['gamma'] == 5./3 and info.field_names == ('p', 'b1')
    description = info.to_dict()
    result_path = sm.save_result(tmp_path/'points.npz', sm.PointSet([[.5, .5, .5]]),
                                metadata={'snapshot': description})
    assert sm.load_result(result_path).metadata['snapshot'] == description
    exported = tmp_path/'density.dat'
    sm.write_amrvac(exported, density, metadata=info)
    with sm.open_amrvac(exported) as written:
        np.testing.assert_array_equal(sm.read_fields(written).values, expected[..., 1:2])
        assert written.metadata.time == 12.5


def test_full_export_maps_slots_and_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    import simesh.io._v5.writer as writer
    source, raw = mixed_source()
    assert source.metadata is None
    original = tmp_path/'mixed.dat'
    write_dat(original, source.mesh, raw)
    with sm.open_amrvac(original) as original_source:
        metadata = original_source.metadata.to_header()
    ids = np.arange(source.mesh.leaf_count-1, -1, -1)
    ready = sm.prepare(source, leaf_ids=ids, scheme='exact-phase')
    derived = sm.curl(ready)
    target = tmp_path/'curl.dat'
    header = sm.write_amrvac(target, derived, metadata=metadata)
    expected = np.moveaxis(derived.interior()[derived.slot_of_leaf], -1, 1)
    with sm.open_amrvac(target) as written:
        np.testing.assert_array_equal(np.moveaxis(sm.read_fields(written).values, -1, 1), expected)
    assert header['nparents'] == 1
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        sm.write_amrvac(target, derived, metadata=metadata)
    with pytest.raises(MemoryError):
        sm.write_amrvac(target, derived, metadata=metadata, overwrite=True, memory_limit=1)
    partial = sm.prepare(source, leaf_ids=[0], scheme='exact-phase')
    with pytest.raises(ValueError, match='complete'):
        sm.write_amrvac(target, partial, metadata=metadata, overwrite=True)
    staggered = dict(metadata, staggered=True)
    with pytest.raises(ValueError, match='staggered'):
        writer.write_datfile_from_sfc(None, raw, staggered, source.mesh.node_leaves >= 0, ())
    assert target.read_bytes() == before
    def fail(*args, **kwargs):
        raise OSError('injected block write failure')
    monkeypatch.setattr(writer, '_write_blocks', fail)
    with pytest.raises(OSError, match='injected'):
        sm.write_amrvac(target, derived, metadata=metadata, overwrite=True)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob('.simesh-*'))


def test_array_tool_connects_to_native_source():
    from simesh.tools import potential_field_green
    bottom = np.zeros((6, 6))
    bottom[2:4, 2:4] = 1.
    field, geometry = potential_field_green(bottom, [0, 0, 0], [1, 1, 1], 4,
                                          backend='direct', balance_flux=False)
    assert np.all(field[2] > 0)
    np.testing.assert_allclose(field[0], -field[0, ::-1], rtol=0, atol=2e-16)
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]), lower=geometry.xmin,
                              upper=geometry.xmax, block_shape=field.shape[1:])
    with sm.source_from_arrays(mesh, field[np.newaxis], ('b1', 'b2', 'b3')) as source:
        fields = sm.read_fields(source)
    np.testing.assert_array_equal(fields.interior()[0], np.moveaxis(field, 0, -1))


def test_export_matches_v5_bytes_and_preserves_float_bits(tmp_path):
    source, original = mixed_source()
    mesh = source.mesh
    source.close()
    patterns = np.array([0, 0x8000000000000000, 0x3ff0000000000000,
                         0x7ff0000000000000, 0xfff0000000000000,
                         0x7ff8000000001234, 0x7ff0000000001234], dtype=np.uint64)
    raw = np.resize(patterns, original.shape).view(np.float64)
    expected = tmp_path/'expected.dat'
    write_dat(expected, mesh, raw, byte_order='=')
    with sm.open_amrvac(expected) as stored:
        metadata = stored.metadata
    with sm.source_from_arrays(mesh, raw, ('b1', 'b2', 'b3'), copy=False) as source:
        fields = sm.read_fields(source, leaf_ids=np.arange(mesh.leaf_count)[::-1])
    actual = tmp_path/'actual.dat'
    sm.write_amrvac(actual, fields, metadata=metadata)
    assert actual.read_bytes() == expected.read_bytes()
