"""Compatibility workflows and explicit crossings into independent fields."""

import gc
import weakref

import numpy as np
import pytest

import simesh as sm
from simesh.amrvac import (open_dataset, load_from_uniform, load_uniform_data,
                          read_blocks, read_uniform, write_datfile_from_uniform,
                          datfile_to_vtk)
from simesh.amrvac.datio import get_metadata
from fixtures import mixed_source, write_dat


def test_dataset_mutation_derived_snapshot_and_file_product(tmp_path):
    x, y, z = (np.indices((8, 8, 8))+.5)/8
    values = np.stack((1+x, 2+y), axis=-1)
    path = tmp_path/'original.dat'
    write_datfile_from_uniform(path, values, ['rho', 'e'], [0, 0, 0], [1, 1, 1], [4, 4, 4],
                               time=12.5, it=17)
    ds = open_dataset(path, ghost_width=2)
    ds.load_data(field_indices=[1, 0])
    ds.register_derived('p', lambda ctx: ctx.field('e')-ctx.field('rho'), dependencies=['e', 'rho'])
    ds.materialize_fields(['p'])
    expected = np.ascontiguousarray(ds.blocks(field_names=['p', 'rho']))
    snapshot = sm.source_from_dataset(ds, ['p', 'rho'], memory_limit=16*1024**2)
    metadata = snapshot.metadata
    assert snapshot.fields[0].interpretation == 'materialized-derived'
    ds.metadata['time'] = 99.
    ds.metadata['params'][0] = 2.
    ds.metadata['xmin'][0] = -100.
    ds.data[...] = 100.
    ds.drop_derived_fields(['p'])
    ref = weakref.ref(ds)
    del ds
    gc.collect()
    assert ref() is None
    assert metadata.time == 12.5 and metadata.iteration == 17
    assert metadata.header['xmin'] == (0., 0., 0.)
    assert metadata.parameters['gamma'] != 2.
    assert metadata.field_names == ('rho', 'e')
    raw = sm.read_fields(snapshot)
    np.testing.assert_array_equal(np.moveaxis(raw.values, -1, 1), expected)
    ready = sm.prepare(snapshot, scheme='exact-phase')
    snapshot.close()
    output = tmp_path/'snapshot.dat'
    sm.write_amrvac(output, ready, metadata=metadata, memory_limit=16*1024**2)
    np.testing.assert_array_equal(read_blocks(output), expected)
    assert get_metadata(output)[0]['w_names'] == ['p', 'rho']
    with sm.open_amrvac(output, fields=('rho',)) as original:
        with sm.select_source(original, ('rho',)) as selected:
            with sm.cache_source(selected, capacity=2) as cached:
                info = original.metadata
                assert selected.metadata is info and cached.metadata is info
                density = sm.read_fields(cached)
        assert info.time == metadata.time and info.parameters == metadata.parameters
        assert info.field_names == ('p', 'rho')
        assert tuple(f.name for f in original.fields) == ('rho',)
        with pytest.raises(AttributeError):
            original.metadata = None
    assert original.metadata is info
    with pytest.raises(TypeError):
        info.header['time'] = 1.
    with pytest.raises(TypeError):
        info.header['xmin'][0] = 1.
    with pytest.raises(TypeError):
        info.parameters['gamma'] = 2.
    mutable_header = info.to_header()
    mutable_header['params'][0] = -1.
    mutable_header['w_names'].clear()
    assert info.parameters['gamma'] == metadata.parameters['gamma'] and info.field_names == ('p', 'rho')
    description = info.to_dict()
    points = sm.PointSet([[.5, .5, .5]])
    result_path = sm.save_result(tmp_path/'points.npz', points, metadata={'snapshot': description})
    assert sm.load_result(result_path).metadata['snapshot'] == description
    description['header']['params'][0] = -2.
    assert info.parameters['gamma'] == metadata.parameters['gamma']
    exported = tmp_path/'density.dat'
    sm.write_amrvac(exported, density, metadata=info)
    np.testing.assert_array_equal(read_blocks(exported), expected[:, 1:2])
    assert get_metadata(exported)[0]['time'] == 12.5


def test_full_export_maps_slots_and_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    import simesh.amrvac.datio as datio
    source, raw = mixed_source()
    assert source.metadata is None
    original = tmp_path/'mixed.dat'
    write_dat(original, source.mesh, raw)
    metadata = get_metadata(original)[0]
    ids = np.arange(source.mesh.leaf_count-1, -1, -1)
    ready = sm.prepare(source, leaf_ids=ids, scheme='exact-phase')
    derived = sm.curl(ready)
    target = tmp_path/'curl.dat'
    header = sm.write_amrvac(target, derived, metadata=metadata)
    expected = np.moveaxis(derived.interior()[derived.slot_of_leaf], -1, 1)
    np.testing.assert_array_equal(read_blocks(target), expected)
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
        datio.write_datfile_from_sfc(target, raw, staggered, source.mesh.node_leaves >= 0,
                                   get_metadata(original)[2], overwrite=True)
    assert target.read_bytes() == before
    def fail(*args, **kwargs):
        raise OSError('injected block write failure')
    monkeypatch.setattr(datio, 'write_blocks', fail)
    with pytest.raises(OSError, match='injected'):
        sm.write_amrvac(target, derived, metadata=metadata, overwrite=True)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob('.simesh-*'))


def test_2d_periodic_metadata_and_vtk_continuity(tmp_path):
    values = np.arange(8*8, dtype=float).reshape(8, 8, 1, 1)
    path = tmp_path/'two.dat'
    write_datfile_from_uniform(path, values, ['rho'], [0, 0], [2, 1], [4, 4])
    actual, geometry = load_uniform_data(path)
    np.testing.assert_array_equal(actual, values)
    assert geometry['ndim'] == 2
    np.testing.assert_array_equal(read_uniform(path, resolution=(8, 8)), values)
    linear = read_uniform(path, resolution=(5, 5), ghost_width=2, interpolation='linear')
    assert linear.shape == (5, 5, 1, 1) and np.isfinite(linear).all()
    vtk = tmp_path/'two.vtk'
    datfile_to_vtk(path, vtk)
    header, payload = vtk.read_bytes().split(b'LOOKUP_TABLE default\n')
    assert b'DIMENSIONS 8 8 1\n' in header
    np.testing.assert_array_equal(np.frombuffer(payload[:-1], dtype='>f8'), values[..., 0].ravel(order='F'))
    periodic = tmp_path/'periodic.dat'
    write_datfile_from_uniform(periodic, values, ['rho'], [0, 0], [2, 1], [4, 4],
                               periodic=np.array([True, False]))
    ds = open_dataset(periodic)
    copy = tmp_path/'periodic-copy.dat'
    ds.write_datfile(copy)
    np.testing.assert_array_equal(get_metadata(copy)[0]['periodic'], [True, False])
    with pytest.raises(ValueError, match='periodic ghost'):
        open_dataset(periodic, ghost_width=2)
    with pytest.raises(ValueError, match='nonperiodic Cartesian 3D'):
        sm.source_from_dataset(ds)
    with pytest.raises(ValueError, match='3D'):
        sm.open_amrvac(path)


def test_compatibility_coarse_block_both_physical_sides(tmp_path):
    source, raw = mixed_source(lambda x, y, z: np.ones((3, *x.shape)))
    path = tmp_path/'constant.dat'
    write_dat(path, source.mesh, raw)
    ds = open_dataset(path, ghost_width=2)
    ds.load_data()
    np.testing.assert_array_equal(ds.blocks(include_ghosts=True), 1.)
    ds.register_derivative('dx', [('b1', 0, 1.)])
    ds.materialize_fields(['dx'])
    np.testing.assert_array_equal(ds.blocks(field_names=['dx']), 0.)
    output = tmp_path/'derived.dat'
    ds.write_datfile(output, field_names=['dx'])
    np.testing.assert_array_equal(read_blocks(output), 0.)


def test_array_tool_is_independent_and_uniform_dataset_roundtrips():
    from simesh.tools import potential_field_green
    bottom = np.zeros((6, 6))
    bottom[2:4, 2:4] = 1.
    field, geometry = potential_field_green(bottom, [0, 0, 0], [1, 1, 1], 4,
                                             backend='direct', balance_flux=False)
    assert np.all(field[2] > 0)
    np.testing.assert_allclose(field[0], -field[0, ::-1], rtol=0, atol=2e-16)
    values = np.moveaxis(field, 0, -1)
    ds = load_from_uniform(values, ['b1', 'b2', 'b3'], geometry.xmin, geometry.xmax, [6, 6, 4])
    np.testing.assert_array_equal(np.moveaxis(ds.uniform_full(), 0, -1), values)
