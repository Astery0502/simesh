"""Interior-only placement, containing-cell sampling and bounded source delivery."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from fixtures import mixed_source, write_dat


def uniform_source():
    mesh = sm.mesh_from_forest((2, 1, 1), np.ones(2, dtype=bool),
        lower=(0, 0, 0), upper=(2, 1, 1), block_shape=(4, 4, 4))
    raw = np.arange(2*3*4**3, dtype=float).reshape(2, 3, 4, 4, 4)
    raw.view('u8')[0, 0, 1, 2, 3] = 0x7ff8000000001234
    raw[1, 2, 0, 0, 0] = -0.
    return sm.source_from_arrays(mesh, raw, ('b1', 'b2', 'b3'), copy=False), raw


def reference(fields, shape, bounds, components):
    lower, upper = np.asarray(bounds)
    spacing = (upper-lower)/np.asarray(shape)
    axes = [lo+(np.arange(n)+.5)*dx for lo, dx, n in zip(lower, spacing, shape)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    owners = fields.mesh.locate(points)
    expected = np.full((len(points), len(components)), np.nan)
    valid = np.zeros(len(points), dtype=bool)
    for i, (point, leaf) in enumerate(zip(points, owners)):
        if leaf < 0 or fields.slot_of_leaf[leaf] < 0:
            continue
        cell = np.floor((point-fields.mesh.bounds[leaf, 0])/fields.mesh.spacing[leaf]).astype(int)
        cell = np.clip(cell, 0, np.asarray(fields.mesh.block_shape)-1)+fields.storage_halo
        expected[i] = fields.values[(fields.slot_of_leaf[leaf], *cell)][components]
        valid[i] = True
    return expected.reshape(*shape, len(components)), valid.reshape(shape), owners.reshape(shape)


@pytest.mark.parametrize('workers', [1, 3])
def test_zero_mixed_faces_partial_coverage_and_invalid_padding(workers):
    with mixed_source()[0] as source:
        interior = sm.read_fields(source, leaf_ids=[8, 3, 0])
    padded = np.full((3, 12, 12, 12, 3), 987654.)
    padded[:, 2:-2, 2:-2, 2:-2] = interior.values
    fields = replace(interior, _values=padded, storage_halo=2)
    # Cell centers hit domain and coarse/fine faces exactly.
    shape, bounds = (9, 5, 5), ((-.125, -.125, -.125), (2.125, 1.125, 1.125))
    result = app.uniform_grid(fields, shape, bounds=bounds, components=('b3', 'b1'),
                              interpolation='zero', workers=workers)
    values, valid, owners = reference(fields, shape, bounds, [2, 0])
    np.testing.assert_array_equal(result.values, values)
    np.testing.assert_array_equal(result.valid, valid)
    slices = list(sm.iter_uniform(fields, shape, bounds=bounds, components=(2, 0),
                                 interpolation='zero', workers=workers))
    for iz, slab in slices:
        np.testing.assert_array_equal(slab.values, values[:, :, iz])
        np.testing.assert_array_equal(slab.valid, valid[:, :, iz])
        np.testing.assert_array_equal(slab.owners, owners[:, :, iz])
    with pytest.raises(ValueError, match='valid_halo=0'):
        app.uniform_grid(fields, shape)


@pytest.mark.parametrize('mode', ['native', 'zero'])
def test_uniform_bitwise_placement_source_and_fields(tmp_path, mode):
    source, raw = uniform_source()
    path = tmp_path/'uniform.dat'
    write_dat(path, source.mesh, raw, saved_ghosts=True)
    expected = np.moveaxis(np.concatenate(raw[:, [2, 0]], axis=1), 0, -1)
    with source:
        fields = sm.read_fields(source, leaf_ids=[1, 0])
        result = app.uniform_grid(fields, (8, 4, 4), components=(2, 0), interpolation=mode, workers=2)
        streamed = sm.export_uniform(source, (8, 4, 4), fields=('b3', 'b1'), interpolation=mode, batch_size=1)
        source.validate()
    direct = sm.export_uniform(path, (8, 4, 4), fields=(2, 0), interpolation=mode, batch_size=1)
    for grid in (result, streamed, direct):
        np.testing.assert_array_equal(grid.values.view('u8'), expected.view('u8'))
        assert grid.valid.all()
        assert not grid.usable.all()
    slices = list(sm.iter_uniform(fields, (8, 4, 4), components=(2, 0), interpolation=mode))
    for iz, slab in slices:
        np.testing.assert_array_equal(slab.values.view('u8'), expected[:, :, iz].view('u8'))
    cropped = app.uniform_grid(fields, (4, 2, 2), interpolation=mode, components=(2, 0),
                               bounds=((.5, .25, .25), (1.5, .75, .75)))
    np.testing.assert_array_equal(cropped.values.view('u8'), expected[2:6, 1:3, 1:3].view('u8'))


def test_zero_matches_containing_cells_and_streams_selected_fields(tmp_path):
    source, raw = mixed_source()
    path = tmp_path/'mixed.dat'
    write_dat(path, source.mesh, raw, saved_ghosts=True, byte_order='>')
    shape = (19, 13, 11)
    bounds = ((.13, .08, .17), (1.81, .91, .94))
    expected, expected_valid, _ = reference(sm.read_fields(source), shape, bounds, [2, 0])
    assert expected_valid.all()
    values = np.lib.format.open_memmap(tmp_path/'values.npy', mode='w+', dtype=float, shape=(*shape, 2))
    valid = np.lib.format.open_memmap(tmp_path/'valid.npy', mode='w+', dtype=bool, shape=shape)
    with sm.open_amrvac(path) as current:
        original = current._read_native
        reads = []
        def read(ids, columns, target):
            assert len(ids) <= 2 and target.shape[1:4] == (8, 8, 8)
            np.testing.assert_array_equal(columns, [2, 0])
            reads.extend(ids.tolist())
            return original(ids, columns, target)
        current._read_native = read
        result = sm.export_uniform(current, shape, bounds=bounds, fields=(2, 0),
            output=(values, valid), batch_size=2, workers=2)
        assert len(reads) == len(set(reads)) == source.mesh.leaf_count
    assert result.values is values and result.valid is valid
    np.testing.assert_array_equal(values, expected)
    assert valid.all()
    values.flush()
    np.testing.assert_array_equal(np.load(tmp_path/'values.npy'), expected)
    source.close()


def test_native_partial_and_rejections_before_output_writes():
    with uniform_source()[0] as source:
        fields = sm.read_fields(source, leaf_ids=[1])
        result = app.uniform_grid(fields, (8, 4, 4), interpolation='native')
        assert not result.valid[:4].any() and result.valid[4:].all()
        assert np.isnan(result.values[:4]).all()
        output = (np.full((8, 4, 4, 3), 31.), np.ones((8, 4, 4), dtype=bool))
        for operation in (
            lambda: sm.export_uniform(source, (8, 4, 4), output=output, memory_limit=1),
            lambda: app.uniform_grid(fields, (8, 4, 4), output=output, interpolation='bad'),
            lambda: sm.export_uniform(source, (8, 4, 4), output=output, interpolation='linear'),
            lambda: sm.export_uniform(source, (8, 4, 4), output=output, interpolation='native',
                                      bounds=((.01, 0, 0), (2.01, 1, 1))),
        ):
            with pytest.raises((ValueError, MemoryError)):
                operation()
            assert np.all(output[0] == 31.) and output[1].all()
        with pytest.raises(ValueError, match='alias'):
            sm.export_uniform(source, (8, 4, 4), output=(source._memory_arrays[0].reshape(8, 4, 4, 3), output[1]))
    with pytest.raises(OSError, match='closed'):
        sm.export_uniform(source, (8, 4, 4))
    with mixed_source()[0] as source:
        with pytest.raises(ValueError, match='matching cell spacing'):
            sm.export_uniform(source, (16, 8, 8), interpolation='native')


def test_categorical_and_nonpacked_slots():
    with uniform_source()[0] as source:
        fields = sm.read_fields(source)
    fields = replace(fields, fields=tuple(sm.FieldDefinition(f.name, interpretation='categorical-label') for f in fields.fields))
    selection = sm.Selection(fields.mesh, [1, 0])
    fields = replace(fields, selection=selection)
    zero = app.uniform_grid(fields, (8, 4, 4), interpolation='zero')
    native = app.uniform_grid(fields, (8, 4, 4), interpolation='native')
    np.testing.assert_array_equal(zero.values.view('u8'), native.values.view('u8'))


@pytest.mark.parametrize('bounds', [((.13, .11, .17), (1.87, .89, .93)),
                                   ((-.125, -.125, -.25), (2.125, 1.125, 1.25))])
def test_linear_file_export_matches_prepared_fields(tmp_path, bounds):
    source, raw = mixed_source()
    path = tmp_path/'linear.dat'
    write_dat(path, source.mesh, raw, saved_ghosts=True, byte_order='>')
    shape = (9, 5, 3)
    with source:
        ready = sm.prepare(source, ('b3', 'b1'), scheme='exact-phase')
        expected = app.uniform_grid(ready, shape, bounds=bounds, tile_rows=2)
        zero = sm.export_uniform(source, shape, fields=('b3', 'b1'), bounds=bounds)
    values = np.lib.format.open_memmap(tmp_path/'linear.npy', mode='w+', dtype=float, shape=(*shape, 2))
    valid = np.full(shape, False)
    result = sm.export_uniform(path, shape, fields=('b3', 'b1'), bounds=bounds,
        interpolation='linear', scheme='exact-phase', batch_size=1, tile_rows=2,
        workers=2, output=(values, valid))
    assert result.values is values and result.valid is valid
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.valid, expected.valid)
    assert np.any(result.values[result.valid] != zero.values[result.valid])
    assert [f.name for f in result.definitions] == ['b3', 'b1']
    values.flush()
    np.testing.assert_array_equal(np.load(tmp_path/'linear.npy'), expected.values)


def test_linear_source_lifetime_capacity_and_failed_admission(monkeypatch):
    from simesh.preparation import exact
    source, raw = mixed_source()
    shape = (7, 4, 3)
    with source:
        expected = app.uniform_grid(sm.prepare(source, scheme='exact-phase'), shape)
        original_fill = exact.fill
        batch_counts = []
        def fill(current, selection, fields, output, workspace):
            batch_counts.append(len(selection.leaf_ids))
            return original_fill(current, selection, fields, output, workspace)
        monkeypatch.setattr(exact, 'fill', fill)
        result = sm.export_uniform(source, shape, interpolation='linear', scheme='exact-phase',
                                   batch_size=2, tile_rows=2)
        assert batch_counts and max(batch_counts) <= 2
        source.validate()
        np.testing.assert_array_equal(result.values, expected.values)
        output = (np.full((*shape, 3), 41.), np.ones(shape, dtype=bool))
        batch_counts.clear()
        for controls in ({'memory_limit': 1}, {'support_capacity': 0}, {'tile_rows': 0}):
            with pytest.raises((ValueError, MemoryError)):
                sm.export_uniform(source, shape, interpolation='linear', scheme='exact-phase',
                                   output=output, **controls)
            assert not batch_counts
            assert np.all(output[0] == 41.) and output[1].all()
        for scheme in (None, 'coordinate-phase'):
            with pytest.raises(ValueError, match="explicit scheme='exact-phase'"):
                sm.export_uniform(source, shape, interpolation='linear', scheme=scheme, output=output)
            assert np.all(output[0] == 41.)
    np.testing.assert_array_equal(result.values, expected.values)
    categorical = [sm.FieldDefinition(name, interpretation='categorical-label') for name in ('b1', 'b2', 'b3')]
    with sm.source_from_arrays(source.mesh, raw, categorical) as source:
        with pytest.raises(ValueError, match='continuous'):
            sm.export_uniform(source, shape, interpolation='linear', scheme='exact-phase')
