"""Uniform VTK geometry, payload ordering, coverage and atomic publication."""

from dataclasses import replace
import io

import numpy as np
import pytest

import simesh as sm
from simesh import applications as app
from fixtures import mixed_source, write_dat


def read_vtk(path):
    with path.open('rb') as stream:
        assert stream.readline() == b'# vtk DataFile Version 3.0\n'
        stream.readline()
        assert stream.readline() == b'BINARY\n'
        assert stream.readline() == b'DATASET STRUCTURED_POINTS\n'
        dimensions = tuple(map(int, stream.readline().split()[1:]))
        origin = np.array(list(map(float, stream.readline().split()[1:])))
        spacing = np.array(list(map(float, stream.readline().split()[1:])))
        shape = tuple(n-1 for n in dimensions)
        assert stream.readline().split() == [b'CELL_DATA', str(np.prod(shape)).encode()]
        arrays = {}
        while line := stream.readline():
            keyword, name, kind, components = line.split()
            assert keyword == b'SCALARS' and components == b'1'
            assert stream.readline() == b'LOOKUP_TABLE default\n'
            dtype = np.dtype('>f8' if kind == b'double' else 'u1')
            payload = stream.read(int(np.prod(shape))*dtype.itemsize)
            arrays[name.decode()] = np.frombuffer(payload, dtype=dtype).reshape(shape, order='F')
            assert stream.read(1) == b'\n'
    return shape, origin, spacing, arrays


def small_grid():
    values = np.arange(4*3*2*2, dtype=float).reshape(4, 3, 2, 2)[::2, :, :1]
    values[0, 0, 0, 0] = -0.
    values.view('u8')[0, 1, 0, 0] = 0x7ff8000000001234
    valid = np.ones((4, 3, 2), dtype=bool)[::2, :, :1]
    valid[1, 2, 0] = False
    return app.UniformResult(values, valid, np.array([-2., .5, 4.]), np.array([-1., 5., 6.]),
                             (sm.FieldDefinition('rho'), sm.FieldDefinition('b3')), None)


def test_uniform_vtk_cell_geometry_strides_and_float_bits(tmp_path, monkeypatch):
    import simesh.io.vtk as vtk
    monkeypatch.setattr(vtk, '_BUFFER_BYTES', 16)
    grid = small_grid()
    path = sm.write_uniform_vtk(tmp_path/'grid.vtk', grid)
    shape, origin, spacing, arrays = read_vtk(path)
    assert shape == grid.valid.shape
    np.testing.assert_array_equal(origin, grid.lower)
    np.testing.assert_array_equal(spacing, grid.spacing)
    assert list(arrays) == ['rho', 'b3', 'simesh_valid']
    for column, name in enumerate(('rho', 'b3')):
        actual = arrays[name].astype(np.float64)
        np.testing.assert_array_equal(actual.view('u8'), grid.values[..., column].view('u8'))
    np.testing.assert_array_equal(arrays['simesh_valid'], grid.valid)
    assert arrays['simesh_valid'][0, 1, 0] == 1 and np.isnan(arrays['rho'][0, 1, 0])


@pytest.mark.parametrize('interpolation', ['zero', 'native', 'linear'])
def test_file_to_vtk_reuses_uniform_export(tmp_path, interpolation):
    source, raw = mixed_source()
    mesh = source.mesh
    source.close()
    if interpolation == 'native':
        mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]),
            lower=(0, 0, 0), upper=(1, 1, 1), block_shape=(4, 4, 4))
        raw = np.arange(3*4**3, dtype=float).reshape(1, 3, 4, 4, 4)
    snapshot = tmp_path/'snapshot.dat'
    write_dat(snapshot, mesh, raw, saved_ghosts=True, byte_order='>')
    shape = (4, 4, 4)
    options = dict(fields=('b3', 'b1'), interpolation=interpolation, batch_size=1,
                   scheme='exact-phase' if interpolation == 'linear' else None, tile_rows=2)
    expected = sm.export_uniform(snapshot, shape, **options)
    values = np.lib.format.open_memmap(tmp_path/'values.npy', mode='w+',
                                       dtype=float, shape=(*shape, 2))
    valid = np.empty(shape, dtype=bool)
    with sm.open_amrvac(snapshot) as source:
        output = sm.export_uniform_vtk(source, tmp_path/'grid.vtk', shape,
                                       output=(values, valid), **options)
        source.validate()
    direct = sm.export_uniform_vtk(snapshot, tmp_path/'direct.vtk', shape, **options)
    assert direct.read_bytes() == output.read_bytes()
    _, origin, spacing, arrays = read_vtk(output)
    np.testing.assert_array_equal(origin, expected.lower)
    np.testing.assert_array_equal(spacing, expected.spacing)
    np.testing.assert_array_equal(arrays['b3'], expected.values[..., 0])
    np.testing.assert_array_equal(arrays['b1'], expected.values[..., 1])
    np.testing.assert_array_equal(arrays['simesh_valid'], expected.valid)
    np.testing.assert_array_equal(values, expected.values)


def test_vtk_validation_and_failed_write_preserve_destination(tmp_path, monkeypatch):
    import simesh.io.vtk as vtk
    grid = small_grid()
    path = tmp_path/'grid.vtk'
    path.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        sm.export_uniform_vtk(tmp_path/'missing.dat', path, (2, 3, 1))
    for invalid in (object(), replace(grid, upper=grid.lower),
                    replace(grid, values=grid.values[..., :1]),
                    replace(grid, definitions=(sm.FieldDefinition('rho'),)*2),
                    replace(grid, definitions=(sm.FieldDefinition('simesh_valid'), sm.FieldDefinition('b3'))),
                    replace(grid, definitions=(sm.FieldDefinition('bad name'), sm.FieldDefinition('b3')))):
        with pytest.raises((TypeError, ValueError)):
            sm.write_uniform_vtk(path, invalid, overwrite=True)
        assert path.read_bytes() == b'existing'
    with pytest.raises(MemoryError):
        sm.write_uniform_vtk(path, grid, overwrite=True, memory_limit=1)
    def fail(stream, *args):
        stream.write(b'partial')
        raise OSError('injected write failure')
    monkeypatch.setattr(vtk, '_write_array', fail)
    with pytest.raises(OSError, match='injected'):
        sm.write_uniform_vtk(path, grid, overwrite=True)
    assert path.read_bytes() == b'existing'
    assert not list(tmp_path.glob('.simesh-vtk-*'))


def test_vtk_buffered_conversion_is_bounded(monkeypatch):
    import simesh.io.vtk as vtk
    monkeypatch.setattr(vtk, '_BUFFER_BYTES', 32)
    array = np.arange(6*5*4., dtype=float).reshape(6, 5, 4)[::2]
    class Stream(io.BytesIO):
        def write(self, payload):
            assert len(payload) <= 32
            return super().write(payload)
    stream = Stream()
    vtk._write_array(stream, array, '>f8')
    assert stream.getvalue() == array.astype('>f8').tobytes(order='F') + b'\n'
