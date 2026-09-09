"""Root-aligned ordinary-field export, source lifetime and bounded payload I/O."""

import io
import os

import numpy as np
import pytest

import simesh as sm
from simesh._amr.morton import level1_morton
from simesh.io import products
from simesh.io._v5 import reader, writer
from fixtures import write_dat


BOX = ((1, 1, 0), (4, 3, 2))


def snapshot(tmp_path, *, byte_order='<', saved_ghosts=False, staggered=False, bits=False):
    _, coordinates = level1_morton(np.array((4, 3, 2), dtype=np.int64))
    flags = np.concatenate([np.array([False]+[True]*8 if sum(c)%3 == 0 else [True])
                            for c in coordinates])
    mesh = sm.mesh_from_forest((4, 3, 2), flags, lower=(-2., -3., 1.),
                              upper=(6., 3., 5.), block_shape=(4, 4, 4))
    raw = np.arange(mesh.leaf_count*3*64, dtype=np.float64).reshape(mesh.leaf_count, 3, 4, 4, 4)
    if bits:
        patterns = np.array([0, 0x8000000000000000, 0x7ff0000000000000,
                             0xfff0000000000000, 0x7ff8000000001234,
                             0x7ff0000000001234], dtype=np.uint64)
        raw = np.resize(patterns, raw.shape).view(np.float64)
    path = tmp_path/'input.dat'
    write_dat(path, mesh, raw, byte_order=byte_order, saved_ghosts=saved_ghosts, staggered=staggered)
    return path, mesh, raw


def assert_crop(original, cropped, ids, box):
    old = original.forest
    nodes = original.leaf_nodes[ids]
    new = cropped.forest
    levels = old.node_levels[nodes]
    np.testing.assert_array_equal(new.node_levels[cropped.leaf_nodes], levels)
    np.testing.assert_array_equal(new.node_coords[cropped.leaf_nodes],
                                  old.node_coords[nodes] - np.array(box[0])*(1 << (levels-1))[:, None])
    np.testing.assert_allclose(cropped.bounds, original.bounds[ids], rtol=0, atol=1e-14)
    np.testing.assert_allclose(cropped.spacing, original.spacing[ids], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(cropped.root_shape, np.subtract(box[1], box[0]))
    # Every nonleaf remains a complete octree node, with unchanged parent levels.
    parents = np.flatnonzero(new.node_leaf_ids < 0)
    assert np.all(new.child_node_ids[parents] >= 0)
    np.testing.assert_array_equal(new.node_levels[new.child_node_ids[parents]],
                                  np.broadcast_to(new.node_levels[parents, None]+1, (len(parents), 8)))


@pytest.mark.parametrize('box', [BOX, ((1, 0, 0), (2, 1, 1)), ((0, 0, 0), (4, 3, 2))])
def test_selection_and_both_export_paths(tmp_path, box):
    path, mesh, raw = snapshot(tmp_path)
    with sm.open_amrvac(path) as source:
        selection = sm.select_roots(source.mesh, box)
        metadata = source.metadata
        fields = sm.read_fields(source, ('b3', 'b1'), region=selection)
        assert selection.mesh is fields.mesh is source.mesh
    ids = selection.leaf_ids
    if box == BOX:
        assert not np.array_equal(ids, np.sort(ids))
    assert fields.valid_halo == fields.storage_halo == 0
    detached = tmp_path/'fields.dat'
    direct = tmp_path/'direct.dat'
    sm.write_amrvac(detached, fields, metadata=metadata, root_bounds=box)
    sm.crop_amrvac(path, direct, root_bounds=box, fields=('b3', 'b1'))
    assert detached.read_bytes() == direct.read_bytes()
    with sm.open_amrvac(direct) as output:
        assert_crop(mesh, output.mesh, ids, box)
        values = sm.read_fields(output)
        np.testing.assert_array_equal(np.moveaxis(values.values, -1, 1), raw[ids][:, [2, 0]])
        assert output.metadata.field_names == ('b3', 'b1')
        # Native analysis agrees over the same selected cell volumes.
        before = sm.volume_integral(fields, 'b3')
        after = sm.volume_integral(values, 'b3')
        np.testing.assert_allclose(after.value, before.value, rtol=1e-14)


@pytest.mark.parametrize('byte_order,saved_ghosts,staggered', [('<', False, False), ('>', True, False),
                                                           ('<', True, True), ('>', False, True)])
def test_saved_records_and_field_bit_patterns(tmp_path, byte_order, saved_ghosts, staggered):
    path, mesh, raw = snapshot(tmp_path, byte_order=byte_order, saved_ghosts=saved_ghosts,
                               staggered=staggered, bits=True)
    target = tmp_path/'crop.dat'
    header = sm.crop_amrvac(path, target, root_bounds=BOX, fields=[2, 0])
    ids = sm.select_roots(mesh, BOX).leaf_ids
    assert not header['staggered']
    with sm.open_amrvac(target) as output:
        actual = np.moveaxis(sm.read_fields(output).values, -1, 1)
        np.testing.assert_array_equal(actual.view(np.uint64), raw[ids][:, [2, 0]].view(np.uint64))
    assert target.stat().st_size == header['offset_blocks'] + len(ids)*(24+2*64*8)


def test_multilevel_subtree_is_not_flattened(tmp_path):
    # One root with all level-2 children refined to level 3; neighbors are level 2.
    flags = np.array([False] + ([False]+[True]*8)*8 + [False]+[True]*8)
    mesh = sm.mesh_from_forest((2, 1, 1), flags, lower=(.1, .2, .3),
                              upper=(1.7, 1.3, 2.4), block_shape=(4, 4, 4))
    raw = np.arange(mesh.leaf_count*64, dtype=float).reshape(mesh.leaf_count, 1, 4, 4, 4)
    path, target = tmp_path/'levels.dat', tmp_path/'subtree.dat'
    write_dat(path, mesh, raw)
    box = ((0, 0, 0), (1, 1, 1))
    sm.crop_amrvac(path, target, root_bounds=box)
    with sm.open_amrvac(target) as output:
        assert output.mesh.forest.max_level == 3
        assert len(output.mesh.node_leaves) - output.mesh.leaf_count == 9
        assert_crop(mesh, output.mesh, np.arange(64), box)
        np.testing.assert_array_equal(np.moveaxis(sm.read_fields(output).values, -1, 1), raw[:64])


@pytest.mark.parametrize('box', [((-1, 0, 0), (2, 1, 1)), ((0, 0, 0), (5, 1, 1)),
                               ((0, 0, 0), (0, 1, 1)), ((0., 0., 0.), (1., 1., 1.)),
                               ((False, False, False), (True, True, True)), ((0, 0), (1, 1))])
def test_invalid_root_bounds(tmp_path, box):
    path, mesh, _ = snapshot(tmp_path)
    with pytest.raises(ValueError, match='root_bounds'):
        sm.select_roots(mesh, box)
    with pytest.raises(ValueError, match='root_bounds'):
        sm.crop_amrvac(path, tmp_path/'invalid.dat', root_bounds=box)
    assert not (tmp_path/'invalid.dat').exists()


def test_coverage_nonpacked_fields_and_derived_columns(tmp_path):
    path, _, _ = snapshot(tmp_path)
    with sm.open_amrvac(path) as source:
        ids = sm.select_roots(source.mesh, BOX).leaf_ids
        fields = sm.read_fields(source, leaf_ids=np.arange(source.mesh.leaf_count)[::-1])
        metadata = source.metadata
        missing = sm.read_fields(source, leaf_ids=ids[:-1])
    with pytest.raises(ValueError, match='selected-root coverage'):
        sm.write_amrvac(tmp_path/'missing.dat', missing, metadata=metadata, root_bounds=BOX)
    with pytest.raises(ValueError, match='original-mesh coverage'):
        sm.write_amrvac(tmp_path/'missing.dat', missing, metadata=metadata)
    derived = sm.derive(fields, 'sum', lambda ctx: ctx.field('b1')+ctx.field('b3'))
    # Selection ordering need not equal storage-row order.
    from dataclasses import replace
    reordered = replace(derived, selection=sm.Selection(derived.mesh, derived.leaf_ids[::-1]))
    target = tmp_path/'derived.dat'
    sm.write_amrvac(target, reordered, metadata=metadata, root_bounds=BOX)
    with sm.open_amrvac(target) as output:
        expected = np.stack([reordered.window(leaf, (0, 0, 0), (4, 4, 4)) for leaf in ids])
        np.testing.assert_array_equal(sm.read_fields(output).values, expected)
        assert output.metadata.field_names == ('sum',)


def test_selected_payload_io_is_bounded_and_not_published(tmp_path, monkeypatch):
    path, mesh, raw = snapshot(tmp_path)
    monkeypatch.setattr(products, '_BATCH_LEAVES', 3)
    reads = []
    original = sm.Source.read_into
    def observe(self, leaves, fields, output):
        reads.append((leaves.copy(), fields.copy(), output.shape, output.ctypes.data))
        return original(self, leaves, fields, output)
    monkeypatch.setattr(sm.Source, 'read_into', observe)
    def forbid(*args, **kwargs):
        raise AssertionError('file crop must not publish Fields or build another Mesh')
    import simesh.io.source as source_module
    monkeypatch.setattr(source_module, 'publish', forbid)
    monkeypatch.setattr(sm, 'mesh_from_forest', forbid)
    intervals = []
    original_pread = reader._pread_exact
    def pread(fd, count, offset, *, section):
        if section == 'payload':
            intervals.append((offset, offset+count))
        return original_pread(fd, count, offset, section=section)
    monkeypatch.setattr(reader, '_pread_exact', pread)
    sm.crop_amrvac(path, tmp_path/'bounded.dat', root_bounds=BOX, fields=[2, 0])
    ids = sm.select_roots(mesh, BOX).leaf_ids
    np.testing.assert_array_equal(np.concatenate([read[0] for read in reads]), ids)
    assert max(len(read[0]) for read in reads) <= 3
    assert len({read[3] for read in reads}) == 1
    assert all(np.array_equal(read[1], [0, 1]) for read in reads)
    from simesh.io._v5.index import read_amrvac_v5_index
    fd = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(fd)
    finally:
        os.close(fd)
    expected = {(int(index.block_offsets[leaf])+24+field*64*8,
                 int(index.block_offsets[leaf])+24+(field+1)*64*8)
                for leaf in ids for field in [2, 0]}
    assert set(intervals) == expected
    assert sum(stop-start for start, stop in intervals) == len(ids)*2*64*8


def test_failures_preserve_files_and_clean_temporary_output(tmp_path, monkeypatch):
    path, _, _ = snapshot(tmp_path)
    target = tmp_path/'existing.dat'
    target.write_bytes(b'existing output')
    for kwargs, error in [({}, FileExistsError), ({'overwrite': True, 'memory_limit': 1}, MemoryError),
                          ({'overwrite': True, 'fields': []}, ValueError),
                          ({'overwrite': True, 'fields': [0, 0]}, ValueError),
                          ({'overwrite': True, 'fields': ['unknown']}, ValueError)]:
        with pytest.raises(error):
            sm.crop_amrvac(path, target, root_bounds=BOX, **kwargs)
        assert target.read_bytes() == b'existing output'
    with pytest.raises(ValueError, match='different files'):
        sm.crop_amrvac(path, path, root_bounds=BOX, overwrite=True)
    alias = tmp_path/'alias.dat'
    os.link(path, alias)
    with pytest.raises(ValueError, match='different files'):
        sm.crop_amrvac(path, alias, root_bounds=BOX, overwrite=True)
    monkeypatch.setattr(products, '_BATCH_LEAVES', 2)
    real_write = writer._write_blocks
    calls = 0
    def fail(stream, data):
        nonlocal calls
        calls += 1
        real_write(stream, data)
        if calls == 2:
            raise OSError('injected midstream failure')
    monkeypatch.setattr(writer, '_write_blocks', fail)
    with pytest.raises(OSError, match='midstream'):
        sm.crop_amrvac(path, target, root_bounds=BOX, overwrite=True)
    assert target.read_bytes() == b'existing output'
    assert not list(tmp_path.glob('.simesh-*'))


def test_batch_budget_and_serializer_count_validation(tmp_path, monkeypatch):
    path, _, _ = snapshot(tmp_path)
    with sm.open_amrvac(path) as source:
        selection = sm.select_roots(source.mesh, BOX)
        fields = sm.read_fields(source, region=selection)
        metadata = source.metadata
    plan = products._export_plan(fields.mesh, metadata, [f.name for f in fields.fields], BOX,
                                 fields.mesh.nbytes+fields.nbytes, None)
    block_bytes = 3*64*8
    budget = plan[4] + 3*block_bytes
    seen = []
    real_write = writer._write_blocks
    def observe(stream, data):
        seen.append(len(data))
        real_write(stream, data)
    monkeypatch.setattr(writer, '_write_blocks', observe)
    target = tmp_path/'budget.dat'
    sm.write_amrvac(target, fields, metadata=metadata, root_bounds=BOX, memory_limit=budget)
    assert max(seen) == 2
    with pytest.raises(MemoryError, match='batch'):
        sm.write_amrvac(target, fields, metadata=metadata, root_bounds=BOX,
                        overwrite=True, memory_limit=plan[4]+block_bytes)
    header, flags, tree, ids, _ = plan
    block = np.zeros((1, 3, 4, 4, 4))
    with pytest.raises(ValueError, match='cover all'):
        writer.write_datfile_from_batches(io.BytesIO(), [block], header, flags, tree)
    with pytest.raises(ValueError, match='shape and count'):
        writer.write_datfile_from_batches(io.BytesIO(), [np.zeros((len(ids)+1, 3, 4, 4, 4))],
                                          header, flags, tree)


def test_source_change_and_expired_borrow_are_rejected(tmp_path, monkeypatch):
    path, _, _ = snapshot(tmp_path)
    target = tmp_path/'protected.dat'
    target.write_bytes(b'keep')
    monkeypatch.setattr(products, '_BATCH_LEAVES', 2)
    original_write = writer._write_blocks
    calls = 0
    def mutate(stream, data):
        nonlocal calls
        original_write(stream, data)
        calls += 1
        if calls == 1:
            with path.open('ab') as changed:
                changed.write(b'changed')
    monkeypatch.setattr(writer, '_write_blocks', mutate)
    with pytest.raises(OSError, match='changed'):
        sm.crop_amrvac(path, target, root_bounds=BOX, overwrite=True)
    assert target.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.simesh-*'))
    monkeypatch.setattr(writer, '_write_blocks', original_write)
    path, _, _ = snapshot(tmp_path)
    from dataclasses import replace
    from simesh.fields import _Lease
    with sm.open_amrvac(path) as source:
        fields = sm.read_fields(source, region=sm.select_roots(source.mesh, BOX))
        metadata = source.metadata
    lease = _Lease()
    borrowed = replace(fields, _lease=lease)
    def expire(stream, data):
        original_write(stream, data)
        lease.active = False
    monkeypatch.setattr(writer, '_write_blocks', expire)
    with pytest.raises(RuntimeError, match='expired'):
        sm.write_amrvac(target, borrowed, metadata=metadata, root_bounds=BOX, overwrite=True)
    assert target.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.simesh-*'))


@pytest.mark.parametrize('name', ['bad\N{SNOWMAN}', 'a'*17, ' padded', 'padded ', 'rho\x00', '\x00'])
def test_nonroundtripping_field_names_are_rejected(tmp_path, name):
    path, _, _ = snapshot(tmp_path)
    with sm.open_amrvac(path) as source:
        fields = sm.select_fields(sm.read_fields(source, [0]), names=[name])
        metadata = source.metadata
    with pytest.raises(ValueError, match='ASCII'):
        sm.write_amrvac(tmp_path/'invalid.dat', fields, metadata=metadata, root_bounds=BOX)


def test_regional_prepared_interiors_omit_halo(tmp_path):
    path, _, _ = snapshot(tmp_path)
    with sm.open_amrvac(path) as source:
        selection = sm.select_roots(source.mesh, BOX)
        ready = sm.prepare(source, [0], region=selection, scheme='exact-phase')
        metadata = source.metadata
    assert ready.valid_halo == 2
    output = tmp_path/'interiors.dat'
    sm.write_amrvac(output, ready, metadata=metadata, root_bounds=BOX)
    with sm.open_amrvac(output) as source:
        actual = sm.read_fields(source)
        assert actual.storage_halo == actual.valid_halo == 0
        np.testing.assert_array_equal(actual.values, ready.interior())


def test_header_offset_limit_has_explicit_error(tmp_path):
    path, _, _ = snapshot(tmp_path)
    with sm.open_amrvac(path) as source:
        header = source.metadata.to_header()
    header['nleafs'] = 100_000_000
    with pytest.raises(ValueError, match='32-bit'):
        writer._header_bytes(header)
