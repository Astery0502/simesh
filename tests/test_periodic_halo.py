"""Periodic halo geometry against independently tiled interior AMR contacts."""

import numpy as np
import pytest

import simesh as sm
from simesh._amr import halo
from simesh._amr.morton import level1_morton
from simesh._amr.relations import RELATION_PHYSICAL, RELATION_SAME, balanced_refined_relations
from simesh.preparation import exact, coordinate
from simesh.bounded import PreparedPool
from simesh._kernels.primitives._relation_phases import validate_refined_relation_phases_unchecked
from fixtures import write_dat


PERIODIC = [tuple(bool(mask & (1 << a)) for a in range(3)) for mask in range(1, 8)]


def make_mesh(roots, periodic, mixed=False, block=(4, 6, 8)):
    _, coordinates = level1_morton(np.array(roots, dtype=np.int64))
    flags = []
    for coord in coordinates:
        if mixed and roots == (1, 1, 1):
            flags.append(False)
            for child in range(8):
                flags.extend([False] + [True]*8 if child in (0, 3, 6) else [True])
        elif mixed and sum(coord) % 3 == 0:
            flags.extend([False] + [True]*8)
        else:
            flags.append(True)
    return sm.mesh_from_forest(roots, np.array(flags), lower=(-3,)*3, upper=(3,)*3,
                               block_shape=block, periodic=periodic)


def forest_args(mesh):
    f = mesh.forest
    return (mesh.root_shape, mesh.coord_to_rank, f.root_node_ids, f.node_levels,
            f.node_coords, f.child_node_ids, f.node_leaf_ids, f.leaf_node_ids)


def source(mesh, raw):
    return sm.source_from_arrays(mesh, raw, tuple(f'b{i+1}' for i in range(raw.shape[1])))


def tiled_reference(mesh, raw):
    """Repeat root subtrees and discrete values, preserving the middle tile IDs.

    The oracle uses a nonperiodic Mesh and ordinary internal transfers. Tiling is
    confined to these small tests; production payload is never replicated.
    """
    periodic = np.array(mesh.periodic, dtype=np.int64)
    repetitions = 1 + 2*periodic
    roots = mesh.root_shape*repetitions
    _, coords = level1_morton(roots)
    root_nodes = mesh.forest.root_node_ids
    stops = np.r_[root_nodes[1:], len(mesh.node_leaves)]
    flags = []
    for coord in coords:
        rank = mesh.coord_to_rank[tuple(coord % mesh.root_shape)]
        flags.extend(mesh.node_leaves[root_nodes[rank]:stops[rank]] >= 0)
    length = mesh.upper-mesh.lower
    expanded = sm.mesh_from_forest(roots, np.array(flags),
        lower=mesh.lower-periodic*length, upper=mesh.upper+periodic*length,
        block_shape=mesh.block_shape)
    lookup = {(int(mesh.forest.node_levels[n]), *mesh.forest.node_coords[n]): leaf
              for leaf, n in enumerate(mesh.leaf_nodes)}
    originals = []
    central = np.full(mesh.leaf_count, -1, dtype=np.int64)
    for leaf, node in enumerate(expanded.leaf_nodes):
        level = int(expanded.forest.node_levels[node])
        extent = mesh.root_shape*(1 << (level-1))
        coord = expanded.forest.node_coords[node]
        original = lookup[(level, *(coord % extent))]
        originals.append(original)
        if np.array_equal(coord // extent, periodic):
            central[original] = leaf
    assert np.all(central >= 0)
    return source(expanded, raw[originals]), central


@pytest.mark.parametrize('periodic', PERIODIC)
@pytest.mark.parametrize('roots', [(1, 1, 1), (2, 3, 2)])
def test_same_level_analytic_halos_and_self_neighbors(periodic, roots):
    mesh = make_mesh(roots, periodic)
    block = np.array(mesh.block_shape)
    local = np.indices(block).astype(float)+.5

    def field(xyz):
        return np.sin(np.pi*xyz[0]/3) + 2*np.cos(np.pi*xyz[1]/3) + 3*np.sin(np.pi*xyz[2]/3)

    raw = np.array([field(mesh.bounds[i, 0, :, None, None, None] +
                          local*mesh.spacing[i, :, None, None, None])
                    for i in range(mesh.leaf_count)])[:, None]
    with source(mesh, raw) as current:
        ready = sm.prepare(current, scheme='exact-phase')
        planned = sm.plan_preparation(mesh, scheme='exact-phase').prepare(current)
    np.testing.assert_array_equal(ready.values, planned.values)
    np.testing.assert_array_equal(ready.interior()[..., 0], raw[:, 0])
    assert ready.valid_halo == ready.storage_halo == 2
    padded = np.indices(block+4).astype(float)-1.5
    for leaf in range(mesh.leaf_count):
        xyz = mesh.bounds[leaf, 0, :, None, None, None] + padded*mesh.spacing[leaf, :, None, None, None]
        for axis in range(3):
            if periodic[axis]:
                xyz[axis] = (xyz[axis]+3) % 6 - 3
            else:
                xyz[axis] = np.clip(xyz[axis], -3+mesh.spacing[leaf, axis]/2,
                                    3-mesh.spacing[leaf, axis]/2)
        np.testing.assert_allclose(ready.values[leaf, ..., 0], field(xyz), atol=2e-14, rtol=0)
    kinds, masks, counts, ids = balanced_refined_relations(*forest_args(mesh),
        np.arange(mesh.leaf_count, dtype=np.int64), halo.CANONICAL_DIRECTIONS, periodic=periodic)
    periodic_mask = sum(1 << a for a in range(3) if periodic[a])
    assert not np.any(masks & periodic_mask)
    if roots == (1, 1, 1):
        for row, direction in enumerate(halo.CANONICAL_DIRECTIONS):
            expected_mask = sum(1 << a for a in range(3) if direction[a] and not periodic[a])
            assert masks[0, row] == expected_mask
            has_neighbor = any(direction[a] and periodic[a] for a in range(3))
            assert kinds[0, row] == (RELATION_SAME if has_neighbor else RELATION_PHYSICAL)
            assert counts[0, row] == int(has_neighbor)
            if has_neighbor:
                assert ids[0, row, 0] == 0
        assert ready.preparation_stats['selected_load_count'] == 1


@pytest.mark.parametrize('periodic', PERIODIC)
@pytest.mark.parametrize('roots', [(1, 1, 1), (3, 2, 2)])
def test_amr_seams_match_internal_tiling_and_checked_support(periodic, roots):
    mesh = make_mesh(roots, periodic, mixed=True)
    raw = np.random.default_rng(742).normal(size=(mesh.leaf_count, 3, *mesh.block_shape))
    reference, central = tiled_reference(mesh, raw)
    with source(mesh, raw) as current, reference:
        expected = sm.prepare(reference, leaf_ids=central, scheme='exact-phase')
        ready = sm.prepare(current, scheme='exact-phase', support_capacity=57)
        np.testing.assert_array_equal(ready.values, expected.values)
        plan = sm.plan_preparation(mesh, scheme='exact-phase', support_capacity=57)
        np.testing.assert_array_equal(plan.prepare(current).values, ready.values)
        # Exercise checked coarse-workspace geometry, including repeated source slots.
        block, capacity, _ = exact.parameters(mesh, 3, 57)
        workspace = exact.Workspace.allocate(mesh, 3, block, capacity)
        for leaf in range(mesh.leaf_count):
            count, selected = exact.plan_chunk(workspace, np.array([leaf], dtype=np.int64))
            halo._preflight_chunk_actions(workspace.w, count, selected, workspace.lower,
                workspace.upper, workspace.modes, workspace.normals)
            w = workspace.w
            assert validate_refined_relation_phases_unchecked(
                w.selected_leaf_ids[:selected], mesh.forest.node_levels, mesh.forest.node_coords,
                mesh.leaf_nodes, halo.CANONICAL_DIRECTIONS, w.relation_kinds[:count],
                w.physical_masks[:count], w.source_counts[:count], w.source_slots[:count],
                mesh.root_shape, sum(1 << a for a in range(3) if periodic[a]))[0] == 0


@pytest.mark.parametrize('periodic', [True, (0, 0, 1), (False, True), ('no', 'no', 'yes')])
def test_invalid_periodic_configuration(periodic):
    with pytest.raises(ValueError, match='three boolean'):
        make_mesh((1, 1, 1), periodic)


def test_periodic_configuration_is_detached_and_coordinate_backend_rejects():
    flags = np.array([False, False, True])
    mesh = make_mesh((1, 1, 1), flags)
    flags[:] = False
    assert mesh.periodic == (False, False, True)
    with source(mesh, np.ones((1, 1, *mesh.block_shape))) as current:
        for backend in ('threadpool', 'openmp'):
            with pytest.raises(ValueError, match='periodic.*exact-phase'):
                sm.prepare(current, scheme='coordinate-phase', backend=backend)
        with pytest.raises(ValueError, match='periodic.*exact-phase'):
            coordinate.build_geometry(mesh)


@pytest.mark.parametrize('axes', [tuple(a for a in range(3) if flags[a]) for flags in PERIODIC])
def test_balance_rejects_gap_only_across_periodic_face_edge_corner(axes):
    roots = np.array((3, 3, 3), dtype=np.int64)
    periodic = tuple(a in axes for a in range(3))
    opposite = tuple(2 if a in axes else 0 for a in range(3))
    _, coordinates = level1_morton(roots)
    flags = []
    for coord in coordinates:
        if tuple(coord) == opposite:
            flags.append(True)
        else:
            flags.append(False)
            for child in range(8):
                flags.extend([False] + [True]*8 if tuple(coord) == (0, 0, 0) and child == 0 else [True])
    options = dict(lower=(-3,)*3, upper=(3,)*3, block_shape=(4,)*3)
    nonperiodic = sm.mesh_from_forest(roots, np.array(flags), **options)
    assert nonperiodic.leaf_count > 0  # The interior forest itself is balanced.
    with pytest.raises(ValueError, match='all-touch two-to-one balance violation'):
        sm.mesh_from_forest(roots, np.array(flags), periodic=periodic, **options)


def test_regions_batches_pool_plan_reuse_and_lifetimes():
    mesh = make_mesh((3, 2, 2), (False, False, True), mixed=True)
    raw = np.random.default_rng(386).normal(size=(mesh.leaf_count, 3, *mesh.block_shape))
    ids = np.flatnonzero(mesh.bounds[:, 0, 2] == mesh.lower[2])[::-1].copy()
    plan = sm.plan_preparation(mesh, leaf_ids=ids, scheme='exact-phase', support_capacity=57)
    with source(mesh, raw) as first:
        full = sm.prepare(first, scheme='exact-phase')
        ready = sm.prepare(first, leaf_ids=ids, fields=('b3', 'b1'), scheme='exact-phase')
        np.testing.assert_array_equal(ready.values, full.values[ids][..., [2, 0]])
        # Only the low side is published; opposite-side support remains private.
        np.testing.assert_array_equal(ready.leaf_ids, ids)
        assert np.all(mesh.bounds[ready.leaf_ids, 0, 2] == -3)
        with sm.select_source(first, ('b3', 'b1')) as subset:
            np.testing.assert_array_equal(plan.prepare(subset).values, ready.values)
        batches = sm.iter_prepared(first, fields=('b3', 'b1'), leaf_ids=ids,
                                  scheme='exact-phase', batch_size=2, support_capacity=57)
        copied = []
        for batch in batches:
            copied.append(batch.values.copy())
        np.testing.assert_array_equal(np.concatenate(copied), ready.values)
        with pytest.raises(RuntimeError, match='expired'):
            _ = batch.values
        with PreparedPool(first, fields=('b3', 'b1'), capacity=2,
                             scheme='exact-phase', support_capacity=57) as pool:
            for leaf in ids:
                with pool.borrow([leaf]) as borrowed:
                    np.testing.assert_array_equal(borrowed.window(leaf, (0, 0, 0), mesh.block_shape, support=2),
                        full.values[leaf][..., [2, 0]])
            with pytest.raises(RuntimeError, match='expired'):
                _ = borrowed.values
        with pytest.raises(MemoryError):
            sm.prepare(first, scheme='exact-phase', memory_limit=1)
        with pytest.raises(MemoryError):
            plan.prepare(first, memory_limit=1)
    np.testing.assert_array_equal(ready.interior(), full.interior()[ids][..., [2, 0]])
    assert sm.curl(full).valid_halo == 1
    # A reusable plan contains geometry, never field values or limiter slopes.
    different = np.sin(raw*2) + raw**2
    with source(mesh, different) as second:
        changed = plan.prepare(second, fields=('b1', 'b3'))
        direct = sm.prepare(second, leaf_ids=ids, fields=('b1', 'b3'), scheme='exact-phase')
        np.testing.assert_array_equal(changed.values, direct.values)
        assert not np.array_equal(changed.values, ready.values)
    other = make_mesh((3, 2, 2), (False, False, False), mixed=True)
    with source(other, raw) as wrong:
        with pytest.raises(ValueError, match='different immutable Mesh'):
            plan.prepare(wrong)


def test_interior_region_and_consumers_keep_finite_domain_semantics():
    roots = (5, 5, 5)
    mesh = make_mesh(roots, (False, False, True), block=(4,)*3)
    plain = make_mesh(roots, (False, False, False), block=(4,)*3)
    local = np.indices(mesh.block_shape)+.5
    raw = np.empty((mesh.leaf_count, 3, *mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        x, y, z = mesh.bounds[leaf, 0, :, None, None, None] + local*mesh.spacing[leaf, :, None, None, None]
        raw[leaf] = [y+np.sin(np.pi*z/3), 2*x+np.cos(np.pi*z/3), x-y]
    region = ((-.5, -.5, -.5), (.5, .5, .5))
    selected = sm.select_region(mesh, region)
    ids = selected.leaf_ids
    assert np.all(mesh.bounds[ids, 0]-4*mesh.spacing[ids] > mesh.lower)
    assert np.all(mesh.bounds[ids, 1]+4*mesh.spacing[ids] < mesh.upper)
    with source(mesh, raw) as periodic, source(plain, raw) as original:
        ready = sm.prepare(periodic, region=region, scheme='exact-phase')
        expected = sm.prepare(original, region=region, scheme='exact-phase')
        np.testing.assert_array_equal(ready.values, expected.values)
        np.testing.assert_array_equal(sm.curl(ready).values, sm.curl(expected).values)
        np.testing.assert_array_equal(sm.divergence(ready).values, sm.divergence(expected).values)
        outside = [[0., 0., 3.1], [0., 0., -3.1]]
        np.testing.assert_array_equal(mesh.locate(outside), [-1, -1])
        assert not sm.sample(ready, outside)[2].any()
        with pytest.raises(ValueError, match='do not intersect'):
            sm.select_region(mesh, ((0, 0, 3.1), (1, 1, 3.5)))
        interior = sm.read_fields(periodic)
        before = sm.volume_integral(interior, 'b1')
        after = sm.volume_integral(sm.read_fields(original), 'b1')
        assert before.value == after.value


@pytest.mark.parametrize('endian', ['<', '>'])
def test_binary_metadata_opposite_leaf_reads_and_export_rejections(tmp_path, endian):
    mesh = make_mesh((1, 1, 3), (False, False, True), block=(4,)*3)
    raw = np.arange(mesh.leaf_count*3*64, dtype=float).reshape(mesh.leaf_count, 3, 4, 4, 4)
    path = tmp_path/'periodic.dat'
    write_dat(path, mesh, raw, byte_order=endian, saved_ghosts=True)
    low = int(np.argmin(mesh.bounds[:, 0, 2]))
    high = int(np.argmax(mesh.bounds[:, 1, 2]))
    with sm.open_amrvac(path, fields=('b3', 'b1')) as current:
        assert current.mesh.periodic == mesh.periodic
        assert current.metadata.header['periodic'] == mesh.periodic
        ready = sm.prepare(current, leaf_ids=[low], scheme='exact-phase')
        assert ready.mesh is current.mesh
        np.testing.assert_array_equal(ready.values[0, 2:-2, 2:-2, :2],
                                      np.moveaxis(raw[high, [2, 0], :, :, -2:], 0, -1))
        all_fields = sm.read_fields(current)
        metadata = current.metadata.to_header()
        metadata['periodic'] = [False, False, False]
        with pytest.raises(ValueError, match='periodic meshes'):
            sm.write_amrvac(tmp_path/'wrong.dat', all_fields, metadata=metadata)
        for interpolation in ('zero', 'native', 'linear'):
            options = {'scheme': 'exact-phase'} if interpolation == 'linear' else {}
            with pytest.raises(ValueError, match='periodic meshes'):
                sm.export_uniform(current, (4, 4, 12), interpolation=interpolation, **options)
        sliced = sm.slice_axis(all_fields, 'z', 0.)
        with pytest.raises(sm.ResultFileError, match='periodic meshes'):
            sm.save_result(tmp_path/'slice.npz', sliced)
        # Numeric results without a Mesh can still be saved.
        sm.save_result(tmp_path/'integral.npz', sm.volume_integral(all_fields, 'b1'))
    with pytest.raises(ValueError, match='periodic meshes'):
        sm.export_uniform(path, (4, 4, 12))
    with pytest.raises(ValueError, match='periodic meshes'):
        sm.crop_amrvac(path, tmp_path/'crop.dat', root_bounds=((0, 0, 0), (1, 1, 1)))
    assert not (tmp_path/'wrong.dat').exists()
    assert not (tmp_path/'slice.npz').exists()
    assert not (tmp_path/'crop.dat').exists()
    np.testing.assert_array_equal(ready.interior()[0], np.moveaxis(raw[low, [2, 0]], 0, -1))


@pytest.mark.parametrize('endian', ['<', '>'])
def test_periodic_refined_binary_plan_matches_array_source(tmp_path, endian):
    mesh = make_mesh((3, 2, 2), (False, False, True), mixed=True)
    raw = np.random.default_rng(94).normal(size=(mesh.leaf_count, 3, *mesh.block_shape))
    path = tmp_path/'refined.dat'
    write_dat(path, mesh, raw, byte_order=endian, saved_ghosts=True)
    ids = np.flatnonzero((mesh.bounds[:, 0, 2] == -3) | (mesh.bounds[:, 1, 2] == 3))[::-1].copy()
    with source(mesh, raw) as arrays, sm.open_amrvac(path) as file_source:
        expected = sm.prepare(arrays, leaf_ids=ids, fields=('b2', 'b3'), scheme='exact-phase')
        planned = sm.plan_preparation(file_source.mesh, leaf_ids=ids, scheme='exact-phase')
        actual = planned.prepare(file_source, fields=('b2', 'b3'))
        np.testing.assert_array_equal(actual.values, expected.values)
    assert np.isfinite(actual.values).all()


def test_curl_consumes_periodic_ghosts_at_both_domain_faces():
    mesh = make_mesh((1, 1, 3), (False, False, True))
    k = np.pi/3
    raw = np.empty((mesh.leaf_count, 3, *mesh.block_shape))
    expected = np.empty((mesh.leaf_count, *mesh.block_shape, 3))
    local = np.indices(mesh.block_shape)+.5
    for leaf in range(mesh.leaf_count):
        z = mesh.bounds[leaf, 0, 2] + local[2]*mesh.spacing[leaf, 2]
        raw[leaf] = [np.cos(k*z), np.sin(k*z), np.ones_like(z)]
        factor = np.sin(k*mesh.spacing[leaf, 2])/mesh.spacing[leaf, 2]
        expected[leaf] = np.moveaxis(np.array([-factor*np.cos(k*z), -factor*np.sin(k*z), np.zeros_like(z)]), 0, -1)
    with source(mesh, raw) as current:
        ready = sm.prepare(current, scheme='exact-phase')
    curled = sm.curl(ready)
    assert curled.valid_halo == 1
    np.testing.assert_allclose(curled.interior(), expected, atol=2e-14, rtol=0)
