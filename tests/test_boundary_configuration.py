"""Physical parity through public sources and every preparation lifetime."""

import numpy as np
import pytest

import simesh as sm
from simesh.bounded import PreparedPool
from simesh._amr.morton import level1_morton
from fixtures import write_dat
from test_periodic_halo import make_mesh


def padded_reference(values, rows):
    """Independent cell-centered extension, including products of corner parity."""
    result = []
    for component, modes in zip(values, rows):
        index, signs = [], []
        for axis, size in enumerate(component.shape):
            positions = np.arange(-2, size+2)
            source = positions.copy()
            sign = np.ones(len(positions))
            for side, mask in ((0, positions < 0), (1, positions >= size)):
                mode = modes[2*axis+side]
                if mode == 'periodic':
                    source[mask] %= size
                elif mode == 'continuous':
                    source[mask] = 0 if side == 0 else size-1
                else:
                    source[mask] = -positions[mask]-1 if side == 0 else 2*size-positions[mask]-1
                    if mode == 'asymmetric':
                        sign[mask] = -1
            index.append(source)
            signs.append(sign)
        result.append(component[np.ix_(*index)] * signs[0][:, None, None] *
                      signs[1][None, :, None] * signs[2][None, None, :])
    return np.stack(result, axis=-1)


@pytest.mark.parametrize('scheme,periodic', [
    ('exact-phase', (False, False, False)),
    ('coordinate-phase', (False, False, False)),
    ('exact-phase', (False, False, True)),
])
def test_face_edge_corner_rules_and_component_order(scheme, periodic):
    mesh = make_mesh((1, 1, 1), periodic)
    raw = np.random.default_rng(84).normal(size=(1, 3, *mesh.block_shape))
    rows = [('symmetric', 'asymmetric', 'continuous', 'symmetric',
             'periodic' if periodic[2] else 'asymmetric',
             'periodic' if periodic[2] else 'continuous'),
            ('asymmetric', 'symmetric', 'asymmetric', 'continuous',
             'periodic' if periodic[2] else 'symmetric',
             'periodic' if periodic[2] else 'asymmetric')]
    config = {'b1': list(rows[0]), 'b2': list(rows[1])}
    with sm.source_from_arrays(mesh, raw, ['b1', 'b2', 'b3'], boundary=config) as source:
        config['b1'][0] = 'continuous'
        assert source.boundary[0] == rows[0]
        with pytest.raises(AttributeError):
            source.boundary = ()
        result = sm.prepare(source, ['b2', 'b1'], scheme=scheme)
        np.testing.assert_array_equal(result.values[0], padded_reference(raw[0, [1, 0]], rows[::-1]))
        np.testing.assert_array_equal(result.values[0, 2:-2, 2:-2, 2:-2],
                                      np.moveaxis(raw[0, [1, 0]], 0, -1))
        default = sm.prepare(source, ['b3'], scheme=scheme)
        np.testing.assert_array_equal(default.values[0], padded_reference(raw[0, [2]], [source.boundary[2]]))


def mirrored_reference(mesh, raw, rows):
    """Unfold physical faces into ordinary internal AMR neighbors, with parity."""
    lookup = {(int(mesh.forest.node_levels[n]), *mesh.forest.node_coords[n]): leaf
              for leaf, n in enumerate(mesh.leaf_nodes)}
    roots = mesh.root_shape*3
    _, coordinates = level1_morton(roots)
    flags, values, central = [], [], []

    def visit(level, coord):
        extent = mesh.root_shape * (1 << (level-1))
        tile, local = coord // extent, coord % extent
        reflected = (tile != 1) & ~np.array(mesh.periodic)
        original = np.where(reflected, extent-1-local, local)
        leaf = lookup.get((level, *original))
        flags.append(leaf is not None)
        if leaf is None:
            for child in range(8):
                visit(level+1, coord*2+np.array([(child >> axis) & 1 for axis in range(3)]))
            return
        if np.all(tile == 1):
            central.append((leaf, len(values)))
        value = raw[leaf].copy()
        for axis in range(3):
            if reflected[axis]:
                value = np.flip(value, axis=axis+1)
                side = 0 if tile[axis] == 0 else 1
                sign = np.array([-1 if row[2*axis+side] == 'asymmetric' else 1 for row in rows])
                value = value*sign[:, None, None, None]
        values.append(value)

    for coord in coordinates:
        visit(1, coord)
    length = mesh.upper-mesh.lower
    expanded = sm.mesh_from_forest(roots, np.array(flags), lower=mesh.lower-length,
                                  upper=mesh.upper+length, block_shape=mesh.block_shape)
    selected = np.array([mapped for _, mapped in sorted(central)])
    return expanded, np.ascontiguousarray(values), selected


@pytest.mark.parametrize('scheme,periodic', [
    ('exact-phase', (False, False, False)),
    ('coordinate-phase', (False, False, False)),
    ('exact-phase', (False, False, True)),
])
def test_reflected_amr_matches_unfolded_internal_interfaces(scheme, periodic):
    mesh = make_mesh((2, 1, 1), periodic, mixed=True, block=(4, 4, 4))
    raw = np.random.default_rng(39).normal(size=(mesh.leaf_count, 2, *mesh.block_shape))
    with sm.source_from_arrays(mesh, raw, ['even', 'odd'],
                               boundary={'even': 'symmetric', 'odd': 'asymmetric'}) as source:
        expanded, values, selected = mirrored_reference(mesh, raw, source.boundary)
        with sm.source_from_arrays(expanded, values, ['even', 'odd']) as reference:
            expected = sm.prepare(reference, scheme=scheme).values[selected]
        actual = sm.prepare(source, scheme=scheme)
        np.testing.assert_allclose(actual.values, expected, rtol=0, atol=2e-15)
        if scheme == 'coordinate-phase':
            return
        plan = sm.plan_preparation(mesh, leaf_ids=list(range(mesh.leaf_count))[::-1], scheme='exact-phase')
        planned = sm.prepare(source, ['odd', 'even'], scheme='exact-phase', plan=plan)
        np.testing.assert_array_equal(planned.values, actual.values[::-1, ..., ::-1])
        with sm.select_source(source, ['odd', 'even']) as subset:
            with sm.cache_source(subset, capacity=mesh.leaf_count) as cached:
                assert cached.boundary == source.boundary[::-1]
                batches = list(batch.values.copy() for batch in sm.iter_prepared(
                    cached, scheme='exact-phase', batch_size=2))
                np.testing.assert_array_equal(np.concatenate(batches), actual.values[..., ::-1])
                pool = PreparedPool(cached, capacity=mesh.leaf_count, scheme='exact-phase')
                try:
                    with pool.borrow(np.arange(mesh.leaf_count)) as borrowed:
                        np.testing.assert_array_equal(borrowed.values, actual.values[..., ::-1])
                finally:
                    pool.close()
        with sm.source_from_arrays(mesh, raw, ['even', 'odd']) as continuous:
            reused = plan.prepare(continuous)
            direct = sm.prepare(continuous, leaf_ids=plan.selection.leaf_ids, scheme='exact-phase')
            np.testing.assert_array_equal(reused.values, direct.values)
            assert not np.array_equal(reused.values, planned.values[..., ::-1])


@pytest.mark.parametrize('boundary', ['invalid', {'missing': 'symmetric'}, {'b1': ['symmetric']},
                                      {'b1': ['periodic']*6}, {'b1': ['symmetric']*5+[12]},
                                      {'b1': 'noinflow'}, [['symmetric']*6]*2])
def test_invalid_boundary_rejected(boundary):
    mesh = make_mesh((1, 1, 1), (False, False, False))
    with pytest.raises((ValueError, KeyError)):
        sm.source_from_arrays(mesh, np.ones((1, 1, *mesh.block_shape)), ['b1'], boundary=boundary)


def test_periodic_faces_cannot_be_overridden():
    mesh = make_mesh((1, 1, 1), (False, False, True))
    with pytest.raises(ValueError, match='Mesh.periodic'):
        sm.source_from_arrays(mesh, np.ones((1, 1, *mesh.block_shape)), ['b1'],
                              boundary={'b1': ['symmetric']*6})


def test_binary_source_explicit_rules_and_selection(tmp_path):
    mesh = make_mesh((1, 1, 1), (False, False, True), block=(4, 4, 4))
    raw = np.random.default_rng(61).normal(size=(1, 2, *mesh.block_shape))
    path = tmp_path/'boundary.dat'
    write_dat(path, mesh, raw)
    with sm.open_amrvac(path, fields=['b2'], boundary={'b2': 'asymmetric'}) as source:
        result = sm.prepare(source, scheme='exact-phase')
        expected = padded_reference(raw[0, [1]], source.boundary)
    np.testing.assert_array_equal(result.values[0], expected)


def test_boundary_arrays_are_admitted_and_retained_by_adapters():
    mesh = make_mesh((1, 1, 1), (False, False, False))
    raw = np.ones((1, 2, *mesh.block_shape))
    old_budget = mesh.nbytes+raw.nbytes
    with pytest.raises(MemoryError):
        sm.source_from_arrays(mesh, raw, ['even', 'odd'], copy=False,
                              boundary='symmetric', memory_limit=old_budget)
    with sm.source_from_arrays(mesh, raw, ['even', 'odd'], copy=False,
                               boundary='symmetric', memory_limit=old_budget+12) as source:
        assert mesh.nbytes+source.nbytes == old_budget+12
        with sm.select_source(source, ['odd']) as subset:
            assert subset.nbytes == source.nbytes+8+6
