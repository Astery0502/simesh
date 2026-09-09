"""Two-layer exchange admission is independent of derivative support."""

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from simesh._amr import halos
from simesh.preparation import coordinate, exact
from fixtures import mixed_source


@pytest.mark.parametrize('exchange', ['physical', 'same-level'])
@pytest.mark.parametrize('lower_width,upper_width', [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
def test_exchange_widths(exchange, lower_width, upper_width):
    lower = np.full(3, lower_width, dtype=np.int64)
    upper = lower+4
    payload = np.full((1, 1, *(upper+upper_width)), 31.)
    ids = np.array([0], dtype=np.int64)
    neighbors = np.full((1, 6), -1, dtype=np.int64)
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = np.full(3, -1, dtype=np.int64)
    arguments = (payload, lower, upper, ids)
    if exchange == 'physical':
        fill = halos.fill_physical_halos
    else:
        fill = halos.fill_same_level_halos
        arguments += (1,)
    arguments += (neighbors, modes, normals)
    if 1 in (lower_width, upper_width):
        with pytest.raises(ValueError, match='ghost exchange requires at least two layers'):
            fill(*arguments)
        assert np.all(payload == 31.)
    else:
        fill(*arguments)
        assert np.all(payload == 31.)


def test_refined_exchange_rejects_one_layer_before_binding():
    with mixed_source()[0] as source:
        block, capacity, _ = exact.parameters(source.mesh, 1, 128)
        workspace = exact.Workspace.allocate(source.mesh, 1, block, capacity)
        workspace.lower[:] = 1
        workspace.upper[:] = block+1
        workspace.w.payload.fill(31.)
        workspace.w.selected_leaf_ids.fill(-1)
        with pytest.raises(ValueError, match='ghost exchange requires at least two layers'):
            exact.plan_chunk(workspace, np.array([0], dtype=np.int64))
        assert np.all(workspace.w.payload == 31.)
        assert np.all(workspace.w.selected_leaf_ids == -1)


def test_coordinate_exchange_rejects_one_layer():
    with mixed_source()[0] as source:
        geometry = coordinate.build_geometry(source.mesh)
        workspace = coordinate.allocate_workspace(geometry, 1)
        values = np.full((source.mesh.leaf_count, 10, 10, 10, 1), 31.)
        with pytest.raises(ValueError, match='padded blocks'):
            coordinate.execute(geometry, workspace, values, workers=1, backend='threadpool')
        assert np.all(values == 31.)


def test_one_layer_storage_reads_and_derivative_consumption_remain_valid():
    with mixed_source()[0] as source:
        output = np.full((1, 10, 10, 10, 1), 31.)
        source.read_native_into([0], [0], output, storage_halo=1)
        np.testing.assert_array_equal(output[:, 1:-1, 1:-1, 1:-1],
                                      sm.read_fields(source, [0], leaf_ids=[0]).values)
        assert np.all(output[:, 0] == 31.)
        ready = sm.prepare(source, scheme='exact-phase')
    derived = sm.curl(ready)
    assert derived.valid_halo == derived.storage_halo == 1
    values, _, valid = sm.sample(derived, [[1.5, .5, .5], [.25, .25, .25]])
    assert valid.all()
    np.testing.assert_allclose(values, [[3., -3., 3.]]*2, atol=2e-13, rtol=0)
    grid = app.uniform_grid(derived, (4, 3, 2))
    assert grid.valid.all()
    assert sm.curl(derived).valid_halo == 0
