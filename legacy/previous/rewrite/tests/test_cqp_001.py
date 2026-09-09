"""Cache index equivalence with the retained serial scan and failure boundary."""

import numpy as np
import pytest

import simesh_rewrite.completed_halo_sampling as c
from test_chs_001 import (Artifact, ReaderState, counting_reader, make_session,
                          points_for_leaves)
from test_rhe_001 import make_artifact, axis_coded_backing


def artifact():
    a = make_artifact((4, 4, 4), {(1, 1, 1)})
    block = np.full(3, 4, dtype=np.int64)
    return Artifact(a.root_shape, a.coord_to_rank, a.root_node_ids, a.node_levels,
                    a.node_coords, a.child_node_ids, a.node_leaf_ids, a.leaf_node_ids,
                    2, np.zeros(3), np.full(3, 4.0), a.root_shape * block, block,
                    axis_coded_backing(71, 4))


@pytest.mark.parametrize("capacity", [0, 1, 16, 36, 71])
def test_index_and_scan_match_across_cold_warm_eviction_histories(monkeypatch, capacity):
    a = artifact()
    readers = [ReaderState(a.backing), ReaderState(a.backing)]
    sessions = [make_session(a, counting_reader(r), capacity) for r in readers]
    original = c._cache_access_plan
    rng = np.random.default_rng(719)
    histories = [np.arange(36), np.arange(36), np.arange(15, 60),
                 rng.integers(0, 71, 80), np.arange(71)]
    for ids in histories:
        points = points_for_leaves(a, ids.tolist())
        results, stats = [], []
        for variant, session in enumerate(sessions):
            monkeypatch.setattr(c, "_cache_access_plan",
                                lambda state, owners, _variant=variant:
                                original(state, owners, _indexed=bool(_variant)))
            values = np.full((len(ids), 3), -1.0)
            owners = np.full(len(ids), -1, dtype=np.int64)
            stats.append(c.sample_refined_trilinear_vectors_cached(
                session, points, ids.copy(), values, owners))
            results.append((values, owners))
        assert stats[0] == stats[1]
        for x, y in zip(results[0], results[1]):
            np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
        for name in ("cache_leaf_ids", "cache_recency"):
            np.testing.assert_array_equal(getattr(sessions[0]._state, name),
                                          getattr(sessions[1]._state, name))
        assert sessions[0]._state.clock == sessions[1]._state.clock
    for name in ("calls", "field_calls"):
        left, right = getattr(readers[0], name), getattr(readers[1], name)
        assert len(left) == len(right)
        for x, y in zip(left, right):
            np.testing.assert_array_equal(x, y)


def test_indexed_plan_failure_preserves_cache_and_outputs_before_io(monkeypatch):
    a = artifact()
    reader = ReaderState(a.backing)
    session = make_session(a, counting_reader(reader), 36)
    points = points_for_leaves(a, list(range(36)))
    values = np.full((36, 3), -17.0)
    owners = np.full(36, -8, dtype=np.int64)
    original = c._prepare_refined_halo_chunk
    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise ValueError("injected planning failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(c, "_prepare_refined_halo_chunk", fail)
    with pytest.raises(ValueError, match="planning failure"):
        c.sample_refined_trilinear_vectors_cached(session, points,
            np.arange(36, dtype=np.int64), values, owners)
    assert reader.nonempty_calls == 0
    assert np.all(values == -17.0) and np.all(owners == -8)
    assert np.all(session._state.cache_leaf_ids == -1)
    assert session._state.clock == 0


def test_plan_arrays_and_tied_victims_match_scan():
    a = artifact()
    session = make_session(a, counting_reader(ReaderState(a.backing)), 36)
    state = session._state
    state.cache_leaf_ids[:] = np.arange(36)
    state.cache_recency[:] = 0
    keys = state.cache_leaf_ids.copy()
    for owners in (np.arange(36, dtype=np.int64),
                   np.arange(30, 71, dtype=np.int64),
                   np.array([0, 40, 1, 41, 0, 2, 42, 3, 43], dtype=np.int64)):
        scan = c._cache_access_plan(state, owners, _indexed=False)
        indexed = c._cache_access_plan(state, owners)
        for x, y in zip(scan, indexed):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(state.cache_leaf_ids, keys)
        assert np.all(state.cache_recency == 0)
