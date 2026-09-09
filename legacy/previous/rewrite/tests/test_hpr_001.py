"""Owned preflight conformance against the complete standalone CSP boundary."""

import numpy as np
import pytest

import simesh_rewrite.refined_halo as h
from test_rhe_001 import (
    make_artifact, axis_coded_backing, execute_args, i3,
    ReaderState, WriterState, adapters, assert_bits_equal,
)


def request(lower, upper, capacity, refined=(1, 1, 1)):
    artifact = make_artifact((4, 4, 4), {refined})
    count = len(artifact.leaf_node_ids)
    backing = axis_coded_backing(count, 2)
    shape = tuple(4 + a + b for a, b in zip(lower, upper))
    output = np.full((count, 2, *shape), -917.0)
    rs, ws = ReaderState(backing), WriterState(output)
    reader, writer = adapters(rs, ws)
    args = execute_args(artifact, reader, writer, np.arange(count, dtype=np.int64),
                        i3(1, 0), i3(*lower), i3(*upper),
                        np.zeros((2, 6), dtype=np.uint8), i3(-1, -1, -1), capacity)
    return args, output, rs, ws


@pytest.mark.parametrize("lower,upper", [((2, 2, 2), (2, 2, 2)),
                                         ((0, 1, 2), (2, 0, 1))])
@pytest.mark.parametrize("capacity", [57, 71])
@pytest.mark.parametrize("refined", [(1, 1, 1), (0, 0, 0)])
def test_every_owned_plan_and_composed_output_matches_checked(
    monkeypatch, lower, upper, capacity, refined
):
    args, output, reader, writer = request(lower, upper, capacity, refined)
    original = h._fill_owned_coarser_plan
    checked_plans = 0
    physical_plans = 0
    phases = set()
    selected_count = 0
    preflight = h._preflight_chunk_actions

    def capture(workspace, primary_count, selected, *rest):
        nonlocal selected_count
        selected_count = selected
        return preflight(workspace, primary_count, selected, *rest)

    monkeypatch.setattr(h, "_preflight_chunk_actions", capture)

    def compare(workspace, primary, lo, hi):
        nonlocal checked_plans, physical_plans
        actual_count = original(workspace, primary, lo, hi)
        actual = [value.copy() for value in workspace.csp_outputs]
        expected_count = h.fill_coarser_slope_support_plan(
            lo, hi, selected_count, primary,
            int(workspace.action_phases[0]), workspace.action_directions[0],
            h.CANONICAL_DIRECTIONS, workspace.relation_kinds[primary],
            workspace.physical_masks[primary], workspace.source_counts[primary],
            workspace.source_slots[primary], workspace.action_target_lower[0],
            workspace.action_target_upper[0],
            *(v[0] for v in workspace.cwp_outputs), *workspace.csp_outputs,
        )
        assert actual_count == expected_count
        for a, b in zip(actual, workspace.csp_outputs):
            np.testing.assert_array_equal(a, b)
        checked_plans += 1
        physical_plans += actual_count[1] > actual_count[0]
        phases.add(int(workspace.action_phases[0]))
        return actual_count

    monkeypatch.setattr(h, "_fill_owned_coarser_plan", compare)
    stats = h.execute_selected_refined_halos_from_blocks(*args)
    actual = output.copy()
    assert checked_plans > 8
    if refined == (1, 1, 1):
        assert phases == set(range(8))
    else:
        assert physical_plans > 0
    args2, expected, reader2, writer2 = request(lower, upper, capacity, refined)
    monkeypatch.setattr(h, "_preflight_chunk_actions",
                        h._preflight_chunk_actions_checked_reference)
    reference = h.execute_selected_refined_halos_from_blocks(*args2)
    assert stats == reference
    assert_bits_equal(actual, expected)
    assert reader.nonempty_calls == reader2.nonempty_calls
    assert writer.nonempty_calls == writer2.nonempty_calls
    for left, right in [(reader.calls, reader2.calls), (writer.calls, writer2.calls)]:
        assert len(left) == len(right)
        for a, b in zip(left, right):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("defect", ["inactive", "mask"])
def test_owned_preflight_rejects_corrupt_artifacts_before_io(monkeypatch, defect):
    args, output, reader, writer = request((2, 2, 2), (2, 2, 2), 71)
    original = h._preflight_chunk_actions

    def corrupt(workspace, primary_count, selected_count, *rest):
        rows = np.argwhere((workspace.relation_kinds[:primary_count] == 2)
                           & (workspace.physical_masks[:primary_count] == 0))
        primary, direction = map(int, rows[0])
        if defect == "inactive":
            workspace.source_slots[primary, direction, 1] = 0
        elif defect == "mask":
            workspace.physical_masks[primary, (direction + 1) % 26] = 128
        return original(workspace, primary_count, selected_count, *rest)

    monkeypatch.setattr(h, "_preflight_chunk_actions", corrupt)
    with pytest.raises((ValueError, RuntimeError)):
        h.execute_selected_refined_halos_from_blocks(*args)
    assert reader.nonempty_calls == writer.nonempty_calls == 0
    assert np.all(output == -917.0)


def test_owned_preflight_keeps_geometry_rejection_before_io(monkeypatch):
    args, output, reader, writer = request((2, 2, 2), (2, 2, 2), 71)
    original = h.fill_coarser_workspace_boxes

    def corrupt(*args):
        original(*args)
        args[-2][0, 0] = 1000  # Required upper exceeds the half-block lattice.

    monkeypatch.setattr(h, "fill_coarser_workspace_boxes", corrupt)
    with pytest.raises(ValueError, match="half-block lattice"):
        h.execute_selected_refined_halos_from_blocks(*args)
    assert reader.nonempty_calls == writer.nonempty_calls == 0
    assert np.all(output == -917.0)


def test_relation_proof_is_lazy_and_geometry_is_per_action(monkeypatch):
    args, *_ = request((2, 2, 2), (2, 2, 2), 71)
    counts = {"relation": 0, "geometry": 0}
    for name, key in [("_validate_relation_row", "relation"),
                      ("_validate_plan_geometry", "geometry")]:
        original = getattr(h, name)

        def count(*args, _original=original, _key=key):
            counts[_key] += 1
            return _original(*args)

        monkeypatch.setattr(h, name, count)
    h.execute_selected_refined_halos_from_blocks(*args)
    assert counts["relation"] == 8
    assert counts["geometry"] > counts["relation"]
