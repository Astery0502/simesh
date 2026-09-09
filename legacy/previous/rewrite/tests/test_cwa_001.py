from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.coarser_support import (
    CANONICAL_DIRECTIONS,
    NO_SOURCE,
    PLAN_CAPACITY,
    fill_coarser_slope_support_plan,
)
from simesh_rewrite.coarser_workspace import fill_coarser_workspace_boxes
from simesh_rewrite.coarser_workspace_application import (
    _apply_coarser_workspace_plan_unchecked,
    _validate_coarser_workspace_plan,
    apply_coarser_workspace_plan,
)
from simesh_rewrite.coarser_workspace_application_reference import (
    apply_coarser_workspace_plan_reference,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.prolongation import prolong_cartesian_2to1_into
from simesh_rewrite.prolongation_reference import prolong_cartesian_2to1_reference
from simesh_rewrite.relation_phases import fill_refined_relation_phase_codes
from simesh_rewrite.relation_slots import resolve_refined_relation_source_slots
from simesh_rewrite.relations import RELATION_COARSER, balanced_refined_relations


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def direction_index(direction: tuple[int, int, int]) -> int:
    column = (
        (direction[2] + 1) * 9
        + (direction[1] + 1) * 3
        + direction[0]
        + 1
    )
    return column if column < 13 else column - 1


def make_flags(
    root_shape: tuple[int, int, int],
    refined_roots: list[tuple[int, int, int]],
) -> np.ndarray:
    root = i3(*root_shape)
    _, root_coords = level1_morton(root)
    flags: list[bool] = []
    for root_coord_array in root_coords:
        root_coord = tuple(int(value) for value in root_coord_array)
        if root_coord in refined_roots:
            flags.append(False)
            flags.extend([True] * 8)
        else:
            flags.append(True)
    return np.asarray(flags, dtype=np.bool_)


def relation_case(
    root_shape: tuple[int, int, int],
    refined_roots: list[tuple[int, int, int]],
):
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        make_flags(root_shape, refined_roots),
    )
    validate_refined_forest_arrays(
        root,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    validate_refined_all_touch_2to1(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    selected = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    kinds, masks, counts, source_ids = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        selected,
        CANONICAL_DIRECTIONS,
    )
    slots = np.empty_like(source_ids)
    resolve_refined_relation_source_slots(
        selected.size, selected, counts, source_ids, slots
    )
    phases = np.empty_like(source_ids, dtype=np.uint8)
    fill_refined_relation_phase_codes(
        selected,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        CANONICAL_DIRECTIONS,
        kinds,
        masks,
        counts,
        slots,
        phases,
    )
    return selected, kinds, masks, counts, slots, phases


def fresh_plan() -> tuple[np.ndarray, ...]:
    return (
        np.empty(PLAN_CAPACITY, dtype=np.int64),
        np.empty(PLAN_CAPACITY, dtype=np.uint8),
        np.empty(PLAN_CAPACITY, dtype=np.uint8),
        *(np.empty((PLAN_CAPACITY, 3), dtype=np.int64) for _ in range(10)),
    )


def axis_coded_payload(
    slots: int,
    fields: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(shape, dtype=np.float64)
    result = np.empty((slots, fields, *shape), dtype=np.float64)
    for slot in range(slots):
        for field in range(fields):
            result[slot, field] = (
                100000.0 * slot
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return result


def make_case(
    refined_root: tuple[int, int, int] = (0, 0, 0),
    direction: tuple[int, int, int] = (1, 0, 0),
    phase: int = 1,
    *,
    fields: int = 2,
) -> dict:
    selected, kinds, masks, counts, slots, phases = relation_case(
        (2, 2, 2), [refined_root]
    )
    row = direction_index(direction)
    primary = next(
        index
        for index in range(selected.size)
        if int(kinds[index, row]) == RELATION_COARSER
        and int(masks[index, row]) == 0
        and int(phases[index, row, 0]) == phase
    )

    lower = i3(2, 2, 2)
    upper = i3(6, 6, 6)
    target_lower = lower.copy()
    target_upper = upper.copy()
    for axis, component in enumerate(direction):
        if component < 0:
            target_lower[axis] = 0
            target_upper[axis] = 2
        elif component > 0:
            target_lower[axis] = 6
            target_upper[axis] = 8

    cwp = tuple(np.empty(3, dtype=np.int64) for _ in range(7))
    fill_coarser_workspace_boxes(
        lower,
        upper,
        np.asarray([direction], dtype=np.int64),
        np.asarray([phase], dtype=np.uint8),
        target_lower.reshape(1, 3),
        target_upper.reshape(1, 3),
        *(value.reshape(1, 3) for value in cwp),
    )
    plan = fresh_plan()
    transfer_count, record_count = fill_coarser_slope_support_plan(
        lower,
        upper,
        int(selected.size),
        primary,
        phase,
        np.asarray(direction, dtype=np.int64),
        CANONICAL_DIRECTIONS,
        kinds[primary],
        masks[primary],
        counts[primary],
        slots[primary],
        target_lower,
        target_upper,
        *cwp,
        *plan,
    )

    selected_payload = axis_coded_payload(
        int(selected.size), fields, (8, 8, 8)
    )
    workspace = np.full((1, fields, 5, 5, 5), np.nan, dtype=np.float64)
    modes = np.zeros((fields, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    return {
        "selected_payload": selected_payload,
        "coarse_workspace": workspace,
        "interior_lower": lower,
        "interior_upper": upper,
        "workspace_required_lower": cwp[4],
        "workspace_required_upper": cwp[5],
        "transfer_count": transfer_count,
        "record_count": record_count,
        "plan_source_slots": plan[0],
        "plan_source_is_fine": plan[1],
        "plan_physical_masks": plan[2],
        "plan_directions": plan[3],
        "plan_source_lower": plan[4],
        "plan_source_upper": plan[5],
        "plan_base_lower": plan[6],
        "plan_base_upper": plan[7],
        "plan_target_lower": plan[8],
        "plan_target_upper": plan[9],
        "plan_logical_interior_lower": plan[10],
        "plan_logical_interior_upper": plan[11],
        "plan_storage_logical_offsets": plan[12],
        "boundary_modes": modes,
        "normal_field_slots": normals,
        "workspace_coarse_origin": cwp[6],
        "fine_target_lower": target_lower,
        "fine_target_upper": target_upper,
    }


CWA_NAMES = (
    "selected_payload",
    "coarse_workspace",
    "interior_lower",
    "interior_upper",
    "workspace_required_lower",
    "workspace_required_upper",
    "transfer_count",
    "record_count",
    "plan_source_slots",
    "plan_source_is_fine",
    "plan_physical_masks",
    "plan_directions",
    "plan_source_lower",
    "plan_source_upper",
    "plan_base_lower",
    "plan_base_upper",
    "plan_target_lower",
    "plan_target_upper",
    "plan_logical_interior_lower",
    "plan_logical_interior_upper",
    "plan_storage_logical_offsets",
    "boundary_modes",
    "normal_field_slots",
)


def args(case: dict) -> tuple:
    return tuple(case[name] for name in CWA_NAMES)


def invoke(case: dict) -> None:
    apply_coarser_workspace_plan(*args(case))


def reference(case: dict) -> None:
    apply_coarser_workspace_plan_reference(*args(case))


def clone_case(case: dict) -> dict:
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in case.items()
    }


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def assert_atomic_error(case: dict, error, match: str) -> None:
    workspace_before = case["coarse_workspace"].copy()
    selected_before = case["selected_payload"].copy()
    with pytest.raises(error, match=match):
        invoke(case)
    assert_bits_equal(case["coarse_workspace"], workspace_before)
    assert_bits_equal(case["selected_payload"], selected_before)


@pytest.mark.parametrize(
    ("refined_root", "direction", "phase"),
    [
        ((0, 0, 0), (1, 0, 0), 1),
        ((0, 0, 0), (1, 1, 0), 3),
        ((0, 0, 0), (1, 1, 1), 7),
        ((1, 1, 1), (-1, 0, 0), 6),
    ],
)
def test_face_edge_corner_csp_plans_match_reference_and_preserve(
    refined_root, direction, phase
) -> None:
    case = make_case(refined_root, direction, phase)
    expected = clone_case(case)
    workspace_before = case["coarse_workspace"].copy()
    selected_before = case["selected_payload"].copy()

    normalized = _validate_coarser_workspace_plan(*args(case))
    assert_bits_equal(case["coarse_workspace"], workspace_before)
    _apply_coarser_workspace_plan_unchecked(*normalized)
    reference(expected)

    assert_bits_equal(case["coarse_workspace"], expected["coarse_workspace"])
    assert_bits_equal(case["selected_payload"], selected_before)
    lower = case["workspace_required_lower"]
    upper = case["workspace_required_upper"]
    required = tuple(slice(int(lower[a]), int(upper[a])) for a in range(3))
    assert not np.isnan(case["coarse_workspace"][(0, 0, *required)]).any()

    outside = np.ones(case["coarse_workspace"].shape[2:], dtype=np.bool_)
    outside[required] = False
    assert_bits_equal(
        case["coarse_workspace"][:, :, outside],
        workspace_before[:, :, outside],
    )


def test_mixed_physical_nonfinite_bits_and_read_only_inputs() -> None:
    case = make_case(fields=3)
    selected = case["selected_payload"]
    selected[:, 0] = np.inf
    selected[:, 1].view(np.uint64)[:] = np.uint64(0x8000000000000000)
    selected[:, 2].view(np.uint64)[:] = np.uint64(0x7FF8000000001234)
    case["boundary_modes"][0, :] = 3
    case["boundary_modes"][1, :] = 2
    case["boundary_modes"][2, :] = 1
    case["normal_field_slots"][:] = (0, 0, 0)
    expected = clone_case(case)
    selected_before = selected.copy()
    for name in CWA_NAMES:
        value = case[name]
        if isinstance(value, np.ndarray) and name != "coarse_workspace":
            value.setflags(write=False)

    with np.errstate(all="ignore"):
        invoke(case)
        reference(expected)
    assert_bits_equal(case["coarse_workspace"], expected["coarse_workspace"])
    assert_bits_equal(selected, selected_before)


def test_completed_workspace_composes_bitwise_with_prl() -> None:
    case = make_case(fields=2)
    expected = clone_case(case)
    invoke(case)
    reference(expected)
    assert_bits_equal(case["coarse_workspace"], expected["coarse_workspace"])

    actual_fine = np.full((1, 2, 8, 8, 8), np.nan, dtype=np.float64)
    expected_fine = actual_fine.copy()
    prolong_cartesian_2to1_into(
        case["coarse_workspace"],
        case["workspace_required_lower"],
        case["workspace_required_upper"],
        case["workspace_coarse_origin"],
        actual_fine,
        case["fine_target_lower"],
        case["fine_target_upper"],
        case["interior_lower"],
    )
    prolong_cartesian_2to1_reference(
        expected["coarse_workspace"],
        expected["workspace_required_lower"],
        expected["workspace_required_upper"],
        expected["workspace_coarse_origin"],
        expected_fine,
        expected["fine_target_lower"],
        expected["fine_target_upper"],
        expected["interior_lower"],
    )
    assert_bits_equal(actual_fine, expected_fine)


def test_empty_fields_are_valid_after_complete_plan_validation() -> None:
    case = make_case(fields=1)
    selected_count = case["selected_payload"].shape[0]
    case["selected_payload"] = np.empty(
        (selected_count, 0, 8, 8, 8), dtype=np.float64
    )
    case["coarse_workspace"] = np.empty((1, 0, 5, 5, 5), dtype=np.float64)
    case["boundary_modes"] = np.empty((0, 6), dtype=np.uint8)
    case["normal_field_slots"] = i3(-1, -1, -1)
    invoke(case)


@pytest.mark.parametrize(
    "corruption",
    [
        "counts",
        "slot",
        "flag",
        "transfer_mask",
        "transfer_logical",
        "source_ratio",
        "target_overlap",
        "target_gap",
        "physical_slot",
        "physical_source",
        "physical_base",
        "inactive",
        "mode",
    ],
)
def test_corrupt_plan_and_boundary_failures_are_atomic(corruption: str) -> None:
    case = make_case()
    transfer_count = case["transfer_count"]
    record_count = case["record_count"]
    physical = transfer_count if transfer_count < record_count else None

    if corruption == "counts":
        case["transfer_count"] = record_count + 1
        match = "counts"
    elif corruption == "slot":
        case["plan_source_slots"][0] = case["selected_payload"].shape[0]
        match = "slot"
    elif corruption == "flag":
        case["plan_source_is_fine"][0] = 2
        match = "source_is_fine"
    elif corruption == "transfer_mask":
        case["plan_physical_masks"][0] = 1
        match = "transfer physical mask"
    elif corruption == "transfer_logical":
        case["plan_storage_logical_offsets"][0, 0] = 1
        match = "logical metadata"
    elif corruption == "source_ratio":
        case["plan_source_upper"][0, 0] -= 1
        match = "extent ratio"
    elif corruption == "target_overlap":
        case["plan_target_lower"][1] = case["plan_target_lower"][0]
        case["plan_target_upper"][1] = case["plan_target_upper"][0]
        match = "pairwise disjoint"
    elif corruption == "target_gap":
        case["plan_target_upper"][0, 0] -= 1
        case["plan_base_upper"][0, 0] -= 1
        match = "exactly cover"
    elif corruption == "physical_slot":
        assert physical is not None
        case["plan_source_slots"][physical] = 0
        match = "physical source slot"
    elif corruption == "physical_source":
        assert physical is not None
        case["plan_source_lower"][physical, 0] = 1
        match = "physical source boxes"
    elif corruption == "physical_base":
        assert physical is not None
        case["plan_base_lower"][physical] = (0, 0, 0)
        case["plan_base_upper"][physical] = (1, 1, 1)
        match = "physical base"
    elif corruption == "inactive":
        case["plan_source_slots"][record_count] = 0
        match = "inactive source slot"
    else:
        case["boundary_modes"][0, 0] = 4
        match = "unknown mode"
    assert_atomic_error(case, ValueError, match)


def test_workspace_and_plan_aliases_are_atomic() -> None:
    case = make_case()
    case["coarse_workspace"] = case["selected_payload"][:1]
    assert_atomic_error(case, ValueError, "must not overlap")

    case = make_case()
    raw = np.zeros(PLAN_CAPACITY * 3 * 8, dtype=np.uint8)
    case["plan_directions"] = raw.view(np.int64).reshape(PLAN_CAPACITY, 3)
    case["plan_physical_masks"] = raw[:PLAN_CAPACITY]
    assert_atomic_error(case, ValueError, "pairwise nonoverlapping")


def test_pwa_preflight_failures_through_cwa_are_atomic() -> None:
    case = make_case()
    physical = case["transfer_count"]
    assert physical < case["record_count"]
    masked_axis = next(
        axis
        for axis in range(3)
        if int(case["plan_physical_masks"][physical]) & (1 << axis)
    )
    case["plan_storage_logical_offsets"][physical, masked_axis] = np.iinfo(
        np.int64
    ).max
    assert_atomic_error(case, OverflowError, "forward target translation")

    case = make_case()
    case["boundary_modes"][0, 0] = 3
    assert_atomic_error(case, ValueError, "no-inflow")


def test_representation_and_scalar_failures_are_atomic() -> None:
    case = make_case()
    case["transfer_count"] = True
    assert_atomic_error(case, TypeError, "integer")

    case = make_case()
    case["record_count"] = np.asarray(1, dtype=np.int64)
    assert_atomic_error(case, TypeError, "integer")

    case = make_case()
    case["plan_target_lower"] = case["plan_target_lower"].astype(np.int32)
    assert_atomic_error(case, TypeError, "dtype int64")

    case = make_case()
    case["coarse_workspace"].setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        invoke(case)
