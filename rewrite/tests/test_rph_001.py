from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest as CurrentAMRForest
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.prolongation import prolong_cartesian_2to1_into
from simesh_rewrite.refined_support import (
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relation_phases import (
    fill_refined_relation_phase_codes,
)
from simesh_rewrite.relation_phases_reference import (
    refined_relation_phase_codes_reference,
)
from simesh_rewrite.relation_slots import (
    resolve_refined_relation_source_slots,
)
from simesh_rewrite.relations import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    balanced_refined_relations,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into


NO_PHASE = np.uint8(255)
ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)
FACE_DIRECTIONS = np.asarray(
    [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ],
    dtype=np.int64,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def make_flags(
    root: np.ndarray,
    refined_roots: list[tuple[int, int, int]],
) -> np.ndarray:
    _, root_coords = level1_morton(root)
    flags: list[bool] = []
    for root_coord_array in root_coords:
        root_coord = tuple(int(value) for value in root_coord_array)
        split = any(root_coord == candidate for candidate in refined_roots)
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    return np.asarray(flags, dtype=np.bool_)


def forest_artifact(
    root_shape: tuple[int, int, int],
    refined_roots: list[tuple[int, int, int]],
):
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = make_flags(root, refined_roots)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
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
    return root, coord_to_rank, flags, forest


def relation_case(
    root_shape: tuple[int, int, int],
    refined_roots: list[tuple[int, int, int]],
    directions: np.ndarray = ALL_DIRECTIONS,
    selected: np.ndarray | None = None,
    primary_count: int | None = None,
):
    root, coord_to_rank, flags, forest = forest_artifact(
        root_shape, refined_roots
    )
    leaf_count = int(forest.leaf_node_ids.size)
    if selected is None:
        selected = np.arange(leaf_count, dtype=np.int64)
    if primary_count is None:
        primary_count = int(selected.size)
    kinds, masks, counts, source_leaf_ids = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        selected[:primary_count],
        directions,
    )
    source_slots = np.empty_like(source_leaf_ids)
    resolve_refined_relation_source_slots(
        leaf_count,
        selected,
        counts,
        source_leaf_ids,
        source_slots,
    )
    return (
        root,
        coord_to_rank,
        flags,
        forest,
        selected,
        directions,
        kinds,
        masks,
        counts,
        source_leaf_ids,
        source_slots,
    )


def phase_arguments(case) -> tuple[np.ndarray, ...]:
    forest = case[3]
    return (
        case[4],
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        case[5],
        case[6],
        case[7],
        case[8],
        case[10],
    )


def production_phases(case) -> np.ndarray:
    output = np.full(case[10].shape, 0xA5, dtype=np.uint8)
    fill_refined_relation_phase_codes(*phase_arguments(case), output)
    return output


def assert_reference_equal(case) -> np.ndarray:
    actual = production_phases(case)
    expected = refined_relation_phase_codes_reference(*phase_arguments(case))
    assert np.array_equal(actual, expected)
    return actual


def leaf_id_at(forest, level: int, coord: tuple[int, int, int]) -> int:
    for leaf_id in range(int(forest.leaf_node_ids.size)):
        node = int(forest.leaf_node_ids[leaf_id])
        if int(forest.node_levels[node]) != level:
            continue
        if all(
            int(forest.node_coords[node, axis]) == coord[axis]
            for axis in range(3)
        ):
            return leaf_id
    raise AssertionError("requested leaf coordinate is absent")


def node_phase(forest, leaf_id: int) -> int:
    node = int(forest.leaf_node_ids[leaf_id])
    return sum(
        (int(forest.node_coords[node, axis]) & 1) << axis
        for axis in range(3)
    )


def first_record(
    kinds: np.ndarray,
    kind: int,
    counts: np.ndarray | None = None,
    count: int | None = None,
    masks: np.ndarray | None = None,
    require_unmasked: bool = False,
) -> tuple[int, int]:
    for primary in range(kinds.shape[0]):
        for direction in range(kinds.shape[1]):
            if int(kinds[primary, direction]) != kind:
                continue
            if count is not None and int(counts[primary, direction]) != count:
                continue
            if require_unmasked and int(masks[primary, direction]) != 0:
                continue
            return primary, direction
    raise AssertionError("requested relation record is absent")


def test_all_kinds_eight_phases_counts_and_canonical_order() -> None:
    case = relation_case((3, 3, 3), [(1, 1, 1)])
    forest = case[3]
    selected = case[4]
    directions = case[5]
    kinds, masks, counts, slots = case[6], case[7], case[8], case[10]
    phases = assert_reference_equal(case)
    kinds_seen = [False] * 5
    phases_seen = [False] * 8
    finer_counts_seen = [False] * 5

    for primary in range(kinds.shape[0]):
        primary_phase = node_phase(forest, int(selected[primary]))
        for direction_index in range(kinds.shape[1]):
            kind = int(kinds[primary, direction_index])
            mask = int(masks[primary, direction_index])
            count = int(counts[primary, direction_index])
            kinds_seen[kind] = True
            if kind in (RELATION_PHYSICAL, RELATION_SAME):
                assert np.all(phases[primary, direction_index] == NO_PHASE)
            elif kind == RELATION_COARSER:
                assert phases[primary, direction_index].tolist() == [
                    primary_phase,
                    255,
                    255,
                    255,
                ]
                phases_seen[primary_phase] = True
            else:
                finer_counts_seen[count] = True
                active = phases[primary, direction_index, :count]
                assert all(
                    int(active[index]) < int(active[index + 1])
                    for index in range(count - 1)
                )
                direction = directions[direction_index]
                expected_count = 1 << sum(
                    int(direction[axis]) == 0 or bool(mask & (1 << axis))
                    for axis in range(3)
                )
                assert count == expected_count
                for source in range(count):
                    slot = int(slots[primary, direction_index, source])
                    source_phase = node_phase(forest, int(selected[slot]))
                    assert int(active[source]) == source_phase
                    phases_seen[source_phase] = True
                    for axis in range(3):
                        if mask & (1 << axis):
                            continue
                        component = int(direction[axis])
                        bit = (source_phase >> axis) & 1
                        assert component == 0 or bit == (component < 0)
                assert np.all(
                    phases[primary, direction_index, count:] == NO_PHASE
                )

    assert kinds_seen[1:] == [True, True, True, True]
    assert phases_seen == [True] * 8
    assert finer_counts_seen[1]
    assert finer_counts_seen[2]
    assert finer_counts_seen[4]


def test_coarser_floor_identity_accepts_tangential_high_phase() -> None:
    root, coord_to_rank, flags, forest = forest_artifact(
        (1, 2, 1), [(0, 0, 0)]
    )
    primary_leaf = leaf_id_at(forest, 2, (1, 1, 1))
    selected_values = [primary_leaf]
    for leaf_id in range(int(forest.leaf_node_ids.size)):
        if leaf_id != primary_leaf:
            selected_values.append(leaf_id)
    selected = np.asarray(selected_values, dtype=np.int64)
    direction = np.asarray([(0, 1, 0)], dtype=np.int64)
    case = relation_case(
        (1, 2, 1),
        [(0, 0, 0)],
        direction,
        selected,
        1,
    )
    del root, coord_to_rank, flags
    phases = assert_reference_equal(case)
    assert int(case[6][0, 0]) == RELATION_COARSER
    assert int(case[7][0, 0]) == 0
    assert int(case[8][0, 0]) == 1
    source_slot = int(case[10][0, 0, 0])
    source_leaf = int(selected[source_slot])
    source_node = int(forest.leaf_node_ids[source_leaf])
    primary_node = int(forest.leaf_node_ids[primary_leaf])
    expected_coord = [
        (int(forest.node_coords[primary_node, axis]) + int(direction[0, axis]))
        // 2
        for axis in range(3)
    ]
    assert forest.node_coords[source_node].tolist() == expected_coord
    assert phases[0, 0].tolist() == [7, 255, 255, 255]


@pytest.mark.parametrize(
    ("root_shape", "refined_root", "direction", "expected_phases"),
    [
        ((2, 1, 1), (1, 0, 0), (1, -1, 0), [0, 2, 4, 6]),
        ((2, 1, 2), (1, 0, 1), (1, -1, 1), [0, 2]),
    ],
)
def test_mixed_physical_finer_axes_are_neutral_candidates(
    root_shape,
    refined_root,
    direction,
    expected_phases,
) -> None:
    _, _, _, forest = forest_artifact(root_shape, [refined_root])
    primary_leaf = leaf_id_at(forest, 1, (0, 0, 0))
    selected_values = [primary_leaf]
    for leaf_id in range(int(forest.leaf_node_ids.size)):
        if leaf_id != primary_leaf:
            selected_values.append(leaf_id)
    case = relation_case(
        root_shape,
        [refined_root],
        np.asarray([direction], dtype=np.int64),
        np.asarray(selected_values, dtype=np.int64),
        1,
    )
    phases = assert_reference_equal(case)
    assert int(case[6][0, 0]) == RELATION_FINER
    assert int(case[7][0, 0]) & 2
    count = int(case[8][0, 0])
    assert count == len(expected_phases)
    assert phases[0, 0, :count].tolist() == expected_phases


def test_primary_support_order_and_selected_scaling_do_not_change_phases() -> None:
    root, coord_to_rank, _, forest = forest_artifact(
        (3, 3, 3), [(1, 1, 1)]
    )
    primary_ids = np.arange(5, dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        primary_ids,
        ALL_DIRECTIONS,
    )
    selected_values = [int(value) for value in primary_ids]
    for primary in range(counts.shape[0]):
        for direction in range(counts.shape[1]):
            for source in range(int(counts[primary, direction])):
                leaf_id = int(sources[primary, direction, source])
                if leaf_id not in selected_values:
                    selected_values.append(leaf_id)
    selected = np.asarray(selected_values, dtype=np.int64)

    def phases_for(selection: np.ndarray) -> np.ndarray:
        slots = np.empty_like(sources)
        resolve_refined_relation_source_slots(
            forest.leaf_node_ids.size,
            selection,
            counts,
            sources,
            slots,
        )
        output = np.empty_like(slots, dtype=np.uint8)
        fill_refined_relation_phase_codes(
            selection,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            ALL_DIRECTIONS,
            kinds,
            masks,
            counts,
            slots,
            output,
        )
        return output

    expected = phases_for(selected)
    alternate_values = selected_values[:5]
    for index in range(len(selected_values) - 1, 4, -1):
        alternate_values.append(selected_values[index])
    alternate = np.asarray(alternate_values, dtype=np.int64)
    assert np.array_equal(phases_for(alternate), expected)

    expanded_values = [int(value) for value in selected]
    for leaf_id in range(int(forest.leaf_node_ids.size)):
        if leaf_id not in expanded_values:
            expanded_values.append(leaf_id)
    expanded = np.asarray(expanded_values, dtype=np.int64)
    assert expanded.size >= selected.size > primary_ids.size
    assert np.array_equal(phases_for(expanded), expected)


def test_p_le_s_is_required_even_for_zero_directions() -> None:
    case = relation_case((1, 1, 1), [], np.empty((0, 3), dtype=np.int64))
    arguments = list(phase_arguments(case))
    output = np.empty(case[10].shape, dtype=np.uint8)
    fill_refined_relation_phase_codes(*arguments, output)
    assert output.shape == (1, 0, 4)

    arguments[0] = arguments[0][:0]
    before = output.copy()
    with pytest.raises(ValueError):
        fill_refined_relation_phase_codes(*arguments, output)
    assert np.array_equal(output, before)


def test_random_balanced_forests_match_reference() -> None:
    rng = np.random.default_rng(20260903)
    for _ in range(20):
        root_shape = tuple(int(value) for value in rng.integers(1, 4, size=3))
        root = i3(*root_shape)
        _, root_coords = level1_morton(root)
        refined_roots: list[tuple[int, int, int]] = []
        for coord in root_coords:
            if rng.random() < 0.3:
                refined_roots.append(tuple(int(value) for value in coord))
        _, _, _, forest = forest_artifact(root_shape, refined_roots)
        selected = rng.permutation(forest.leaf_node_ids.size).astype(np.int64)
        primary_count = int(rng.integers(0, selected.size + 1))
        case = relation_case(
            root_shape,
            refined_roots,
            ALL_DIRECTIONS,
            selected,
            primary_count,
        )
        assert_reference_equal(case)


def current_fine_phase_positions(
    direction: np.ndarray,
) -> tuple[list[int], list[int]]:
    phases: list[int] = []
    positions: list[int] = []
    for phase in range(8):
        bits = [(phase >> axis) & 1 for axis in range(3)]
        if not all(
            int(direction[axis]) == 0
            or bits[axis] == (1 if int(direction[axis]) < 0 else 0)
            for axis in range(3)
        ):
            continue
        shell = [
            0
            if int(direction[axis]) < 0
            else 3
            if int(direction[axis]) > 0
            else 1 + bits[axis]
            for axis in range(3)
        ]
        phases.append(phase)
        positions.append(shell[0] + 4 * shell[1] + 16 * shell[2])
    return phases, positions


def test_safe_current_unmasked_phase_facts() -> None:
    case = relation_case((2, 2, 2), [(0, 0, 0)])
    forest = case[3]
    selected = case[4]
    kinds, masks, counts, slots = case[6], case[7], case[8], case[10]
    phases = assert_reference_equal(case)
    current = CurrentAMRForest(3, 2, 2, 2, case[2].astype(np.int32))
    current_types = np.asarray(current.neighbor_type)
    current_ids = np.asarray(current.neighbor_index)
    current_children = np.asarray(current.neighbor_children)

    for primary in range(kinds.shape[0]):
        for direction_index, direction in enumerate(ALL_DIRECTIONS):
            if int(masks[primary, direction_index]) != 0:
                continue
            column = (
                (int(direction[2]) + 1) * 9
                + (int(direction[1]) + 1) * 3
                + int(direction[0])
                + 1
            )
            kind = int(kinds[primary, direction_index])
            assert int(current_types[int(selected[primary]), column]) == kind
            if kind == RELATION_COARSER:
                source_slot = int(slots[primary, direction_index, 0])
                assert int(current_ids[int(selected[primary]), column]) - 1 == int(
                    selected[source_slot]
                )
                assert int(phases[primary, direction_index, 0]) == node_phase(
                    forest, int(selected[primary])
                )
            elif kind == RELATION_FINER:
                expected_phases, positions = current_fine_phase_positions(
                    direction
                )
                count = int(counts[primary, direction_index])
                assert expected_phases == phases[
                    primary, direction_index, :count
                ].tolist()
                for source in range(count):
                    source_slot = int(slots[primary, direction_index, source])
                    assert int(
                        current_children[int(selected[primary]), positions[source]]
                    ) - 1 == int(selected[source_slot])
            else:
                assert np.all(phases[primary, direction_index] == NO_PHASE)


def test_singleton_axes_and_mixed_records_remain_valid() -> None:
    case = relation_case((1, 2, 1), [(0, 0, 0)])
    phases = assert_reference_equal(case)
    mixed = 0
    for primary in range(case[6].shape[0]):
        for direction in range(case[6].shape[1]):
            if (
                int(case[6][primary, direction]) != RELATION_PHYSICAL
                and int(case[7][primary, direction]) != 0
            ):
                mixed += 1
                count = int(case[8][primary, direction])
                if int(case[6][primary, direction]) == RELATION_FINER:
                    assert np.all(phases[primary, direction, :count] < 8)
    assert mixed > 0


def test_rst_and_prl_phase_consumers() -> None:
    case = relation_case((3, 3, 3), [(1, 1, 1)])
    phases = production_phases(case)
    finer_primary, finer_direction = first_record(
        case[6],
        RELATION_FINER,
        case[8],
        4,
        case[7],
        True,
    )
    active_phases = phases[finer_primary, finer_direction, :4]
    fine_payload = np.empty((4, 1, 2, 2, 2), dtype=np.float64)
    for source in range(4):
        fine_payload[source].fill(float(int(active_phases[source]) + 1))
    coarse_payload = np.full((4, 1, 1, 1, 1), np.nan)
    restrict_cartesian_2to1_into(
        fine_payload,
        i3(0, 0, 0),
        i3(2, 2, 2),
        coarse_payload,
        i3(0, 0, 0),
    )
    direction = case[5][finer_direction]
    neutral_axes = [
        axis for axis in range(3) if int(direction[axis]) == 0
    ]
    assert len(neutral_axes) == 2
    tile = np.full((2, 2), np.nan)
    for source in range(4):
        phase = int(active_phases[source])
        tile[
            (phase >> neutral_axes[0]) & 1,
            (phase >> neutral_axes[1]) & 1,
        ] = coarse_payload[source, 0, 0, 0, 0]
    assert np.all(np.isfinite(tile))
    for source in range(4):
        assert coarse_payload[source, 0, 0, 0, 0] == float(
            int(active_phases[source]) + 1
        )

    coarser_primary, coarser_direction = first_record(
        case[6], RELATION_COARSER
    )
    phase = int(phases[coarser_primary, coarser_direction, 0])
    coarse = np.empty((1, 1, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                coarse[0, 0, i, j, k] = float(i + 2 * j + 4 * k)
    fine = np.full((1, 1, 2, 2, 2), np.nan)
    prolong_cartesian_2to1_into(
        coarse,
        i3(0, 0, 0),
        i3(3, 3, 3),
        i3(1, 1, 1),
        fine,
        i3(0, 0, 0),
        i3(2, 2, 2),
        i3(0, 0, 0),
    )
    bits = tuple((phase >> axis) & 1 for axis in range(3))
    eta = [0.25 if bit else -0.25 for bit in bits]
    expected = 7.0 + eta[0] + 2.0 * eta[1] + 4.0 * eta[2]
    assert fine[(0, 0, *bits)] == expected


def test_weno_bounded_rows_match_reference_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags, _ = get_metadata(str(path))
    root = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags, dtype=np.bool_),
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
    leaf_count = int(forest.leaf_node_ids.size)
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        leaf_ids,
        FACE_DIRECTIONS,
    )
    capacity = 256
    selected = np.empty(capacity, dtype=np.int64)
    first = 0
    while first < leaf_count:
        candidate_count = min(capacity, leaf_count - first)
        primary_count, selected_count = plan_balanced_refined_support_prefix(
            first,
            leaf_count,
            counts[first : first + candidate_count],
            sources[first : first + candidate_count],
            selected,
        )
        accepted = slice(first, first + primary_count)
        accepted_counts = counts[accepted]
        accepted_sources = sources[accepted]
        slots = np.empty_like(accepted_sources)
        resolve_refined_relation_source_slots(
            leaf_count,
            selected[:selected_count],
            accepted_counts,
            accepted_sources,
            slots,
        )
        arguments = (
            selected[:selected_count],
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            FACE_DIRECTIONS,
            kinds[accepted],
            masks[accepted],
            accepted_counts,
            slots,
        )
        output = np.empty_like(slots, dtype=np.uint8)
        fill_refined_relation_phase_codes(*arguments, output)
        expected = refined_relation_phase_codes_reference(*arguments)
        assert np.array_equal(output, expected)
        first += primary_count
    assert first == leaf_count


def base_validation_arguments() -> tuple[np.ndarray, ...]:
    return phase_arguments(relation_case((2, 1, 1), [(0, 0, 0)]))


def clone_arguments(arguments: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    return [value.copy() for value in arguments]


def assert_atomic_failure(
    error_type: type[Exception],
    arguments,
    output: np.ndarray,
) -> None:
    before = output.copy()
    with pytest.raises(error_type):
        fill_refined_relation_phase_codes(*arguments, output)
    assert np.array_equal(output, before)


def noncontiguous_copy(value: np.ndarray) -> np.ndarray:
    if value.ndim == 1:
        return np.repeat(value, 2)[::2]
    return np.repeat(value, 2, axis=-1)[..., ::2]


def test_complete_input_representation_validation_is_atomic() -> None:
    base = base_validation_arguments()
    integer_indices = (0, 1, 2, 3, 4, 8)
    byte_indices = (5, 6, 7)
    for index in range(9):
        arguments = clone_arguments(base)
        output = np.full(base[8].shape, 0xC7, dtype=np.uint8)
        arguments[index] = arguments[index].tolist()
        assert_atomic_failure(TypeError, arguments, output)

        arguments = clone_arguments(base)
        output = np.full(base[8].shape, 0xC7, dtype=np.uint8)
        wrong_dtype = np.int32 if index in integer_indices else np.int64
        arguments[index] = arguments[index].astype(wrong_dtype)
        assert_atomic_failure(TypeError, arguments, output)

        if index in integer_indices:
            arguments = clone_arguments(base)
            output = np.full(base[8].shape, 0xC7, dtype=np.uint8)
            nonnative = np.dtype(">i8" if np.little_endian else "<i8")
            arguments[index] = arguments[index].astype(nonnative)
            assert_atomic_failure(TypeError, arguments, output)

        arguments = clone_arguments(base)
        output = np.full(base[8].shape, 0xC7, dtype=np.uint8)
        arguments[index] = noncontiguous_copy(arguments[index])
        assert_atomic_failure(ValueError, arguments, output)

    assert byte_indices == (5, 6, 7)


def test_complete_input_shape_validation_is_atomic() -> None:
    base = base_validation_arguments()
    replacements = [
        np.empty((1, base[0].size), dtype=np.int64),
        np.empty((1, base[1].size), dtype=np.int64),
        np.empty((base[2].shape[0], 2), dtype=np.int64),
        np.empty((1, base[3].size), dtype=np.int64),
        np.empty((base[4].shape[0], 2), dtype=np.int64),
        np.empty((*base[5].shape, 1), dtype=np.uint8),
        np.empty((*base[6].shape, 1), dtype=np.uint8),
        np.empty((*base[7].shape, 1), dtype=np.uint8),
        np.empty((*base[8].shape[:2], 3), dtype=np.int64),
    ]
    for index, replacement in enumerate(replacements):
        arguments = clone_arguments(base)
        output = np.full(base[8].shape, 0xD3, dtype=np.uint8)
        arguments[index] = replacement
        assert_atomic_failure(ValueError, arguments, output)


def test_output_representation_validation_is_atomic() -> None:
    arguments = base_validation_arguments()
    valid = np.full(arguments[8].shape, 0xA9, dtype=np.uint8)
    invalid_outputs = [
        valid.tolist(),
        valid.astype(np.int8),
        np.empty((*valid.shape[:2], 3), dtype=np.uint8),
        noncontiguous_copy(valid),
    ]
    for output in invalid_outputs:
        error = TypeError if not isinstance(output, np.ndarray) or output.dtype != np.uint8 else ValueError
        with pytest.raises(error):
            fill_refined_relation_phase_codes(*arguments, output)
    readonly = valid.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError):
        fill_refined_relation_phase_codes(*arguments, readonly)


def test_read_only_inputs_are_accepted_and_preserved() -> None:
    arguments = clone_arguments(base_validation_arguments())
    before = [value.copy() for value in arguments]
    for value in arguments:
        value.setflags(write=False)
    output = np.empty(arguments[8].shape, dtype=np.uint8)
    fill_refined_relation_phase_codes(*arguments, output)
    expected = refined_relation_phase_codes_reference(*arguments)
    assert np.array_equal(output, expected)
    for value, original in zip(arguments, before, strict=True):
        assert np.array_equal(value, original)


def test_selected_direction_mask_kind_count_and_slot_errors_are_atomic() -> None:
    base = base_validation_arguments()
    semantic_mutations: list[tuple[int, object]] = []

    duplicate = base[0].copy()
    duplicate[1] = duplicate[0]
    semantic_mutations.append((0, duplicate))
    bad_selected = base[0].copy()
    bad_selected[0] = base[3].size
    semantic_mutations.append((0, bad_selected))
    bad_leaf_map = base[3].copy()
    bad_leaf_map[int(base[0][0])] = base[1].size
    semantic_mutations.append((3, bad_leaf_map))
    bad_direction = base[4].copy()
    bad_direction[0, 0] = 2
    semantic_mutations.append((4, bad_direction))
    center = base[4].copy()
    center[0] = 0
    semantic_mutations.append((4, center))
    bad_mask_bits = base[6].copy()
    bad_mask_bits[0, 0] |= np.uint8(8)
    semantic_mutations.append((6, bad_mask_bits))
    bad_kind = base[5].copy()
    bad_kind[0, 0] = np.uint8(5)
    semantic_mutations.append((5, bad_kind))
    bad_count = base[7].copy()
    bad_count[0, 0] = np.uint8(5)
    semantic_mutations.append((7, bad_count))

    active_record = None
    inactive_record = None
    for primary in range(base[7].shape[0]):
        for direction in range(base[7].shape[1]):
            count = int(base[7][primary, direction])
            if active_record is None and count:
                active_record = (primary, direction)
            if inactive_record is None and count < 4:
                inactive_record = (primary, direction, count)
    assert active_record is not None and inactive_record is not None
    bad_active = base[8].copy()
    bad_active[active_record[0], active_record[1], 0] = base[0].size
    semantic_mutations.append((8, bad_active))
    bad_trailing = base[8].copy()
    bad_trailing[
        inactive_record[0], inactive_record[1], inactive_record[2]
    ] = 0
    semantic_mutations.append((8, bad_trailing))

    zero_axis_record = None
    for primary in range(base[5].shape[0]):
        for direction in range(base[5].shape[1]):
            for axis in range(3):
                if int(base[4][direction, axis]) == 0:
                    zero_axis_record = (primary, direction, axis)
                    break
            if zero_axis_record is not None:
                break
        if zero_axis_record is not None:
            break
    bad_zero_mask = base[6].copy()
    bad_zero_mask[zero_axis_record[0], zero_axis_record[1]] |= np.uint8(
        1 << zero_axis_record[2]
    )
    semantic_mutations.append((6, bad_zero_mask))

    for index, replacement in semantic_mutations:
        arguments = clone_arguments(base)
        arguments[index] = replacement
        output = np.full(base[8].shape, 0xE1, dtype=np.uint8)
        assert_atomic_failure(ValueError, arguments, output)


def test_relative_level_coordinate_finer_count_phase_and_order_errors_are_atomic() -> None:
    base = base_validation_arguments()
    kinds, masks, counts, slots = base[5], base[6], base[7], base[8]
    records = [
        first_record(kinds, RELATION_SAME),
        first_record(kinds, RELATION_COARSER),
        first_record(kinds, RELATION_FINER, counts, 4),
    ]

    for kind, record in zip(
        (RELATION_SAME, RELATION_COARSER, RELATION_FINER),
        records,
        strict=True,
    ):
        primary, direction = record
        source_slot = int(slots[primary, direction, 0])
        source_leaf = int(base[0][source_slot])
        source_node = int(base[3][source_leaf])
        arguments = clone_arguments(base)
        arguments[1][source_node] += 3
        output = np.full(base[8].shape, 0xB7, dtype=np.uint8)
        assert_atomic_failure(ValueError, arguments, output)
        assert kind in (RELATION_SAME, RELATION_COARSER, RELATION_FINER)

    coarser_primary, coarser_direction = records[1]
    coarser_slot = int(slots[coarser_primary, coarser_direction, 0])
    coarser_leaf = int(base[0][coarser_slot])
    coarser_node = int(base[3][coarser_leaf])
    arguments = clone_arguments(base)
    arguments[2][coarser_node, 0] += 1
    output = np.full(base[8].shape, 0xB7, dtype=np.uint8)
    assert_atomic_failure(ValueError, arguments, output)

    finer_primary, finer_direction = records[2]
    direction = base[4][finer_direction]
    unmasked_axis = next(
        axis
        for axis in range(3)
        if int(direction[axis]) != 0
        and not int(masks[finer_primary, finer_direction]) & (1 << axis)
    )
    finer_slot = int(slots[finer_primary, finer_direction, 0])
    finer_leaf = int(base[0][finer_slot])
    finer_node = int(base[3][finer_leaf])
    arguments = clone_arguments(base)
    arguments[2][finer_node, unmasked_axis] ^= 1
    output = np.full(base[8].shape, 0xB7, dtype=np.uint8)
    assert_atomic_failure(ValueError, arguments, output)

    arguments = clone_arguments(base)
    arguments[8][finer_primary, finer_direction, 0:2] = arguments[8][
        finer_primary, finer_direction, 1::-1
    ].copy()
    output = np.full(base[8].shape, 0xB7, dtype=np.uint8)
    assert_atomic_failure(ValueError, arguments, output)

    arguments = clone_arguments(base)
    arguments[7][finer_primary, finer_direction] = np.uint8(2)
    arguments[8][finer_primary, finer_direction, 2:] = -1
    output = np.full(base[8].shape, 0xB7, dtype=np.uint8)
    assert_atomic_failure(ValueError, arguments, output)


def test_physical_and_nonphysical_mask_semantics_are_atomic() -> None:
    base = base_validation_arguments()
    physical = first_record(base[5], RELATION_PHYSICAL)
    arguments = clone_arguments(base)
    direction = arguments[4][physical[1]]
    for axis in range(3):
        if int(direction[axis]) != 0:
            arguments[6][physical] &= np.uint8(~(1 << axis) & 0xFF)
            break
    output = np.full(base[8].shape, 0x93, dtype=np.uint8)
    assert_atomic_failure(ValueError, arguments, output)

    same = first_record(base[5], RELATION_SAME)
    arguments = clone_arguments(base)
    direction = arguments[4][same[1]]
    mask = 0
    for axis in range(3):
        if int(direction[axis]) != 0:
            mask |= 1 << axis
    arguments[6][same] = np.uint8(mask)
    output = np.full(base[8].shape, 0x93, dtype=np.uint8)
    assert_atomic_failure(ValueError, arguments, output)


def test_output_overlap_with_every_input_is_atomic() -> None:
    base = base_validation_arguments()
    output_shape = base[8].shape
    output_nbytes = int(np.prod(output_shape, dtype=np.int64))
    for input_index in range(9):
        arguments = clone_arguments(base)
        source = arguments[input_index]
        raw = np.zeros(max(source.nbytes, output_nbytes), dtype=np.uint8)
        shared_source = np.ndarray(
            source.shape, dtype=source.dtype, buffer=raw, offset=0
        )
        np.copyto(shared_source, source)
        shared_output = np.ndarray(
            output_shape, dtype=np.uint8, buffer=raw, offset=0
        )
        arguments[input_index] = shared_source
        assert_atomic_failure(ValueError, arguments, shared_output)
