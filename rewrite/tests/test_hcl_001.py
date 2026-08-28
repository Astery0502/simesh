from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from simesh_rewrite.chunking import (
    minimum_halo_closed_slots,
    plan_level1_halo_chunk,
)
from simesh_rewrite.halo_closure import (
    minimum_level1_halo_closed_slots,
    plan_level1_halo_closed_prefix,
)
from simesh_rewrite.halo_closure_reference import (
    minimum_level1_halo_closed_slots_reference,
    plan_level1_halo_closed_prefix_reference,
)
from simesh_rewrite.halo_plans import level1_halo_relation_plan
from simesh_rewrite.halo_plans_reference import (
    level1_halo_relation_plan_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


INT64_MAX = int(np.iinfo(np.int64).max)
INT64_MIN = int(np.iinfo(np.int64).min)
ALL_DIRECTIONS = tuple(
    (dx, dy, dz)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def topology(shape: tuple[int, int, int]):
    root = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    return coord_to_rank, rank_to_coord, faces


def walk_direction(
    block_id: int,
    direction: tuple[int, int, int],
    faces: np.ndarray,
    order=(0, 1, 2),
) -> int:
    current = int(block_id)
    for axis in order:
        delta = direction[axis]
        if delta:
            current = int(faces[current, 2 * axis + (1 if delta > 0 else 0)])
            if current < 0:
                return -1
    return current


def closure_ids(block_id: int, faces: np.ndarray) -> list[int]:
    values = [block_id]
    for direction in ALL_DIRECTIONS:
        if direction == (0, 0, 0):
            continue
        neighbor = walk_direction(block_id, direction, faces)
        if neighbor >= 0 and neighbor not in values:
            values.append(neighbor)
    return values


def coordinate_closure(
    coord_to_rank: np.ndarray,
    source: tuple[int, int, int],
) -> list[int]:
    values = [int(coord_to_rank[source])]
    shape = coord_to_rank.shape
    for dx, dy, dz in ALL_DIRECTIONS:
        if (dx, dy, dz) == (0, 0, 0):
            continue
        target = (source[0] + dx, source[1] + dy, source[2] + dz)
        if all(0 <= target[axis] < shape[axis] for axis in range(3)):
            values.append(int(coord_to_rank[target]))
    return values


def outcome(function, *arguments):
    try:
        return ("return", function(*arguments))
    except Exception as error:
        return (type(error), str(error))


def reference_outcome(first: int, faces: np.ndarray, capacity: int):
    if first < faces.shape[0] and capacity == 0:
        return (
            ValueError,
            "selected capacity must be positive before the end",
        )
    try:
        return (
            "return",
            plan_level1_halo_closed_prefix_reference(first, faces, capacity),
        )
    except Exception as error:
        return (type(error), str(error))


def assert_plan_invariants(
    first: int,
    faces: np.ndarray,
    selected_ids: np.ndarray,
    primary_count: int,
    selected_count: int,
) -> None:
    primaries = selected_ids[:primary_count]
    support = selected_ids[primary_count:selected_count]
    selected = selected_ids[:selected_count]
    assert primaries.tolist() == list(range(first, first + primary_count))
    assert len(np.unique(selected)) == selected_count
    assert not set(int(value) for value in primaries) & set(
        int(value) for value in support
    )
    expected_union = set(int(value) for value in primaries)
    for primary in primaries:
        expected_union.update(closure_ids(int(primary), faces))
    assert set(int(value) for value in selected) == expected_union


def next_trial_size(
    first: int,
    faces: np.ndarray,
    selected_ids: np.ndarray,
    primary_count: int,
    selected_count: int,
) -> int:
    candidate = first + primary_count
    trial_primaries = list(range(first, candidate + 1))
    trial_support = [
        int(value)
        for value in selected_ids[primary_count:selected_count]
        if value != candidate
    ]
    for neighbor in closure_ids(candidate, faces)[1:]:
        if neighbor not in trial_primaries and neighbor not in trial_support:
            trial_support.append(neighbor)
    return len(trial_primaries) + len(trial_support)


@pytest.mark.parametrize(
    ("shape", "source", "expected_size"),
    [
        ((1, 1, 1), (0, 0, 0), 1),
        ((5, 1, 1), (2, 0, 0), 3),
        ((5, 5, 5), (0, 0, 0), 8),
        ((5, 5, 1), (2, 2, 0), 9),
        ((5, 5, 5), (2, 0, 0), 12),
        ((5, 5, 5), (0, 2, 2), 18),
        ((5, 5, 5), (2, 2, 2), 27),
    ],
)
def test_exact_coordinate_clipping_and_x_fast_order(shape, source, expected_size) -> None:
    coord_to_rank, _, faces = topology(shape)
    expected = coordinate_closure(coord_to_rank, source)
    assert len(expected) == expected_size
    first = int(coord_to_rank[source])
    selected = np.full(expected_size, -91, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_closed_prefix(
        first, faces, selected
    )
    assert (primary_count, selected_count) == (1, expected_size)
    assert selected.tolist() == expected
    assert selected.tolist() == closure_ids(first, faces)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 1), 1),
        ((5, 1, 1), 3),
        ((2, 2, 2), 8),
        ((5, 5, 1), 9),
        ((5, 2, 2), 12),
        ((2, 5, 5), 18),
        ((3, 3, 3), 27),
    ],
)
def test_minimum_clipped_sizes_match_coordinate_product_and_reference(
    shape, expected
) -> None:
    _, _, faces = topology(shape)
    actual = minimum_level1_halo_closed_slots(faces)
    assert type(actual) is int
    assert actual == expected
    assert actual == minimum_level1_halo_closed_slots_reference(faces)
    assert minimum_halo_closed_slots(faces) == actual


def test_empty_minimum_and_end_are_zero_without_mutation() -> None:
    faces = np.empty((0, 6), dtype=np.int64)
    assert minimum_level1_halo_closed_slots(faces) == 0
    assert minimum_halo_closed_slots(faces) == 0
    for capacity in (0, 1, 27):
        selected = np.full(capacity, INT64_MIN, dtype=np.int64)
        before = selected.copy()
        assert plan_level1_halo_closed_prefix(0, faces, selected) == (0, 0)
        assert np.array_equal(selected, before)
        legacy = before.copy()
        assert plan_level1_halo_chunk(0, faces, legacy) == (0, 0)
        assert np.array_equal(legacy, before)


def test_minimum_capacity_guarantees_progress_and_argmax_failure_is_atomic() -> None:
    for shape in ((1, 1, 1), (5, 1, 1), (5, 5, 1), (5, 2, 2), (5, 4, 3)):
        _, _, faces = topology(shape)
        minimum = minimum_level1_halo_closed_slots(faces)
        selected = np.full(minimum, -73, dtype=np.int64)
        first = 0
        covered: list[int] = []
        while first < faces.shape[0]:
            primary_count, selected_count = plan_level1_halo_closed_prefix(
                first, faces, selected
            )
            assert 0 < primary_count <= selected_count <= minimum
            covered.extend(selected[:primary_count].tolist())
            first += primary_count
        assert covered == list(range(faces.shape[0]))

        sizes = np.asarray(
            [len(closure_ids(block, faces)) for block in range(faces.shape[0])]
        )
        argmax = int(np.argmax(sizes))
        too_small = np.full(max(0, minimum - 1), -79, dtype=np.int64)
        before = too_small.copy()
        expected_message = (
            "selected capacity must be positive before the end"
            if minimum == 1
            else "chunk capacity cannot fit the first halo closure"
        )
        with pytest.raises(ValueError, match=f"^{expected_message}$"):
            plan_level1_halo_closed_prefix(argmax, faces, too_small)
        assert np.array_equal(too_small, before)


def test_fixed_xyz_walks_commute_on_valid_topologies() -> None:
    _, _, faces = topology((4, 3, 2))
    for block_id in range(faces.shape[0]):
        for direction in ALL_DIRECTIONS:
            axes = tuple(
                axis for axis, delta in enumerate(direction) if delta
            )
            if not axes:
                assert walk_direction(block_id, direction, faces) == block_id
                continue
            results = {
                walk_direction(block_id, direction, faces, order)
                for order in permutations(axes)
            }
            assert results == {walk_direction(block_id, direction, faces)}


def test_promotion_overlapping_support_and_stable_order_match_reference() -> None:
    _, _, faces = topology((5, 1, 1))
    selected = np.full(3, -1, dtype=np.int64)
    assert plan_level1_halo_closed_prefix(0, faces, selected) == (2, 3)
    assert selected.tolist() == [0, 1, 2]

    coord_to_rank, _, faces = topology((5, 4, 3))
    first = int(coord_to_rank[2, 2, 1])
    selected = np.full(64, -83, dtype=np.int64)
    before = selected.copy()
    expected, expected_primary, expected_selected = (
        plan_level1_halo_closed_prefix_reference(first, faces, selected.size)
    )
    actual = plan_level1_halo_closed_prefix(first, faces, selected)
    assert actual == (expected_primary, expected_selected)
    assert selected[:expected_selected].tolist() == expected
    assert np.array_equal(selected[expected_selected:], before[expected_selected:])
    assert_plan_invariants(
        first, faces, selected, expected_primary, expected_selected
    )


@pytest.mark.parametrize("capacity", [8, 12, 18, 27, 40, 64, 96])
def test_reference_maximality_suffix_and_later_boundary(capacity) -> None:
    _, _, faces = topology((6, 4, 3))
    first = 0
    selected = np.asarray(
        [INT64_MIN + index for index in range(capacity)], dtype=np.int64
    )
    before = selected.copy()
    expected = reference_outcome(first, faces, capacity)
    actual = outcome(plan_level1_halo_closed_prefix, first, faces, selected)
    if expected[0] != "return":
        assert actual == expected
        assert np.array_equal(selected, before)
        return
    values, primary_count, selected_count = expected[1]
    assert actual == ("return", (primary_count, selected_count))
    assert selected[:selected_count].tolist() == values
    assert np.array_equal(selected[selected_count:], before[selected_count:])
    assert_plan_invariants(first, faces, selected, primary_count, selected_count)
    if first + primary_count < faces.shape[0]:
        assert next_trial_size(
            first,
            faces,
            selected,
            primary_count,
            selected_count,
        ) > capacity


def test_end_zero_capacity_and_first_fit_failures_are_atomic() -> None:
    coord_to_rank, _, faces = topology((3, 3, 3))
    end = np.asarray([INT64_MIN, -1, INT64_MAX], dtype=np.int64)
    before = end.copy()
    assert plan_level1_halo_closed_prefix(faces.shape[0], faces, end) == (0, 0)
    assert np.array_equal(end, before)

    empty = np.empty(0, dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="^selected capacity must be positive before the end$",
    ):
        plan_level1_halo_closed_prefix(0, faces, empty)

    center = int(coord_to_rank[1, 1, 1])
    too_small = np.full(26, -89, dtype=np.int64)
    before = too_small.copy()
    with pytest.raises(
        ValueError,
        match="^chunk capacity cannot fit the first halo closure$",
    ):
        plan_level1_halo_closed_prefix(center, faces, too_small)
    assert np.array_equal(too_small, before)


def compare_one_case(first: int, faces: np.ndarray, capacity: int) -> None:
    initial = np.asarray(
        [INT64_MIN + index for index in range(capacity)], dtype=np.int64
    )
    selected = initial.copy()
    expected = reference_outcome(first, faces, capacity)
    actual = outcome(plan_level1_halo_closed_prefix, first, faces, selected)
    if expected[0] != "return":
        assert actual == expected
        assert np.array_equal(selected, initial)
        return
    values, primary_count, selected_count = expected[1]
    assert actual == ("return", (primary_count, selected_count))
    assert selected[:selected_count].tolist() == values
    assert np.array_equal(selected[selected_count:], initial[selected_count:])
    if primary_count:
        assert_plan_invariants(
            first, faces, selected, primary_count, selected_count
        )


def test_exhaustive_small_topologies_starts_and_capacities_match_reference() -> None:
    for nz in range(1, 4):
        for ny in range(1, 4):
            for nx in range(1, 4):
                _, _, faces = topology((nx, ny, nz))
                maximum_capacity = min(faces.shape[0] + 26, 32)
                for first in range(faces.shape[0] + 1):
                    for capacity in range(maximum_capacity + 1):
                        compare_one_case(first, faces, capacity)


def test_random_full_traversals_match_reference_and_cover_primaries_once() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 7, size=3))
        _, _, faces = topology(shape)
        minimum = minimum_level1_halo_closed_slots(faces)
        capacity = int(rng.integers(minimum, max(minimum + 1, 97)))
        selected = np.full(capacity, -97, dtype=np.int64)
        first = 0
        covered: list[int] = []
        while first < faces.shape[0]:
            before = selected.copy()
            expected, expected_primary, expected_selected = (
                plan_level1_halo_closed_prefix_reference(
                    first, faces, capacity
                )
            )
            actual = plan_level1_halo_closed_prefix(first, faces, selected)
            assert actual == (expected_primary, expected_selected)
            assert selected[:expected_selected].tolist() == expected
            assert np.array_equal(
                selected[expected_selected:], before[expected_selected:]
            )
            assert_plan_invariants(
                first,
                faces,
                selected,
                expected_primary,
                expected_selected,
            )
            covered.extend(selected[:expected_primary].tolist())
            first += expected_primary
        assert covered == list(range(faces.shape[0]))


def test_canonical_face_output_scalar_overlap_and_precedence_errors() -> None:
    _, _, faces = topology((3, 3, 3))
    noncontiguous_faces = np.empty((4, 12), dtype=np.int64)[:, ::2]
    face_cases = [
        (
            [[-1] * 6],
            TypeError,
            "face_neighbor_ids must be a NumPy array",
        ),
        (
            np.full((1, 6), -1, dtype=np.int32),
            TypeError,
            "face_neighbor_ids must have dtype int64",
        ),
        (
            np.full((1, 5), -1, dtype=np.int64),
            ValueError,
            "face_neighbor_ids must have shape (block_count, 6), got (1, 5)",
        ),
        (
            noncontiguous_faces,
            ValueError,
            "face_neighbor_ids must be C-contiguous",
        ),
    ]
    for bad_faces, error_type, message in face_cases:
        selected = np.full(27, -1, dtype=np.int64)
        assert outcome(
            plan_level1_halo_closed_prefix, 0, bad_faces, selected
        ) == (error_type, message)
        assert outcome(minimum_level1_halo_closed_slots, bad_faces) == (
            error_type,
            message,
        )

    readonly = np.full(27, -1, dtype=np.int64)
    readonly.setflags(write=False)
    output_cases = [
        ([1, 2], TypeError, "selected_ids must be a NumPy array"),
        (
            np.full(27, -1, dtype=np.int32),
            TypeError,
            "selected_ids must have dtype int64",
        ),
        (
            np.full((1, 27), -1, dtype=np.int64),
            ValueError,
            "selected_ids must be a C-contiguous vector",
        ),
        (
            np.arange(54, dtype=np.int64)[::2],
            ValueError,
            "selected_ids must be a C-contiguous vector",
        ),
        (readonly, ValueError, "selected_ids must be writable"),
    ]
    for selected, error_type, message in output_cases:
        assert outcome(
            plan_level1_halo_closed_prefix, 0, faces, selected
        ) == (error_type, message)

    valid = np.full(27, -1, dtype=np.int64)
    scalar_cases = [
        (True, TypeError, "first_primary_id must be an integer"),
        (np.bool_(False), TypeError, "first_primary_id must be an integer"),
        (1.0, TypeError, "first_primary_id must be an integer"),
        (
            np.array(1, dtype=np.int64),
            TypeError,
            "first_primary_id must be an integer",
        ),
        (-1, ValueError, "first_primary_id must be non-negative"),
        (
            INT64_MAX + 1,
            OverflowError,
            "first_primary_id does not fit in int64",
        ),
        (
            faces.shape[0] + 1,
            ValueError,
            "first_primary_id exceeds block count",
        ),
    ]
    for first, error_type, message in scalar_cases:
        before = valid.copy()
        assert outcome(
            plan_level1_halo_closed_prefix, first, faces, valid
        ) == (error_type, message)
        assert np.array_equal(valid, before)

    overlap_faces = faces.copy()
    overlap = overlap_faces.reshape(-1)[:27]
    before = overlap_faces.copy()
    assert outcome(
        plan_level1_halo_closed_prefix, 0, overlap_faces, overlap
    ) == (
        ValueError,
        "selected_ids must not overlap face_neighbor_ids",
    )
    assert np.array_equal(overlap_faces, before)

    bad_output = np.full(1, -1, dtype=np.int32)
    assert outcome(
        plan_level1_halo_closed_prefix, True, faces, bad_output
    ) == (TypeError, "selected_ids must have dtype int64")
    assert outcome(
        plan_level1_halo_closed_prefix,
        faces.shape[0] + 1,
        faces,
        bad_output,
    ) == (TypeError, "selected_ids must have dtype int64")


def test_legacy_plan_and_minimum_wrappers_match_canonical_exactly() -> None:
    rng = np.random.default_rng(20260829)
    for _ in range(300):
        shape = tuple(int(value) for value in rng.integers(1, 6, size=3))
        _, _, faces = topology(shape)
        assert minimum_halo_closed_slots(faces) == (
            minimum_level1_halo_closed_slots(faces)
        )
        first = int(rng.integers(0, faces.shape[0] + 1))
        capacity = int(rng.integers(0 if first == faces.shape[0] else 1, 97))
        initial = rng.integers(
            INT64_MIN,
            INT64_MAX,
            size=capacity,
            dtype=np.int64,
        )
        canonical = initial.copy()
        legacy = initial.copy()
        canonical_outcome = outcome(
            plan_level1_halo_closed_prefix, first, faces, canonical
        )
        legacy_outcome = outcome(
            plan_level1_halo_chunk, first, faces, legacy
        )
        assert legacy_outcome == canonical_outcome
        assert np.array_equal(legacy, canonical)


def test_legacy_and_canonical_shared_vs_output_named_errors() -> None:
    coord_to_rank, _, faces = topology((3, 3, 3))
    center = int(coord_to_rank[1, 1, 1])
    canonical = np.full(26, -11, dtype=np.int64)
    legacy = canonical.copy()
    assert outcome(
        plan_level1_halo_closed_prefix, center, faces, canonical
    ) == outcome(plan_level1_halo_chunk, center, faces, legacy) == (
        ValueError,
        "chunk capacity cannot fit the first halo closure",
    )
    assert np.array_equal(canonical, np.full(26, -11, dtype=np.int64))
    assert np.array_equal(legacy, canonical)

    empty = np.empty(0, dtype=np.int64)
    assert outcome(plan_level1_halo_closed_prefix, 0, faces, empty) == (
        ValueError,
        "selected capacity must be positive before the end",
    )
    assert outcome(plan_level1_halo_chunk, 0, faces, empty) == (
        ValueError,
        "chunk capacity must be positive before the end",
    )

    bad_canonical = np.full(1, -1, dtype=np.int32)
    bad_legacy = bad_canonical.copy()
    assert outcome(
        plan_level1_halo_closed_prefix, 0, faces, bad_canonical
    ) == (TypeError, "selected_ids must have dtype int64")
    assert outcome(plan_level1_halo_chunk, 0, faces, bad_legacy) == (
        TypeError,
        "chunk_block_ids must have dtype int64",
    )


def test_hpl_sources_are_all_present_in_hcl_selected_prefix() -> None:
    coord_to_rank, _, faces = topology((5, 4, 3))
    first = int(coord_to_rank[2, 2, 1])
    selected_ids = np.empty(64, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_closed_prefix(
        first, faces, selected_ids
    )
    selected = selected_ids[:selected_count]
    actual = level1_halo_relation_plan(selected, primary_count, faces)
    expected = level1_halo_relation_plan_reference(
        selected, primary_count, faces
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.all(actual[0][actual[0] >= 0] < selected_count)


def test_one_block_reach_is_nonrecursive_and_refined_relation_tables_are_rejected() -> None:
    coord_to_rank, rank_to_coord, faces = topology((7, 7, 7))
    source_coord = (3, 3, 3)
    source = int(coord_to_rank[source_coord])
    selected = np.empty(27, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_closed_prefix(
        source, faces, selected
    )
    assert (primary_count, selected_count) == (1, 27)
    selected_coords = {
        tuple(int(value) for value in rank_to_coord[block])
        for block in selected
    }
    assert all(
        max(abs(coord[axis] - source_coord[axis]) for axis in range(3)) <= 1
        for coord in selected_coords
    )
    assert (5, 3, 3) not in selected_coords

    refined_style_relations = np.full((faces.shape[0], 27), -1, dtype=np.int64)
    assert outcome(
        plan_level1_halo_closed_prefix,
        0,
        refined_style_relations,
        selected,
    ) == (
        ValueError,
        "face_neighbor_ids must have shape (block_count, 6), "
        f"got {refined_style_relations.shape}",
    )
