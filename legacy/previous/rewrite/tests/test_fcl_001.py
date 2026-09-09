from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.chunking import (
    minimum_face_closed_slots,
    plan_level1_chunk,
)
from simesh_rewrite.face_closure import (
    minimum_direct_face_closed_slots,
    plan_direct_face_closed_prefix,
)
from simesh_rewrite.face_closure_reference import (
    minimum_direct_face_closed_slots_reference,
    plan_direct_face_closed_prefix_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


INT64_MAX = int(np.iinfo(np.int64).max)
INT64_MIN = int(np.iinfo(np.int64).min)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def topology(shape: tuple[int, int, int]):
    root = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    return coord_to_rank, rank_to_coord, faces


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
            plan_direct_face_closed_prefix_reference(first, faces, capacity),
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
    selected = selected_ids[:selected_count]
    primaries = selected_ids[:primary_count]
    support = selected_ids[primary_count:selected_count]
    assert primaries.tolist() == list(range(first, first + primary_count))
    assert len(np.unique(selected)) == selected_count
    assert not set(int(value) for value in primaries) & set(
        int(value) for value in support
    )
    selected_set = set(int(value) for value in selected)
    expected_union = set(int(value) for value in primaries)
    for primary in primaries:
        expected_union.update(
            int(neighbor)
            for neighbor in faces[int(primary)]
            if neighbor >= 0
        )
    assert selected_set == expected_union
    assert all(
        any(support_id in faces[int(primary)] for primary in primaries)
        for support_id in support
    )


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
    for neighbor in faces[candidate]:
        neighbor = int(neighbor)
        if (
            neighbor >= 0
            and neighbor not in trial_primaries
            and neighbor not in trial_support
        ):
            trial_support.append(neighbor)
    return len(trial_primaries) + len(trial_support)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 1), 1),
        ((2, 1, 1), 2),
        ((2, 2, 1), 3),
        ((2, 2, 2), 4),
        ((3, 2, 2), 5),
        ((3, 3, 2), 6),
        ((3, 3, 3), 7),
    ],
)
def test_minimum_matches_exact_raw_face_count_and_reference(shape, expected) -> None:
    _, _, faces = topology(shape)
    actual = minimum_direct_face_closed_slots(faces)
    assert type(actual) is int
    assert actual == expected
    assert actual == minimum_direct_face_closed_slots_reference(faces)
    assert minimum_face_closed_slots(faces) == actual


def test_empty_minimum_and_end_plan_are_zero_without_mutation() -> None:
    faces = np.empty((0, 6), dtype=np.int64)
    assert minimum_direct_face_closed_slots(faces) == 0
    assert minimum_face_closed_slots(faces) == 0
    for capacity in (0, 1, 7):
        selected = np.full(capacity, INT64_MIN, dtype=np.int64)
        before = selected.copy()
        assert plan_direct_face_closed_prefix(0, faces, selected) == (0, 0)
        assert np.array_equal(selected, before)
        legacy = before.copy()
        assert plan_level1_chunk(0, faces, True, legacy) == (0, 0)
        assert np.array_equal(legacy, before)


def test_minimum_capacity_guarantees_progress_and_one_less_fails_at_argmax() -> None:
    for shape in ((1, 1, 1), (4, 1, 1), (4, 3, 1), (4, 3, 2), (4, 3, 3)):
        _, _, faces = topology(shape)
        minimum = minimum_direct_face_closed_slots(faces)
        selected = np.full(minimum, -71, dtype=np.int64)
        first = 0
        covered: list[int] = []
        while first < faces.shape[0]:
            primary_count, selected_count = plan_direct_face_closed_prefix(
                first, faces, selected
            )
            assert 0 < primary_count <= selected_count <= minimum
            covered.extend(selected[:primary_count].tolist())
            first += primary_count
        assert covered == list(range(faces.shape[0]))

        closure_sizes = 1 + np.count_nonzero(faces >= 0, axis=1)
        argmax = int(np.argmax(closure_sizes))
        too_small = np.full(max(0, minimum - 1), -73, dtype=np.int64)
        before = too_small.copy()
        expected_message = (
            "selected capacity must be positive before the end"
            if minimum == 1
            else "chunk capacity cannot fit the first primary closure"
        )
        with pytest.raises(ValueError, match=f"^{expected_message}$"):
            plan_direct_face_closed_prefix(argmax, faces, too_small)
        assert np.array_equal(too_small, before)


def test_golden_face_order_has_no_edge_or_corner_support() -> None:
    coord_to_rank, _, faces = topology((5, 5, 5))
    center_coord = (2, 2, 2)
    center = int(coord_to_rank[center_coord])
    selected = np.full(7, -91, dtype=np.int64)
    primary_count, selected_count = plan_direct_face_closed_prefix(
        center, faces, selected
    )
    assert (primary_count, selected_count) == (1, 7)
    expected_faces = [int(value) for value in faces[center] if value >= 0]
    assert selected.tolist() == [center, *expected_faces]

    all_neighbors = {
        int(coord_to_rank[
            center_coord[0] + dx,
            center_coord[1] + dy,
            center_coord[2] + dz,
        ])
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    }
    direct_faces = set(expected_faces)
    assert len(all_neighbors) == 26
    assert len(direct_faces) == 6
    assert not (all_neighbors - direct_faces) & set(selected.tolist())


def test_support_promotion_preserves_remaining_first_discovery_order() -> None:
    _, _, faces = topology((5, 1, 1))
    selected = np.full(4, -1, dtype=np.int64)
    primary_count, selected_count = plan_direct_face_closed_prefix(
        0, faces, selected
    )
    assert (primary_count, selected_count) == (3, 4)
    assert selected.tolist() == [0, 1, 2, 3]

    coord_to_rank, _, faces = topology((4, 3, 2))
    first = int(coord_to_rank[1, 1, 0])
    selected = np.full(12, -77, dtype=np.int64)
    before = selected.copy()
    expected, expected_primary, expected_selected = (
        plan_direct_face_closed_prefix_reference(first, faces, selected.size)
    )
    actual = plan_direct_face_closed_prefix(first, faces, selected)
    assert actual == (expected_primary, expected_selected)
    assert selected[:expected_selected].tolist() == expected
    assert np.array_equal(selected[expected_selected:], before[expected_selected:])


@pytest.mark.parametrize("capacity", [4, 5, 7, 10, 16, 31])
def test_reference_exactness_maximality_uniqueness_and_suffix(capacity) -> None:
    _, _, faces = topology((4, 3, 2))
    first = 0
    selected = np.asarray(
        [INT64_MIN + index for index in range(capacity)], dtype=np.int64
    )
    before = selected.copy()
    expected = reference_outcome(first, faces, capacity)
    actual = outcome(plan_direct_face_closed_prefix, first, faces, selected)
    if expected[0] != "return":
        assert actual == expected
        assert np.array_equal(selected, before)
        return
    values, expected_primary, expected_selected = expected[1]
    assert actual == ("return", (expected_primary, expected_selected))
    assert selected[:expected_selected].tolist() == values
    assert np.array_equal(selected[expected_selected:], before[expected_selected:])
    assert_plan_invariants(
        first,
        faces,
        selected,
        expected_primary,
        expected_selected,
    )
    if first + expected_primary < faces.shape[0]:
        assert next_trial_size(
            first,
            faces,
            selected,
            expected_primary,
            expected_selected,
        ) > capacity


def test_end_and_first_fit_failures_are_fully_atomic() -> None:
    coord_to_rank, _, faces = topology((3, 3, 3))
    end = np.asarray([INT64_MIN, -1, INT64_MAX], dtype=np.int64)
    before = end.copy()
    assert plan_direct_face_closed_prefix(faces.shape[0], faces, end) == (0, 0)
    assert np.array_equal(end, before)

    empty = np.empty(0, dtype=np.int64)
    with pytest.raises(
        ValueError,
        match="^selected capacity must be positive before the end$",
    ):
        plan_direct_face_closed_prefix(0, faces, empty)

    center = int(coord_to_rank[1, 1, 1])
    too_small = np.full(6, -83, dtype=np.int64)
    before = too_small.copy()
    with pytest.raises(
        ValueError,
        match="^chunk capacity cannot fit the first primary closure$",
    ):
        plan_direct_face_closed_prefix(center, faces, too_small)
    assert np.array_equal(too_small, before)


def compare_one_case(first: int, faces: np.ndarray, capacity: int) -> None:
    initial = np.asarray(
        [INT64_MIN + index for index in range(capacity)], dtype=np.int64
    )
    selected = initial.copy()
    expected = reference_outcome(first, faces, capacity)
    actual = outcome(plan_direct_face_closed_prefix, first, faces, selected)
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
                maximum_capacity = min(faces.shape[0] + 6, 14)
                for first in range(faces.shape[0] + 1):
                    for capacity in range(maximum_capacity + 1):
                        compare_one_case(first, faces, capacity)


def test_random_topologies_and_full_traversals_match_reference() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 7, size=3))
        _, _, faces = topology(shape)
        minimum = minimum_direct_face_closed_slots(faces)
        capacity = int(rng.integers(minimum, max(minimum + 1, 65)))
        selected = np.full(capacity, -97, dtype=np.int64)
        first = 0
        covered: list[int] = []
        while first < faces.shape[0]:
            before = selected.copy()
            expected, expected_primary, expected_selected = (
                plan_direct_face_closed_prefix_reference(
                    first, faces, capacity
                )
            )
            actual = plan_direct_face_closed_prefix(first, faces, selected)
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


def test_canonical_face_table_validation_messages_are_exact() -> None:
    noncontiguous = np.empty((4, 12), dtype=np.int64)[:, ::2]
    nonnative_dtype = np.dtype(">i8" if np.little_endian else "<i8")
    cases = [
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
            np.full((1, 6), -1, dtype=nonnative_dtype),
            TypeError,
            "face_neighbor_ids must have dtype int64",
        ),
        (
            np.full((1, 5), -1, dtype=np.int64),
            ValueError,
            "face_neighbor_ids must have shape (block_count, 6), got (1, 5)",
        ),
        (
            noncontiguous,
            ValueError,
            "face_neighbor_ids must be C-contiguous",
        ),
    ]
    for faces, error_type, message in cases:
        selected = np.full(1, -1, dtype=np.int64)
        assert outcome(plan_direct_face_closed_prefix, 0, faces, selected) == (
            error_type,
            message,
        )
        assert outcome(minimum_direct_face_closed_slots, faces) == (
            error_type,
            message,
        )


def test_canonical_output_scalar_overlap_and_precedence_messages_are_exact() -> None:
    _, _, faces = topology((2, 2, 2))
    readonly = np.full(4, -1, dtype=np.int64)
    readonly.setflags(write=False)
    output_cases = [
        (
            [1, 2],
            TypeError,
            "selected_ids must be a NumPy array",
        ),
        (
            np.full(4, -1, dtype=np.int32),
            TypeError,
            "selected_ids must have dtype int64",
        ),
        (
            np.full((1, 4), -1, dtype=np.int64),
            ValueError,
            "selected_ids must be a C-contiguous vector",
        ),
        (
            np.arange(8, dtype=np.int64)[::2],
            ValueError,
            "selected_ids must be a C-contiguous vector",
        ),
        (readonly, ValueError, "selected_ids must be writable"),
    ]
    for selected, error_type, message in output_cases:
        before = selected.copy() if isinstance(selected, np.ndarray) else list(selected)
        assert outcome(plan_direct_face_closed_prefix, 0, faces, selected) == (
            error_type,
            message,
        )
        if isinstance(selected, np.ndarray):
            assert np.array_equal(selected, before)
        else:
            assert selected == before

    valid = np.full(7, -1, dtype=np.int64)
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
        assert outcome(plan_direct_face_closed_prefix, first, faces, valid) == (
            error_type,
            message,
        )
        assert np.array_equal(valid, before)

    overlap_faces = faces.copy()
    overlap = overlap_faces.reshape(-1)[:7]
    before = overlap_faces.copy()
    assert outcome(
        plan_direct_face_closed_prefix, 0, overlap_faces, overlap
    ) == (
        ValueError,
        "selected_ids must not overlap face_neighbor_ids",
    )
    assert np.array_equal(overlap_faces, before)

    bad_output = np.full(1, -1, dtype=np.int32)
    assert outcome(
        plan_direct_face_closed_prefix, True, faces, bad_output
    ) == (
        TypeError,
        "selected_ids must have dtype int64",
    )
    assert outcome(
        plan_direct_face_closed_prefix,
        faces.shape[0] + 1,
        faces,
        bad_output,
    ) == (
        TypeError,
        "selected_ids must have dtype int64",
    )


def test_legacy_true_and_minimum_wrappers_match_canonical_exactly() -> None:
    rng = np.random.default_rng(20260829)
    for _ in range(300):
        shape = tuple(int(value) for value in rng.integers(1, 6, size=3))
        _, _, faces = topology(shape)
        assert minimum_face_closed_slots(faces) == (
            minimum_direct_face_closed_slots(faces)
        )
        first = int(rng.integers(0, faces.shape[0] + 1))
        capacity = int(rng.integers(0 if first == faces.shape[0] else 1, 65))
        initial = rng.integers(
            INT64_MIN,
            INT64_MAX,
            size=capacity,
            dtype=np.int64,
        )
        canonical = initial.copy()
        legacy = initial.copy()
        canonical_outcome = outcome(
            plan_direct_face_closed_prefix, first, faces, canonical
        )
        legacy_outcome = outcome(
            plan_level1_chunk, first, faces, True, legacy
        )
        assert legacy_outcome == canonical_outcome
        assert np.array_equal(legacy, canonical)


def test_legacy_wrapper_preserves_bool_validation_precedence() -> None:
    _, _, faces = topology((2, 2, 2))
    valid = np.full(7, -1, dtype=np.int64)
    bad_output = np.full(7, -1, dtype=np.int32)
    bad_faces = [[-1] * 6]
    cases = [
        (
            (0, bad_faces, np.bool_(True), bad_output),
            (TypeError, "face_neighbor_ids must be a NumPy array"),
        ),
        (
            (0, faces, np.bool_(True), bad_output),
            (TypeError, "chunk_block_ids must have dtype int64"),
        ),
        (
            (True, faces, np.bool_(True), valid),
            (TypeError, "first_primary_id must be an integer"),
        ),
        (
            (faces.shape[0] + 1, faces, np.bool_(True), valid),
            (ValueError, "first_primary_id exceeds block count"),
        ),
        (
            (0, faces, np.bool_(True), valid),
            (TypeError, "include_face_closure must be a bool"),
        ),
    ]
    for arguments, expected in cases:
        output = arguments[-1]
        before = output.copy() if isinstance(output, np.ndarray) else list(output)
        assert outcome(plan_level1_chunk, *arguments) == expected
        if isinstance(output, np.ndarray):
            assert np.array_equal(output, before)

    overlap_faces = faces.copy()
    overlap = overlap_faces.reshape(-1)[:7]
    assert outcome(
        plan_level1_chunk,
        0,
        overlap_faces,
        np.bool_(True),
        overlap,
    ) == (TypeError, "include_face_closure must be a bool")


def test_legacy_and_canonical_shared_errors_retain_each_output_name() -> None:
    coord_to_rank, _, faces = topology((3, 3, 3))
    center = int(coord_to_rank[1, 1, 1])
    too_small_new = np.full(6, -11, dtype=np.int64)
    too_small_old = too_small_new.copy()
    new = outcome(
        plan_direct_face_closed_prefix, center, faces, too_small_new
    )
    old = outcome(plan_level1_chunk, center, faces, True, too_small_old)
    assert new == old == (
        ValueError,
        "chunk capacity cannot fit the first primary closure",
    )
    assert np.array_equal(too_small_new, np.full(6, -11, dtype=np.int64))
    assert np.array_equal(too_small_old, too_small_new)

    empty = np.empty(0, dtype=np.int64)
    assert outcome(plan_direct_face_closed_prefix, 0, faces, empty) == (
        ValueError,
        "selected capacity must be positive before the end",
    )
    assert outcome(plan_level1_chunk, 0, faces, True, empty) == (
        ValueError,
        "chunk capacity must be positive before the end",
    )
