from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from simesh_rewrite.field_line_termination import (
    FieldLineStage,
    FieldLineTermination,
    _classify_field_line_candidate_unchecked,
    _classify_field_line_stage_state_unchecked,
    classify_field_line_candidate,
    classify_field_line_seed,
    classify_field_line_stage_state,
    field_line_termination_from_rhs_status,
)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def f4(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


DOMAIN_LOWER = f3(-1.0, -2.0, -3.0)
DOMAIN_UPPER = f3(1.0, 2.0, 3.0)


def bits(value: np.ndarray) -> np.ndarray:
    return value.view(np.uint64).copy()


def test_stable_termination_and_stage_codes() -> None:
    assert [(name, int(value)) for name, value in FieldLineTermination.__members__.items()] == [
        ("ACTIVE", 0),
        ("SEED_OUTSIDE", 1),
        ("MAX_STEPS", 2),
        ("DOMAIN_EXIT", 3),
        ("ZERO_FIELD", 4),
        ("NONFINITE_FIELD", 5),
        ("UNREPRESENTABLE_NORM", 6),
        ("UNREPRESENTABLE_SAMPLE", 7),
        ("NONFINITE_STATE", 8),
        ("NO_PROGRESS", 9),
    ]
    assert [(name, int(value)) for name, value in FieldLineStage.__members__.items()] == [
        ("CONTROL", 0),
        ("K1", 1),
        ("K2", 2),
        ("K3", 3),
        ("K4", 4),
        ("CANDIDATE", 5),
    ]
    assert np.asarray(list(FieldLineTermination), dtype=np.uint8).tolist() == list(
        range(10)
    )
    assert np.asarray(list(FieldLineStage), dtype=np.uint8).tolist() == list(range(6))
    assert int(FieldLineTermination.MAX_STEPS) == 2
    assert int(FieldLineStage.CONTROL) == 0


def test_seed_exact_faces_and_half_open_membership_on_every_axis() -> None:
    center = f3(0.0, 0.0, 0.0)
    for axis in range(3):
        lower_face = center.copy()
        lower_face[axis] = DOMAIN_LOWER[axis]
        assert (
            classify_field_line_seed(lower_face, DOMAIN_LOWER, DOMAIN_UPPER)
            == FieldLineTermination.ACTIVE
        )

        upper_face = center.copy()
        upper_face[axis] = DOMAIN_UPPER[axis]
        assert (
            classify_field_line_seed(upper_face, DOMAIN_LOWER, DOMAIN_UPPER)
            == FieldLineTermination.SEED_OUTSIDE
        )

        just_inside_upper = center.copy()
        just_inside_upper[axis] = np.nextafter(
            DOMAIN_UPPER[axis], DOMAIN_LOWER[axis]
        )
        assert (
            classify_field_line_seed(
                just_inside_upper, DOMAIN_LOWER, DOMAIN_UPPER
            )
            == FieldLineTermination.ACTIVE
        )

        just_outside_lower = center.copy()
        just_outside_lower[axis] = np.nextafter(
            DOMAIN_LOWER[axis], -np.inf
        )
        assert (
            classify_field_line_seed(
                just_outside_lower, DOMAIN_LOWER, DOMAIN_UPPER
            )
            == FieldLineTermination.SEED_OUTSIDE
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis", range(3))
def test_nonfinite_seed_is_an_ordinary_error(bad: float, axis: int) -> None:
    seed = f3(0.0, 0.0, 0.0)
    seed[axis] = bad
    with pytest.raises(ValueError, match="seed coordinates must be finite"):
        classify_field_line_seed(seed, DOMAIN_LOWER, DOMAIN_UPPER)


def test_stage_checks_all_four_nonfinite_components_before_domain_exit() -> None:
    active = f4(0.0, 0.0, 0.0, 5.0)
    assert (
        classify_field_line_stage_state(active, DOMAIN_LOWER, DOMAIN_UPPER)
        == FieldLineTermination.ACTIVE
    )

    for component, bad in product(range(4), (np.nan, np.inf, -np.inf)):
        state = active.copy()
        state[component] = bad
        if component != 0:
            state[0] = DOMAIN_UPPER[0]
        assert (
            classify_field_line_stage_state(state, DOMAIN_LOWER, DOMAIN_UPPER)
            == FieldLineTermination.NONFINITE_STATE
        )


def test_finite_stage_domain_exit_faces_on_every_axis() -> None:
    for axis in range(3):
        for exterior in (
            np.nextafter(DOMAIN_LOWER[axis], -np.inf),
            DOMAIN_UPPER[axis],
            np.nextafter(DOMAIN_UPPER[axis], np.inf),
        ):
            state = f4(0.0, 0.0, 0.0, 7.0)
            state[axis] = exterior
            assert (
                classify_field_line_stage_state(
                    state, DOMAIN_LOWER, DOMAIN_UPPER
                )
                == FieldLineTermination.DOMAIN_EXIT
            )


@pytest.mark.parametrize(
    ("rhs_status", "expected"),
    [
        (0, FieldLineTermination.ACTIVE),
        (1, FieldLineTermination.ZERO_FIELD),
        (2, FieldLineTermination.NONFINITE_FIELD),
        (3, FieldLineTermination.UNREPRESENTABLE_NORM),
        (np.uint8(0), FieldLineTermination.ACTIVE),
        (np.uint8(1), FieldLineTermination.ZERO_FIELD),
        (np.uint8(2), FieldLineTermination.NONFINITE_FIELD),
        (np.uint8(3), FieldLineTermination.UNREPRESENTABLE_NORM),
    ],
)
def test_exact_rhs_status_mapping(rhs_status, expected: FieldLineTermination) -> None:
    assert field_line_termination_from_rhs_status(rhs_status) == expected


@pytest.mark.parametrize("unknown", [-1, 4, 7, 255, np.uint8(255)])
def test_unknown_rhs_status_is_rejected_and_sample_status_is_sle_owned(unknown) -> None:
    with pytest.raises(ValueError, match="known FLN-001 status"):
        field_line_termination_from_rhs_status(unknown)
    assert int(FieldLineTermination.UNREPRESENTABLE_SAMPLE) == 7


@pytest.mark.parametrize(
    "invalid",
    [True, False, 0.0, np.float64(0.0), np.int8(0), np.int64(0), np.array(0, dtype=np.uint8)],
)
def test_rhs_status_requires_exact_scalar_kind(invalid) -> None:
    with pytest.raises(TypeError, match="Python int or NumPy uint8"):
        field_line_termination_from_rhs_status(invalid)


def test_candidate_precedence_nonfinite_then_progress_then_domain() -> None:
    current = f4(0.0, 0.0, 0.0, 2.0)

    nonfinite = current.copy()
    nonfinite[0] = DOMAIN_UPPER[0]
    nonfinite[3] = np.inf
    assert (
        classify_field_line_candidate(
            current, nonfinite, DOMAIN_LOWER, DOMAIN_UPPER
        )
        == FieldLineTermination.NONFINITE_STATE
    )

    integral_only = current.copy()
    integral_only[3] = 3.0
    assert (
        classify_field_line_candidate(
            current, integral_only, DOMAIN_LOWER, DOMAIN_UPPER
        )
        == FieldLineTermination.NO_PROGRESS
    )

    exterior = current.copy()
    exterior[0] = DOMAIN_UPPER[0]
    assert (
        classify_field_line_candidate(
            current, exterior, DOMAIN_LOWER, DOMAIN_UPPER
        )
        == FieldLineTermination.DOMAIN_EXIT
    )

    progress = current.copy()
    progress[0] = np.nextafter(progress[0], 1.0)
    assert (
        classify_field_line_candidate(
            current, progress, DOMAIN_LOWER, DOMAIN_UPPER
        )
        == FieldLineTermination.ACTIVE
    )


def test_candidate_nonfinite_precedence_covers_every_component() -> None:
    current = f4(0.25, -0.5, 0.75, 1.0)
    for component, bad in product(range(4), (np.nan, np.inf, -np.inf)):
        candidate = current.copy()
        candidate[component] = bad
        assert (
            classify_field_line_candidate(
                current, candidate, DOMAIN_LOWER, DOMAIN_UPPER
            )
            == FieldLineTermination.NONFINITE_STATE
        )


def test_all_signed_zero_coordinate_patterns_use_bitwise_progress() -> None:
    lower = f3(-1.0, -1.0, -1.0)
    upper = f3(1.0, 1.0, 1.0)
    signed_zeros = (np.float64(+0.0), np.float64(-0.0))
    for current_signs in product(signed_zeros, repeat=3):
        current = np.asarray((*current_signs, 1.0), dtype=np.float64)
        for candidate_signs in product(signed_zeros, repeat=3):
            candidate = np.asarray((*candidate_signs, 9.0), dtype=np.float64)
            expected = (
                FieldLineTermination.NO_PROGRESS
                if np.array_equal(bits(current[:3]), bits(candidate[:3]))
                else FieldLineTermination.ACTIVE
            )
            assert (
                classify_field_line_candidate(current, candidate, lower, upper)
                == expected
            )


def test_candidate_exact_faces_and_current_precondition() -> None:
    current = f4(0.0, 0.0, 0.0, np.nan)
    candidate = f4(0.25, 0.0, 0.0, 1.0)
    assert (
        classify_field_line_candidate(
            current, candidate, DOMAIN_LOWER, DOMAIN_UPPER
        )
        == FieldLineTermination.ACTIVE
    )

    for axis in range(3):
        lower = candidate.copy()
        lower[axis] = DOMAIN_LOWER[axis]
        assert (
            classify_field_line_candidate(
                current, lower, DOMAIN_LOWER, DOMAIN_UPPER
            )
            == FieldLineTermination.ACTIVE
        )
        upper = candidate.copy()
        upper[axis] = DOMAIN_UPPER[axis]
        assert (
            classify_field_line_candidate(
                current, upper, DOMAIN_LOWER, DOMAIN_UPPER
            )
            == FieldLineTermination.DOMAIN_EXIT
        )

    for invalid in (
        f4(np.nan, 0.0, 0.0, 1.0),
        f4(DOMAIN_UPPER[0], 0.0, 0.0, 1.0),
        f4(np.nextafter(DOMAIN_LOWER[0], -np.inf), 0.0, 0.0, 1.0),
    ):
        with pytest.raises(ValueError, match="current_state coordinates"):
            classify_field_line_candidate(
                invalid, candidate, DOMAIN_LOWER, DOMAIN_UPPER
            )


@pytest.mark.parametrize(
    ("argument", "replacement", "error", "match"),
    [
        ("seed", [0.0, 0.0, 0.0], TypeError, "NumPy array"),
        ("seed", np.zeros(3, dtype=np.float32), TypeError, "float64"),
        ("seed", np.zeros((1, 3), dtype=np.float64), ValueError, "shape"),
        ("seed", np.zeros(6, dtype=np.float64)[::2], ValueError, "C-contiguous"),
        ("lower", np.zeros(3, dtype=np.float32), TypeError, "float64"),
        ("upper", f3(1.0, np.inf, 1.0), ValueError, "finite"),
        ("upper", f3(1.0, -2.0, 3.0), ValueError, "greater"),
    ],
)
def test_seed_representation_and_domain_validation(
    argument: str,
    replacement,
    error: type[Exception],
    match: str,
) -> None:
    values = {
        "seed": f3(0.0, 0.0, 0.0),
        "lower": DOMAIN_LOWER,
        "upper": DOMAIN_UPPER,
    }
    values[argument] = replacement
    with pytest.raises(error, match=match):
        classify_field_line_seed(values["seed"], values["lower"], values["upper"])


def test_stage_and_candidate_representation_validation() -> None:
    with pytest.raises(ValueError, match="stage_state must have shape"):
        classify_field_line_stage_state(
            np.zeros(3, dtype=np.float64), DOMAIN_LOWER, DOMAIN_UPPER
        )
    with pytest.raises(TypeError, match="candidate_state must have dtype float64"):
        classify_field_line_candidate(
            f4(0.0, 0.0, 0.0, 0.0),
            np.zeros(4, dtype=np.float32),
            DOMAIN_LOWER,
            DOMAIN_UPPER,
        )
    with pytest.raises(ValueError, match="current_state must be C-contiguous"):
        classify_field_line_candidate(
            np.zeros(8, dtype=np.float64)[::2],
            f4(0.1, 0.0, 0.0, 0.0),
            DOMAIN_LOWER,
            DOMAIN_UPPER,
        )


def test_classifiers_accept_read_only_inputs_and_never_mutate_them() -> None:
    seed = f3(0.0, 0.0, 0.0)
    stage = f4(0.0, 0.0, 0.0, 1.0)
    candidate = f4(0.25, 0.0, 0.0, 2.0)
    lower = DOMAIN_LOWER.copy()
    upper = DOMAIN_UPPER.copy()
    arrays = (seed, stage, candidate, lower, upper)
    before = tuple(bits(value) for value in arrays)
    for value in arrays:
        value.setflags(write=False)

    assert classify_field_line_seed(seed, lower, upper) == FieldLineTermination.ACTIVE
    assert (
        classify_field_line_stage_state(stage, lower, upper)
        == FieldLineTermination.ACTIVE
    )
    assert (
        classify_field_line_candidate(stage, candidate, lower, upper)
        == FieldLineTermination.ACTIVE
    )
    for value, expected in zip(arrays, before, strict=True):
        assert np.array_equal(bits(value), expected)


def test_prevalidated_stage_helper_matches_checked_truth_table() -> None:
    rows = [
        f4(0.0, 0.0, 0.0, 1.0),
        f4(DOMAIN_UPPER[0], 0.0, 0.0, 1.0),
        f4(0.0, DOMAIN_LOWER[1], 0.0, -3.0),
        f4(0.0, 0.0, 0.0, np.inf),
        f4(np.nan, DOMAIN_UPPER[1], 0.0, 1.0),
    ]
    for row in rows:
        assert _classify_field_line_stage_state_unchecked(
            row,
            DOMAIN_LOWER,
            DOMAIN_UPPER,
        ) == classify_field_line_stage_state(row, DOMAIN_LOWER, DOMAIN_UPPER)


def test_prevalidated_candidate_helper_matches_checked_truth_table() -> None:
    signed_zeros = (np.float64(+0.0), np.float64(-0.0))
    for current_signs in product(signed_zeros, repeat=3):
        current = np.asarray((*current_signs, 1.0), dtype=np.float64)
        candidates = [
            np.asarray((*candidate_signs, 9.0), dtype=np.float64)
            for candidate_signs in product(signed_zeros, repeat=3)
        ]
        candidates.extend(
            (
                f4(0.25, 0.0, 0.0, 2.0),
                f4(DOMAIN_UPPER[0], 0.0, 0.0, 2.0),
                f4(0.25, 0.0, 0.0, np.nan),
            )
        )
        for candidate in candidates:
            assert _classify_field_line_candidate_unchecked(
                current,
                candidate,
                DOMAIN_LOWER,
                DOMAIN_UPPER,
            ) == classify_field_line_candidate(
                current,
                candidate,
                DOMAIN_LOWER,
                DOMAIN_UPPER,
            )
