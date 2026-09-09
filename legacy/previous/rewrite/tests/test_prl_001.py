from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite._prolongation import (
    prolong_cartesian_2to1_into_fine_centric_unchecked,
)
from simesh_rewrite.limiter_reference import three_point_limited_slope_reference
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.prolongation import prolong_cartesian_2to1_into
from simesh_rewrite.prolongation_reference import (
    prolong_cartesian_2to1_reference,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def test_asymmetric_origins_negative_offsets_and_all_phases() -> None:
    x, y, z = np.indices((12, 12, 12), dtype=np.float64)
    coarse = (100.0 * x + 10.0 * y + z).reshape(1, 1, 12, 12, 12)
    coarse.setflags(write=False)
    fine = np.full((1, 1, 10, 11, 10), -91.0)
    expected = fine.copy()
    coarse_origin = i3(5, 4, 6)
    fine_origin = i3(3, 5, 2)
    fine_lower = i3(0, 1, 0)
    fine_upper = i3(9, 10, 9)
    prolong_cartesian_2to1_reference(
        coarse,
        i3(0, 0, 0),
        i3(12, 12, 12),
        coarse_origin,
        expected,
        fine_lower,
        fine_upper,
        fine_origin,
    )
    prolong_cartesian_2to1_into(
        coarse,
        i3(0, 0, 0),
        i3(12, 12, 12),
        coarse_origin,
        fine,
        fine_lower,
        fine_upper,
        fine_origin,
    )
    assert_bits_equal(fine, expected)

    phases: list[tuple[int, int, int]] = []
    for i in range(fine_lower[0], fine_upper[0]):
        rx = int(i - fine_origin[0])
        qx = rx // 2
        px = rx - 2 * qx
        I = int(coarse_origin[0]) + qx
        eta_x = np.float64(-0.25 if px == 0 else 0.25)
        for j in range(fine_lower[1], fine_upper[1]):
            ry = int(j - fine_origin[1])
            qy = ry // 2
            py = ry - 2 * qy
            J = int(coarse_origin[1]) + qy
            eta_y = np.float64(-0.25 if py == 0 else 0.25)
            for k in range(fine_lower[2], fine_upper[2]):
                rz = int(k - fine_origin[2])
                qz = rz // 2
                pz = rz - 2 * qz
                K = int(coarse_origin[2]) + qz
                eta_z = np.float64(-0.25 if pz == 0 else 0.25)
                phase = (px, py, pz)
                if phase not in phases:
                    phases.append(phase)
                center = coarse[0, 0, I, J, K]
                value = np.float64(center + np.float64(100.0 * eta_x))
                value = np.float64(value + np.float64(10.0 * eta_y))
                value = np.float64(value + np.float64(eta_z))
                assert bits(fine[0, 0, i, j, k]) == bits(value)
    assert sorted(phases) == [
        (px, py, pz)
        for px in (0, 1)
        for py in (0, 1)
        for pz in (0, 1)
    ]
    assert np.all(fine[:, :, 9] == -91.0)


def test_tight_six_face_reach_and_diagonal_poison() -> None:
    coarse = np.full((1, 1, 5, 5, 5), np.nan)
    center = (2, 2, 2)
    values = (
        (center, 10.0),
        ((1, 2, 2), 8.0),
        ((3, 2, 2), 12.0),
        ((2, 1, 2), 7.0),
        ((2, 3, 2), 13.0),
        ((2, 2, 1), 6.0),
        ((2, 2, 3), 14.0),
    )
    for coordinate, value in values:
        coarse[(0, 0, *coordinate)] = value
    fine = np.full((1, 1, 2, 2, 2), -7.0)
    expected = fine.copy()
    args = (
        coarse,
        i3(1, 1, 1),
        i3(4, 4, 4),
        i3(2, 2, 2),
    )
    prolong_cartesian_2to1_reference(
        *args, expected, i3(0, 0, 0), i3(2, 2, 2), i3(0, 0, 0)
    )
    prolong_cartesian_2to1_into(
        *args, fine, i3(0, 0, 0), i3(2, 2, 2), i3(0, 0, 0)
    )
    assert np.all(np.isfinite(fine))
    assert_bits_equal(fine, expected)

    for axis in range(3):
        for side in (0, 1):
            valid_lower = i3(1, 1, 1)
            valid_upper = i3(4, 4, 4)
            if side == 0:
                valid_lower[axis] += 1
            else:
                valid_upper[axis] -= 1
            output = np.full((1, 1, 2, 2, 2), -17.0)
            before = output.copy()
            with pytest.raises(ValueError, match="required reach"):
                prolong_cartesian_2to1_into(
                    coarse,
                    valid_lower,
                    valid_upper,
                    i3(2, 2, 2),
                    output,
                    i3(0, 0, 0),
                    i3(2, 2, 2),
                    i3(0, 0, 0),
                )
            assert_bits_equal(output, before)


def test_constant_affine_and_eight_child_average() -> None:
    x, y, z = np.indices((6, 6, 6), dtype=np.float64)
    coarse = np.empty((1, 2, 6, 6, 6), dtype=np.float64)
    coarse[0, 0].fill(5.0)
    coarse[0, 1] = 3.0 + 2.0 * x - 4.0 * y + 0.5 * z
    fine = np.empty((1, 2, 8, 8, 8), dtype=np.float64)
    prolong_cartesian_2to1_into(
        coarse,
        i3(0, 0, 0),
        i3(6, 6, 6),
        i3(1, 1, 1),
        fine,
        i3(0, 0, 0),
        i3(8, 8, 8),
        i3(0, 0, 0),
    )
    assert np.all(fine[0, 0] == 5.0)
    for i, j, k in np.ndindex(8, 8, 8):
        I, J, K = 1 + i // 2, 1 + j // 2, 1 + k // 2
        eta_x = -0.25 if i % 2 == 0 else 0.25
        eta_y = -0.25 if j % 2 == 0 else 0.25
        eta_z = -0.25 if k % 2 == 0 else 0.25
        expected = np.float64(coarse[0, 1, I, J, K] + 2.0 * eta_x)
        expected = np.float64(expected - 4.0 * eta_y)
        expected = np.float64(expected + 0.5 * eta_z)
        assert bits(fine[0, 1, i, j, k]) == bits(expected)

    recovered = np.empty((1, 2, 4, 4, 4), dtype=np.float64)
    restrict_cartesian_2to1_into(
        fine,
        i3(0, 0, 0),
        i3(8, 8, 8),
        recovered,
        i3(0, 0, 0),
    )
    assert_bits_equal(recovered, coarse[:, :, 1:5, 1:5, 1:5])


def test_step_and_quadratic_fields_do_not_create_new_extrema() -> None:
    x = np.arange(6, dtype=np.float64)[:, None, None]
    coarse = np.empty((1, 2, 6, 6, 6), dtype=np.float64)
    coarse[0, 0] = np.broadcast_to(np.where(x < 3, 0.0, 10.0), (6, 6, 6))
    coarse[0, 1] = np.broadcast_to(x**2, (6, 6, 6))
    fine = np.empty((1, 2, 8, 8, 8), dtype=np.float64)
    expected = np.empty_like(fine)
    args = (
        coarse,
        i3(0, 0, 0),
        i3(6, 6, 6),
        i3(1, 1, 1),
    )
    prolong_cartesian_2to1_reference(
        *args, expected, i3(0, 0, 0), i3(8, 8, 8), i3(0, 0, 0)
    )
    prolong_cartesian_2to1_into(
        *args, fine, i3(0, 0, 0), i3(8, 8, 8), i3(0, 0, 0)
    )
    assert_bits_equal(fine, expected)
    assert np.all((fine[0, 0] >= 0.0) & (fine[0, 0] <= 10.0))
    for i in range(8):
        I = 1 + i // 2
        lower = min(coarse[0, 1, I - 1, 0, 0], coarse[0, 1, I, 0, 0])
        upper = max(coarse[0, 1, I, 0, 0], coarse[0, 1, I + 1, 0, 0])
        assert np.all(fine[0, 1, i] >= lower)
        assert np.all(fine[0, 1, i] <= upper)


def test_partial_translated_random_multislot_field_matches_reference() -> None:
    rng = np.random.default_rng(20260831)
    coarse = np.ascontiguousarray(rng.normal(size=(2, 3, 9, 10, 11)))
    coarse_before = coarse.copy()
    fine = np.full((2, 3, 8, 9, 10), -29.0)
    expected = fine.copy()
    fine_centric = fine.copy()
    arguments = (
        coarse,
        i3(1, 1, 1),
        i3(8, 9, 10),
        i3(4, 5, 4),
    )
    tail = (i3(1, 1, 2), i3(8, 8, 9), i3(3, 2, 4))
    prolong_cartesian_2to1_reference(*arguments, expected, *tail)
    prolong_cartesian_2to1_into(*arguments, fine, *tail)
    prolong_cartesian_2to1_into_fine_centric_unchecked(
        coarse,
        arguments[3],
        fine_centric,
        *tail,
    )
    assert_bits_equal(fine, expected)
    assert_bits_equal(fine, fine_centric)
    assert_bits_equal(coarse, coarse_before)
    target = np.zeros(fine.shape, dtype=bool)
    target[:, :, 1:8, 1:8, 2:9] = True
    assert np.all(fine[~target] == -29.0)


def test_signed_zero_and_nonfinite_reconstruction_order() -> None:
    coarse = np.full((1, 1, 5, 5, 5), -0.0)
    fine = np.empty((1, 1, 2, 2, 2), dtype=np.float64)
    expected = np.empty_like(fine)
    args = (
        coarse,
        i3(1, 1, 1),
        i3(4, 4, 4),
        i3(2, 2, 2),
    )
    prolong_cartesian_2to1_reference(
        *args, expected, i3(0, 0, 0), i3(2, 2, 2), i3(0, 0, 0)
    )
    prolong_cartesian_2to1_into(
        *args, fine, i3(0, 0, 0), i3(2, 2, 2), i3(0, 0, 0)
    )
    assert_bits_equal(fine, expected)
    assert bits(fine[0, 0, 0, 0, 0]) == 0x8000000000000000
    assert np.all(fine.view(np.uint64).ravel()[1:] == 0)

    finite_center = np.zeros((1, 1, 5, 5, 5), dtype=np.float64)
    finite_center[0, 0, 2, 2, 2] = 3.0
    finite_center[0, 0, 1, 2, 2] = np.nan
    output = np.empty((1, 1, 1, 1, 1), dtype=np.float64)
    with np.errstate(all="ignore"):
        prolong_cartesian_2to1_into(
            finite_center,
            i3(1, 1, 1),
            i3(4, 4, 4),
            i3(2, 2, 2),
            output,
            i3(0, 0, 0),
            i3(1, 1, 1),
            i3(0, 0, 0),
        )
    assert output[0, 0, 0, 0, 0] == 3.0

    nan_center = finite_center.copy()
    nan_center[0, 0, 2, 2, 2] = np.nan
    with np.errstate(all="ignore"):
        prolong_cartesian_2to1_into(
            nan_center,
            i3(1, 1, 1),
            i3(4, 4, 4),
            i3(2, 2, 2),
            output,
            i3(0, 0, 0),
            i3(1, 1, 1),
            i3(0, 0, 0),
        )
    assert np.isnan(output[0, 0, 0, 0, 0])

    opposing = np.zeros((1, 1, 5, 5, 5), dtype=np.float64)
    opposing[0, 0, 1, 2, 2] = -np.inf
    opposing[0, 0, 3, 2, 2] = np.inf
    opposing[0, 0, 2, 1, 2] = np.inf
    opposing[0, 0, 2, 3, 2] = -np.inf
    with np.errstate(all="ignore"):
        prolong_cartesian_2to1_into(
            opposing,
            i3(1, 1, 1),
            i3(4, 4, 4),
            i3(2, 2, 2),
            output,
            i3(0, 0, 0),
            i3(1, 1, 1),
            i3(0, 0, 0),
        )
    assert np.isnan(output[0, 0, 0, 0, 0])


def test_empty_read_only_and_exterior_preservation() -> None:
    coarse = np.arange(216, dtype=np.float64).reshape(1, 1, 6, 6, 6)
    coarse.setflags(write=False)
    fine = np.full((1, 1, 6, 6, 6), -37.0)
    expected = fine.copy()
    args = (
        coarse,
        i3(0, 0, 0),
        i3(6, 6, 6),
        i3(2, 2, 2),
    )
    tail = (i3(2, 1, 2), i3(4, 5, 4), i3(2, 2, 2))
    prolong_cartesian_2to1_reference(*args, expected, *tail)
    prolong_cartesian_2to1_into(*args, fine, *tail)
    assert_bits_equal(fine, expected)
    assert np.all(fine[:, :, :2] == -37.0)

    for empty_coarse, empty_fine in (
        (
            np.empty((0, 2, 3, 3, 3), dtype=np.float64),
            np.empty((0, 2, 2, 2, 2), dtype=np.float64),
        ),
        (
            np.empty((2, 0, 3, 3, 3), dtype=np.float64),
            np.empty((2, 0, 2, 2, 2), dtype=np.float64),
        ),
    ):
        prolong_cartesian_2to1_into(
            empty_coarse,
            i3(0, 0, 0),
            i3(3, 3, 3),
            i3(1, 1, 1),
            empty_fine,
            i3(0, 0, 0),
            i3(2, 2, 2),
            i3(0, 0, 0),
        )

    spatial_empty = np.full((1, 1, 2, 2, 2), -41.0)
    before = spatial_empty.copy()
    prolong_cartesian_2to1_into(
        np.empty((1, 1, 1, 1, 1), dtype=np.float64),
        i3(0, 0, 0),
        i3(0, 0, 0),
        i3(np.iinfo(np.int64).max, 0, 0),
        spatial_empty,
        i3(1, 0, 0),
        i3(1, 2, 2),
        i3(np.iinfo(np.int64).min, 0, 0),
    )
    assert_bits_equal(spatial_empty, before)


def assert_atomic_error(error, operation, destination: np.ndarray) -> None:
    before = destination.copy()
    with pytest.raises(error):
        operation()
    assert_bits_equal(destination, before)


def test_validation_overflow_and_overlap_are_atomic() -> None:
    coarse = np.ones((1, 1, 5, 5, 5), dtype=np.float64)
    valid_lower = i3(1, 1, 1)
    valid_upper = i3(4, 4, 4)
    coarse_origin = i3(2, 2, 2)
    fine_lower = i3(0, 0, 0)
    fine_upper = i3(2, 2, 2)
    fine_origin = i3(0, 0, 0)

    def check(
        error,
        coarse_value=coarse,
        cvl=valid_lower,
        cvu=valid_upper,
        co=coarse_origin,
        fine_value=None,
        fl=fine_lower,
        fu=fine_upper,
        fo=fine_origin,
    ):
        output = (
            np.full((1, 1, 2, 2, 2), -51.0)
            if fine_value is None
            else fine_value
        )
        assert_atomic_error(
            error,
            lambda: prolong_cartesian_2to1_into(
                coarse_value, cvl, cvu, co, output, fl, fu, fo
            ),
            output,
        )

    check(TypeError, coarse_value=coarse.astype(np.float32))
    check(ValueError, coarse_value=coarse[0])
    check(ValueError, coarse_value=np.asfortranarray(coarse))
    check(TypeError, cvl=valid_lower.astype(np.int32))
    check(ValueError, cvl=i3(1, 1))
    check(TypeError, cvu=valid_upper.astype(np.int32))
    check(TypeError, co=coarse_origin.astype(np.int32))
    check(TypeError, fine_value=np.full((1, 1, 2, 2, 2), -51, dtype=np.float32))
    check(ValueError, fine_value=np.full((1, 2, 2, 2), -51.0))
    noncontiguous = np.full((1, 1, 2, 2, 4), -51.0)[..., ::2]
    check(ValueError, fine_value=noncontiguous)
    readonly = np.full((1, 1, 2, 2, 2), -51.0)
    readonly.setflags(write=False)
    check(ValueError, fine_value=readonly)
    check(TypeError, fl=fine_lower.astype(np.int32))
    check(TypeError, fu=fine_upper.astype(np.int32))
    check(TypeError, fo=fine_origin.astype(np.int32))
    check(ValueError, fine_value=np.full((2, 1, 2, 2, 2), -51.0))
    check(ValueError, cvl=i3(-1, 1, 1))
    check(ValueError, cvl=i3(4, 1, 1), cvu=i3(3, 4, 4))
    check(ValueError, cvu=i3(6, 4, 4))
    check(ValueError, fl=i3(-1, 0, 0))
    check(ValueError, fl=i3(2, 0, 0), fu=i3(1, 2, 2))
    check(ValueError, fu=i3(3, 2, 2))
    check(ValueError, cvl=i3(2, 1, 1))
    check(ValueError, cvu=i3(3, 4, 4))
    check(OverflowError, fo=i3(np.iinfo(np.int64).min, 0, 0))
    check(OverflowError, co=i3(np.iinfo(np.int64).max, 2, 2))

    shared = np.ones((1, 1, 5, 5, 5), dtype=np.float64)
    assert_atomic_error(
        ValueError,
        lambda: prolong_cartesian_2to1_into(
            shared,
            valid_lower,
            valid_upper,
            coarse_origin,
            shared,
            fine_lower,
            fine_upper,
            fine_origin,
        ),
        shared,
    )
    for metadata_position in range(7):
        base = np.full(8, -53.0)
        output = base.reshape(1, 1, 2, 2, 2)
        metadata = base.view(np.int64)[:3]
        metadata[:] = 0
        values = [
            valid_lower,
            valid_upper,
            coarse_origin,
            fine_lower,
            fine_upper,
            fine_origin,
            metadata,
        ]
        values[metadata_position if metadata_position < 6 else 5] = metadata
        assert_atomic_error(
            ValueError,
            lambda values=values, output=output: prolong_cartesian_2to1_into(
                coarse,
                values[0],
                values[1],
                values[2],
                output,
                values[3],
                values[4],
                values[5],
            ),
            output,
        )


def test_rst_assembled_coarse_workspace_reproduces_affine_fine_values() -> None:
    x, y, z = np.indices((8, 8, 8), dtype=np.float64)
    fine_source = (
        1.0 + 2.0 * (x + 0.5) + 4.0 * (y + 0.5) + 8.0 * (z + 0.5)
    ).reshape(1, 1, 8, 8, 8)
    coarse_workspace = np.empty((1, 1, 6, 6, 6), dtype=np.float64)
    I, J, K = np.indices((6, 6, 6), dtype=np.float64)
    qi, qj, qk = I - 1.0, J - 1.0, K - 1.0
    coarse_workspace[0, 0] = (
        1.0 + 2.0 * (2.0 * qi + 1.0) + 4.0 * (2.0 * qj + 1.0) + 8.0 * (2.0 * qk + 1.0)
    )
    restrict_cartesian_2to1_into(
        fine_source,
        i3(0, 0, 0),
        i3(8, 8, 8),
        coarse_workspace,
        i3(1, 1, 1),
    )
    reconstructed = np.empty_like(fine_source)
    prolong_cartesian_2to1_into(
        coarse_workspace,
        i3(0, 0, 0),
        i3(6, 6, 6),
        i3(1, 1, 1),
        reconstructed,
        i3(0, 0, 0),
        i3(8, 8, 8),
        i3(0, 0, 0),
    )
    assert_bits_equal(reconstructed, fine_source)


def test_slot_order_substitution_is_exact() -> None:
    rng = np.random.default_rng(20260901)
    coarse = np.ascontiguousarray(rng.normal(size=(3, 2, 6, 6, 6)))
    baseline = np.empty((3, 2, 8, 8, 8), dtype=np.float64)
    args = (i3(0, 0, 0), i3(6, 6, 6), i3(1, 1, 1))
    tail = (i3(0, 0, 0), i3(8, 8, 8), i3(0, 0, 0))
    prolong_cartesian_2to1_into(coarse, *args, baseline, *tail)
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    permuted_output = np.empty_like(baseline)
    prolong_cartesian_2to1_into(
        np.ascontiguousarray(coarse[permutation]),
        *args,
        permuted_output,
        *tail,
    )
    inverse = np.argsort(permutation)
    assert_bits_equal(permuted_output[inverse], baseline)


def current_prolongation_fixture(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
):
    root = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        if tuple(int(value) for value in coordinate) == (1, 1, 1):
            flags.append(False)
            flags.extend([True] * 8)
        else:
            flags.append(True)
    forest = AMRForest(3, 4, 4, 4, np.asarray(flags, dtype=np.int32))
    block = np.asarray([4, 4, 4], dtype=np.uint32)
    mesh = AMRMesh(
        3,
        block,
        np.asarray([16, 16, 16], dtype=np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(2),
        np.uint32(1),
        forest,
    )
    rng = np.random.default_rng(20260902)
    interior = np.ascontiguousarray(rng.normal(size=(forest.nleafs, 1, 4, 4, 4)))
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()
    neighbor_type = np.asarray(forest.neighbor_type)
    candidates = np.flatnonzero(neighbor_type[:, 14] == 2)
    assert candidates.size > 0
    leaf = int(candidates[0])
    coarse = np.asarray(mesh.datac)[leaf].transpose(3, 0, 1, 2)[None].copy()
    current = mesh.padded_view()[leaf].transpose(3, 0, 1, 2)[None].copy()
    rnode = np.asarray(mesh.rnode)[leaf].copy()
    return coarse, current, rnode


def test_current_zero_origin_dyadic_face_is_bitwise_equal() -> None:
    coarse, current, _ = current_prolongation_fixture(
        np.zeros(3), np.ones(3)
    )
    actual = np.full_like(current, -61.0)
    prolong_cartesian_2to1_into(
        coarse,
        i3(3, 1, 1),
        i3(6, 5, 5),
        i3(2, 2, 2),
        actual,
        i3(6, 2, 2),
        i3(8, 6, 6),
        i3(2, 2, 2),
    )
    assert_bits_equal(actual[:, :, 6:8, 2:6, 2:6], current[:, :, 6:8, 2:6, 2:6])


def test_current_nondyadic_face_satisfies_fixed_eta_error_bound() -> None:
    coarse, current, rnode = current_prolongation_fixture(
        np.asarray([-0.7, 1.1, -2.3]),
        np.asarray([1.9, 4.7, 7.2]),
    )
    exact = np.full_like(current, -67.0)
    prolong_cartesian_2to1_into(
        coarse,
        i3(3, 1, 1),
        i3(6, 5, 5),
        i3(2, 2, 2),
        exact,
        i3(6, 2, 2),
        i3(8, 6, 6),
        i3(2, 2, 2),
    )
    eps = np.finfo(np.float64).eps
    gamma6 = (6.0 * eps) / (1.0 - 6.0 * eps)
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    maximum_eta_delta = 0.0
    maximum_error = 0.0
    for i in range(6, 8):
        for j in range(2, 6):
            for k in range(2, 6):
                centers = []
                current_eta = []
                exact_eta = []
                for axis, index in enumerate((i, j, k)):
                    h_fine = rnode[6 + axis]
                    h_coarse = 2.0 * h_fine
                    inverse = 1.0 / h_coarse
                    x_fine_min = rnode[axis] - 2.0 * h_fine
                    x_coarse_min = rnode[axis] - 2.0 * h_coarse
                    x_fine = x_fine_min + (float(index) + 0.5) * h_fine
                    center_index = int((x_fine - x_coarse_min) * inverse)
                    expected_index = 2 + ((index - 2) // 2)
                    assert center_index == expected_index
                    x_coarse = x_coarse_min + (
                        float(center_index) + 0.5
                    ) * h_coarse
                    eta = (x_fine - x_coarse) * inverse
                    current_eta.append(eta)
                    exact_eta.append(-0.25 if (index - 2) % 2 == 0 else 0.25)
                    centers.append(center_index)
                    maximum_eta_delta = max(
                        maximum_eta_delta,
                        abs(eta - exact_eta[-1]),
                    )
                I, J, K = centers
                center = coarse[0, 0, I, J, K]
                slopes = (
                    three_point_limited_slope_reference(
                        coarse[0, 0, I - 1, J, K], center, coarse[0, 0, I + 1, J, K]
                    ),
                    three_point_limited_slope_reference(
                        coarse[0, 0, I, J - 1, K], center, coarse[0, 0, I, J + 1, K]
                    ),
                    three_point_limited_slope_reference(
                        coarse[0, 0, I, J, K - 1], center, coarse[0, 0, I, J, K + 1]
                    ),
                )
                b_eta = sum(
                    abs(float(slope)) * abs(current_eta[axis] - exact_eta[axis])
                    for axis, slope in enumerate(slopes)
                )
                m_exact = abs(float(center)) + sum(
                    abs(float(np.float64(slope) * np.float64(exact_eta[axis])))
                    for axis, slope in enumerate(slopes)
                )
                m_current = abs(float(center)) + sum(
                    abs(float(np.float64(slope) * np.float64(current_eta[axis])))
                    for axis, slope in enumerate(slopes)
                )
                bound = b_eta + gamma6 * (m_exact + m_current) + 8.0 * smallest
                error = abs(float(current[0, 0, i, j, k] - exact[0, 0, i, j, k]))
                maximum_error = max(maximum_error, error)
                assert error <= bound
    assert maximum_eta_delta > 0.0
    assert maximum_error >= 0.0


def test_weno_mapping_indices_match_and_eta_rounding_is_descriptive() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags, _ = get_metadata(str(path))
    root = header["domain_nx"] // header["block_nx"]
    forest = AMRForest(3, *map(int, root), np.asarray(flags, dtype=np.int32))
    mesh = AMRMesh(
        3,
        header["block_nx"].astype(np.uint32),
        header["domain_nx"].astype(np.uint32),
        header["xmin"].astype(np.float64),
        header["xmax"].astype(np.float64),
        np.uint32(0),
        np.uint32(1),
        forest,
    )
    rnode = np.asarray(mesh.rnode)
    maximum_eta_delta = 0.0
    wrong_indices = 0
    eta_patterns: list[int] = []
    for leaf in range(rnode.shape[0]):
        for axis in range(3):
            h_fine = rnode[leaf, 6 + axis]
            h_coarse = 2.0 * h_fine
            inverse = 1.0 / h_coarse
            x_fine_min = rnode[leaf, axis] - 2.0 * h_fine
            x_coarse_min = rnode[leaf, axis] - 2.0 * h_coarse
            for index in range(int(header["block_nx"][axis]) + 4):
                x_fine = x_fine_min + (float(index) + 0.5) * h_fine
                center_index = int((x_fine - x_coarse_min) * inverse)
                expected_index = 2 + ((index - 2) // 2)
                wrong_indices += center_index != expected_index
                x_coarse = x_coarse_min + (
                    float(center_index) + 0.5
                ) * h_coarse
                eta = np.float64((x_fine - x_coarse) * inverse)
                exact_eta = -0.25 if (index - 2) % 2 == 0 else 0.25
                maximum_eta_delta = max(
                    maximum_eta_delta, abs(float(eta) - exact_eta)
                )
                eta_pattern = bits(eta)
                if eta_pattern not in eta_patterns:
                    eta_patterns.append(eta_pattern)
    assert wrong_indices == 0
    assert maximum_eta_delta == np.float64(2.842170943040401e-14)
    assert len(eta_patterns) == 27
