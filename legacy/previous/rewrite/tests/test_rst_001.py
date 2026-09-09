from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_support import (
    maximum_balanced_refined_support_slots,
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relations import RELATION_FINER, balanced_refined_relations
from simesh_rewrite.restriction import restrict_cartesian_2to1_into
from simesh_rewrite.restriction_reference import (
    restrict_cartesian_2to1_reference,
)
from simesh_rewrite.storage import gather_blocks_into


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


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def fill_eight_values(payload: np.ndarray, values: list[float]) -> None:
    coordinates = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
    for coordinate, value in zip(coordinates, values, strict=True):
        payload[(0, 0, *coordinate)] = np.float64(value)


def test_nonassociative_cell_freezes_current_addition_order() -> None:
    fine = np.empty((1, 1, 2, 2, 2), dtype=np.float64)
    fill_eight_values(
        fine,
        [1.0e16, 1.0, -1.0e16, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    expected = np.full((1, 1, 1, 1, 1), -9.0)
    actual = expected.copy()
    restrict_cartesian_2to1_reference(
        fine, i3(0, 0, 0), i3(2, 2, 2), expected, i3(0, 0, 0)
    )
    restrict_cartesian_2to1_into(
        fine, i3(0, 0, 0), i3(2, 2, 2), actual, i3(0, 0, 0)
    )
    assert expected[0, 0, 0, 0, 0] == np.float64(0.625)
    assert_bits_equal(actual, expected)


def test_random_multislot_field_translated_box_matches_scalar_reference() -> None:
    rng = np.random.default_rng(20260828)
    fine = np.ascontiguousarray(rng.normal(size=(2, 3, 10, 12, 14)))
    fine_before = fine.copy()
    fine_lower = i3(1, 2, 3)
    fine_upper = i3(9, 10, 13)
    coarse_lower = i3(2, 1, 3)
    expected = np.full((2, 3, 8, 7, 10), -19.0)
    actual = expected.copy()
    restrict_cartesian_2to1_reference(
        fine,
        fine_lower,
        fine_upper,
        expected,
        coarse_lower,
    )
    restrict_cartesian_2to1_into(
        fine,
        fine_lower,
        fine_upper,
        actual,
        coarse_lower,
    )
    assert_bits_equal(actual, expected)
    assert_bits_equal(fine, fine_before)
    target = np.zeros(actual.shape, dtype=bool)
    target[:, :, 2:6, 1:5, 3:8] = True
    assert np.all(actual[~target] == -19.0)


def test_constants_affine_centers_and_complete_partition() -> None:
    fine = np.empty((1, 2, 4, 6, 8), dtype=np.float64)
    fine[:, 0].fill(np.float64(1.5))
    x, y, z = np.indices((4, 6, 8), dtype=np.float64)
    fine[0, 1] = 3.0 + 2.0 * (x + 0.5) - 4.0 * (y + 0.5) + 0.5 * (z + 0.5)
    coarse = np.full((1, 2, 2, 3, 4), -77.0)
    restrict_cartesian_2to1_into(
        fine,
        i3(0, 0, 0),
        i3(4, 6, 8),
        coarse,
        i3(0, 0, 0),
    )
    assert np.all(coarse[0, 0] == np.float64(1.5))
    I, J, K = np.indices((2, 3, 4), dtype=np.float64)
    expected_affine = 3.0 + 2.0 * (2.0 * I + 1.0) - 4.0 * (
        2.0 * J + 1.0
    ) + 0.5 * (2.0 * K + 1.0)
    np.testing.assert_allclose(coarse[0, 1], expected_affine, rtol=0.0, atol=0.0)
    assert not np.any(coarse == -77.0)

    partitioned = np.full_like(coarse, -79.0)
    restrict_cartesian_2to1_into(
        fine,
        i3(0, 0, 0),
        i3(2, 6, 8),
        partitioned,
        i3(0, 0, 0),
    )
    restrict_cartesian_2to1_into(
        fine,
        i3(2, 0, 0),
        i3(4, 6, 8),
        partitioned,
        i3(1, 0, 0),
    )
    assert_bits_equal(partitioned, coarse)


def test_ieee_classification_and_signed_zero_follow_scalar_rule() -> None:
    cases = [
        ([-0.0] * 8, "negative_zero"),
        ([0.0] * 8, "positive_zero"),
        ([np.inf] * 8, "positive_infinity"),
        ([np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "nan"),
        ([np.finfo(np.float64).max] * 8, "positive_infinity"),
        ([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "nan"),
    ]
    with np.errstate(all="ignore"):
        for values, expected_class in cases:
            fine = np.empty((1, 1, 2, 2, 2), dtype=np.float64)
            fill_eight_values(fine, values)
            expected = np.empty((1, 1, 1, 1, 1), dtype=np.float64)
            actual = np.empty_like(expected)
            restrict_cartesian_2to1_reference(
                fine,
                i3(0, 0, 0),
                i3(2, 2, 2),
                expected,
                i3(0, 0, 0),
            )
            restrict_cartesian_2to1_into(
                fine,
                i3(0, 0, 0),
                i3(2, 2, 2),
                actual,
                i3(0, 0, 0),
            )
            value = actual[0, 0, 0, 0, 0]
            if expected_class == "negative_zero":
                assert value.view(np.uint64) == np.uint64(0x8000000000000000)
                assert_bits_equal(actual, expected)
            elif expected_class == "positive_zero":
                assert value.view(np.uint64) == np.uint64(0)
                assert_bits_equal(actual, expected)
            elif expected_class == "positive_infinity":
                assert np.isposinf(value)
                assert np.isposinf(expected[0, 0, 0, 0, 0])
            else:
                assert np.isnan(value)
                assert np.isnan(expected[0, 0, 0, 0, 0])


def test_empty_axes_read_only_input_and_outside_preservation() -> None:
    fine = np.arange(64, dtype=np.float64).reshape(1, 1, 4, 4, 4)
    fine.setflags(write=False)
    actual = np.full((1, 1, 3, 3, 3), -11.0)
    expected = actual.copy()
    restrict_cartesian_2to1_reference(
        fine, i3(0, 0, 0), i3(4, 4, 4), expected, i3(1, 1, 1)
    )
    restrict_cartesian_2to1_into(
        fine, i3(0, 0, 0), i3(4, 4, 4), actual, i3(1, 1, 1)
    )
    assert_bits_equal(actual, expected)
    assert np.all(actual[:, :, 0] == -11.0)

    for empty_fine, empty_coarse in (
        (
            np.empty((0, 2, 4, 4, 4), dtype=np.float64),
            np.empty((0, 2, 2, 2, 2), dtype=np.float64),
        ),
        (
            np.empty((2, 0, 4, 4, 4), dtype=np.float64),
            np.empty((2, 0, 2, 2, 2), dtype=np.float64),
        ),
    ):
        restrict_cartesian_2to1_into(
            empty_fine,
            i3(0, 0, 0),
            i3(4, 4, 4),
            empty_coarse,
            i3(0, 0, 0),
        )

    zero_extent_output = np.full((1, 1, 2, 2, 2), -13.0)
    before = zero_extent_output.copy()
    restrict_cartesian_2to1_into(
        np.empty((1, 1, 4, 4, 4), dtype=np.float64),
        i3(4, 0, 0),
        i3(4, 4, 4),
        zero_extent_output,
        i3(2, 0, 0),
    )
    assert_bits_equal(zero_extent_output, before)

    with pytest.raises(ValueError, match="even"):
        restrict_cartesian_2to1_into(
            np.empty((0, 1, 3, 4, 4), dtype=np.float64),
            i3(0, 0, 0),
            i3(3, 4, 4),
            np.empty((0, 1, 2, 2, 2), dtype=np.float64),
            i3(0, 0, 0),
        )


def assert_atomic_error(error, operation, destination: np.ndarray) -> None:
    before = destination.copy()
    with pytest.raises(error):
        operation()
    assert_bits_equal(destination, before)


def test_validation_errors_are_atomic() -> None:
    fine = np.arange(64, dtype=np.float64).reshape(1, 1, 4, 4, 4)
    lower = i3(0, 0, 0)
    upper = i3(4, 4, 4)
    coarse_lower = i3(0, 0, 0)

    def check(error, fine_value=fine, lo=lower, hi=upper, coarse_value=None, clo=coarse_lower):
        destination = (
            np.full((1, 1, 2, 2, 2), -31.0)
            if coarse_value is None
            else coarse_value
        )
        assert_atomic_error(
            error,
            lambda: restrict_cartesian_2to1_into(
                fine_value, lo, hi, destination, clo
            ),
            destination,
        )

    check(TypeError, fine_value=[fine])
    check(TypeError, fine_value=fine.astype(np.float32))
    check(ValueError, fine_value=fine[0])
    check(ValueError, fine_value=np.asfortranarray(fine))
    check(TypeError, lo=[0, 0, 0])
    check(TypeError, lo=lower.astype(np.int32))
    check(ValueError, lo=i3(0, 0))
    check(ValueError, lo=np.arange(6, dtype=np.int64)[::2])
    check(TypeError, hi=[4, 4, 4])
    check(TypeError, hi=upper.astype(np.int32))
    check(ValueError, hi=i3(4, 4))
    check(ValueError, hi=np.arange(6, dtype=np.int64)[::2])
    check(TypeError, coarse_value=np.full((1, 1, 2, 2, 2), -31, dtype=np.float32))
    check(ValueError, coarse_value=np.full((1, 2, 2, 2), -31.0))
    noncontiguous_coarse = np.full((1, 1, 2, 2, 4), -31.0)[..., ::2]
    assert not noncontiguous_coarse.flags.c_contiguous
    check(ValueError, coarse_value=noncontiguous_coarse)
    readonly = np.full((1, 1, 2, 2, 2), -31.0)
    readonly.setflags(write=False)
    check(ValueError, coarse_value=readonly)
    check(TypeError, clo=[0, 0, 0])
    check(TypeError, clo=coarse_lower.astype(np.int32))
    check(ValueError, clo=i3(0, 0))
    check(ValueError, clo=np.arange(6, dtype=np.int64)[::2])
    check(ValueError, coarse_value=np.full((2, 1, 2, 2, 2), -31.0))
    check(ValueError, coarse_value=np.full((1, 2, 2, 2, 2), -31.0))
    check(ValueError, lo=i3(-1, 0, 0))
    check(ValueError, lo=i3(3, 0, 0), hi=i3(2, 4, 4))
    check(ValueError, hi=i3(6, 4, 4))
    check(ValueError, hi=i3(3, 4, 4))
    check(ValueError, clo=i3(-1, 0, 0))
    check(ValueError, clo=i3(1, 0, 0))
    check(OverflowError, clo=i3(np.iinfo(np.int64).max, 0, 0))

    shared = fine.copy()
    assert_atomic_error(
        ValueError,
        lambda: restrict_cartesian_2to1_into(
            shared,
            i3(0, 0, 0),
            i3(2, 2, 2),
            shared,
            i3(0, 0, 0),
        ),
        shared,
    )
    metadata_base = np.full(8, -31.0)
    metadata_output = metadata_base.reshape(1, 1, 2, 2, 2)
    overlapping_lower = metadata_base.view(np.int64)[:3]
    overlapping_lower[:] = 0
    assert_atomic_error(
        ValueError,
        lambda: restrict_cartesian_2to1_into(
            fine,
            lower,
            upper,
            metadata_output,
            overlapping_lower,
        ),
        metadata_output,
    )

    for metadata_name in ("fine_lower", "fine_upper"):
        shared_metadata = np.full(8, -31.0)
        shared_output = shared_metadata.reshape(1, 1, 2, 2, 2)
        overlapping = shared_metadata.view(np.int64)[:3]
        overlapping[:] = 0 if metadata_name == "fine_lower" else 4
        call_lower = overlapping if metadata_name == "fine_lower" else lower
        call_upper = overlapping if metadata_name == "fine_upper" else upper
        assert_atomic_error(
            ValueError,
            lambda call_lower=call_lower, call_upper=call_upper, shared_output=shared_output: restrict_cartesian_2to1_into(
                fine,
                call_lower,
                call_upper,
                shared_output,
                coarse_lower,
            ),
            shared_output,
        )


def test_current_amrmesh_coarse_scratch_interior_is_bitwise_equal() -> None:
    flags = np.asarray([False, *([True] * 8), True], dtype=np.int32)
    forest = AMRForest(3, 2, 1, 1, flags)
    block = np.asarray([4, 4, 4], dtype=np.uint32)
    mesh = AMRMesh(
        3,
        block,
        np.asarray([8, 4, 4], dtype=np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(2),
        np.uint32(2),
        forest,
    )
    rng = np.random.default_rng(20260829)
    interior = np.ascontiguousarray(rng.normal(size=(forest.nleafs, 2, 4, 4, 4)))
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()
    coarse_scratch = np.asarray(mesh.datac)
    neighbor_type = np.asarray(forest.neighbor_type)
    eligible = np.flatnonzero(np.any(neighbor_type == 2, axis=1))
    assert eligible.size > 0
    for leaf in eligible:
        expected = np.empty((1, 2, 2, 2, 2), dtype=np.float64)
        restrict_cartesian_2to1_reference(
            interior[leaf : leaf + 1],
            i3(0, 0, 0),
            i3(4, 4, 4),
            expected,
            i3(0, 0, 0),
        )
        current = coarse_scratch[leaf, 2:4, 2:4, 2:4, :].transpose(3, 0, 1, 2)
        assert_bits_equal(current, expected[0])


def make_flags(
    root_shape: np.ndarray,
    refined_roots: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    _, root_coordinates = level1_morton(root_shape)
    flags: list[bool] = []
    for coordinate in root_coordinates:
        root = tuple(int(value) for value in coordinate)
        split = root in refined_roots
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    return np.asarray(flags, dtype=np.bool_)


def test_sto004_gather_restriction_bounded_matches_resident() -> None:
    root = i3(4, 3, 2)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = make_flags(root, ((0, 0, 0), (1, 1, 0), (2, 1, 1)))
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
    assert validate_refined_forest_arrays(
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
    ) == forest.max_level
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
    leaf_count = forest.leaf_node_ids.size
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    relation_kinds, _, source_counts, source_leaf_ids = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        leaf_ids,
        ALL_DIRECTIONS,
    )
    minimum = maximum_balanced_refined_support_slots(
        0,
        leaf_count,
        source_counts,
        source_leaf_ids,
    )

    rng = np.random.default_rng(20260830)
    backing = np.ascontiguousarray(rng.normal(size=(leaf_count, 3, 4, 4, 4)))
    backing.setflags(write=False)
    resident = np.empty((leaf_count, 3, 2, 2, 2), dtype=np.float64)
    restrict_cartesian_2to1_into(
        backing,
        i3(0, 0, 0),
        i3(4, 4, 4),
        resident,
        i3(0, 0, 0),
    )
    expected_fine_sources: list[int] = []
    for primary in range(leaf_count):
        for direction in range(ALL_DIRECTIONS.shape[0]):
            if relation_kinds[primary, direction] != RELATION_FINER:
                continue
            for source in range(int(source_counts[primary, direction])):
                source_leaf = int(
                    source_leaf_ids[primary, direction, source]
                )
                if source_leaf not in expected_fine_sources:
                    expected_fine_sources.append(source_leaf)
    assert expected_fine_sources

    capacities = [minimum]
    if leaf_count != minimum:
        capacities.append(leaf_count)
    for capacity in capacities:
        selected_ids = np.empty(capacity, dtype=np.int64)
        fine_workspace = np.empty((capacity, 3, 4, 4, 4), dtype=np.float64)
        restricted_source = np.empty((1, 3, 2, 2, 2), dtype=np.float64)
        seen_fine_sources: list[int] = []
        first = 0
        while first < leaf_count:
            candidate_count = min(capacity, leaf_count - first)
            primary_count, selected_count = plan_balanced_refined_support_prefix(
                first,
                leaf_count,
                source_counts[first : first + candidate_count],
                source_leaf_ids[first : first + candidate_count],
                selected_ids,
            )
            gather_blocks_into(
                backing,
                i3(0, 0, 0),
                i3(4, 4, 4),
                selected_ids[:selected_count],
                i3(0, 1, 2),
                fine_workspace[:selected_count],
                i3(0, 0, 0),
            )
            selected_values = selected_ids[:selected_count].tolist()
            chunk_fine_sources: list[int] = []
            for primary in range(first, first + primary_count):
                for direction in range(ALL_DIRECTIONS.shape[0]):
                    if relation_kinds[primary, direction] != RELATION_FINER:
                        continue
                    for source in range(int(source_counts[primary, direction])):
                        source_leaf = int(
                            source_leaf_ids[primary, direction, source]
                        )
                        if source_leaf not in chunk_fine_sources:
                            chunk_fine_sources.append(source_leaf)
            for source_leaf in chunk_fine_sources:
                assert source_leaf in selected_values
                source_slot = selected_values.index(source_leaf)
                restrict_cartesian_2to1_into(
                    fine_workspace[source_slot : source_slot + 1],
                    i3(0, 0, 0),
                    i3(4, 4, 4),
                    restricted_source,
                    i3(0, 0, 0),
                )
                assert_bits_equal(restricted_source[0], resident[source_leaf])
                if source_leaf not in seen_fine_sources:
                    seen_fine_sources.append(source_leaf)
            first += primary_count
        assert sorted(seen_fine_sources) == sorted(expected_fine_sources)
