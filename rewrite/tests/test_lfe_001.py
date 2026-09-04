from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
import pytest

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import array_block_reader, make_block_reader
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.curl_reference import cartesian_curl_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.local_field import (
    SelectedCurlExecutionStats,
    execute_selected_refined_curl_from_blocks,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.reductions_reference import accumulate_field_sum_reference
from simesh_rewrite.refined_halo_reference import (
    fill_selected_refined_halos_reference,
)
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.storage import gather_blocks_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class Artifact:
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    max_level: int


def make_artifact(
    root_shape: tuple[int, int, int],
    refined_roots: set[tuple[int, int, int]] = frozenset(),
) -> Artifact:
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        split = tuple(int(value) for value in coordinate) in refined_roots
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        np.asarray(flags, dtype=np.bool_),
    )
    max_level = validate_refined_forest_arrays(
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
    return Artifact(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        max_level,
    )


def geometry_inputs(
    artifact: Artifact,
    block_shape: tuple[int, int, int],
    *,
    unit_domain: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block_counts = i3(*block_shape)
    domain_counts = np.ascontiguousarray(
        artifact.root_shape * block_counts, dtype=np.int64
    )
    domain_lower = f3(0.0, 0.0, 0.0)
    if unit_domain:
        domain_upper = f3(1.0, 1.0, 1.0)
    else:
        domain_upper = np.ascontiguousarray(domain_counts, dtype=np.float64)
    return domain_lower, domain_upper, domain_counts, block_counts


def axis_coded_backing(
    leaf_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    result = np.empty(
        (leaf_count, field_count, *block_shape), dtype=np.float64
    )
    for leaf in range(leaf_count):
        for field_position in range(field_count):
            result[leaf, field_position] = (
                100000.0 * leaf
                + 10000.0 * field_position
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return result


def coordinate_backing(
    artifact: Artifact,
    block_shape: tuple[int, int, int],
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    field_function,
) -> np.ndarray:
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    result = np.empty((leaf_count, 3, *block_shape), dtype=np.float64)
    domain_counts = artifact.root_shape * np.asarray(block_shape, dtype=np.int64)
    base_h = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        base_h[axis] = extent / float(domain_counts[axis])
    for leaf, node_value in enumerate(artifact.leaf_node_ids):
        node = int(node_value)
        shift = int(artifact.node_levels[node]) - 1
        spacing = np.asarray(
            [math.ldexp(float(value), -shift) for value in base_h],
            dtype=np.float64,
        )
        centers = []
        for axis in range(3):
            first = int(artifact.node_coords[node, axis]) * block_shape[axis]
            factors = np.arange(first, first + block_shape[axis], dtype=np.float64)
            factors += 0.5
            centers.append(float(domain_lower[axis]) + factors * spacing[axis])
        x = centers[0][:, None, None]
        y = centers[1][None, :, None]
        z = centers[2][None, None, :]
        values = field_function(x, y, z)
        for field in range(3):
            result[leaf, field] = values[field]
    return result


@dataclass
class ReaderState:
    backing: np.ndarray
    fail_nonempty_call: int | None = None
    calls: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    nonempty_calls: int = 0


def recording_read(
    state: ReaderState,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls.append((block_ids.copy(), field_ids.copy()))
    if block_ids.size:
        state.nonempty_calls += 1
        if state.nonempty_calls == state.fail_nonempty_call:
            raise OSError("injected LFE reader failure")
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def recording_reader(state: ReaderState):
    return make_block_reader(
        state,
        state.backing.shape,
        recording_read,
        memory_arrays=(state.backing,),
    )


def lfe_arguments(
    artifact: Artifact,
    reader,
    primaries: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    magnetic_fields: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    modes: np.ndarray,
    normals: np.ndarray,
    capacity: int,
    curl_values: np.ndarray,
    reduction_component: int,
    accumulator: np.ndarray,
) -> tuple:
    domain_lower, domain_upper, domain_counts, block_counts = geometry
    return (
        reader,
        primaries,
        cell_lower,
        cell_upper,
        magnetic_fields,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        artifact.max_level,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        modes,
        normals,
        capacity,
        curl_values,
        reduction_component,
        accumulator,
    )


def slot_spacings(
    artifact: Artifact,
    primaries: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    domain_lower, domain_upper, domain_counts, _ = geometry
    base = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        base[axis] = extent / float(domain_counts[axis])
    result = np.empty((primaries.shape[0], 3), dtype=np.float64)
    for slot, leaf_value in enumerate(primaries):
        node = int(artifact.leaf_node_ids[int(leaf_value)])
        shift = int(artifact.node_levels[node]) - 1
        for axis in range(3):
            result[slot, axis] = math.ldexp(float(base[axis]), -shift)
    return result


def reference_lfe(
    artifact: Artifact,
    backing: np.ndarray,
    primaries: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    magnetic_fields: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    modes: np.ndarray,
    normals: np.ndarray,
    curl_values: np.ndarray,
    reduction_component: int,
    accumulator: np.ndarray,
) -> None:
    block_shape = tuple(int(value) for value in geometry[3])
    padded_shape = tuple(value + 2 for value in block_shape)
    completed = np.full(
        (artifact.leaf_node_ids.shape[0], 3, *padded_shape),
        np.nan,
        dtype=np.float64,
    )
    one = i3(1, 1, 1)
    fill_selected_refined_halos_reference(
        backing,
        primaries,
        magnetic_fields,
        artifact.root_shape,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        one,
        one,
        modes,
        normals,
        completed,
    )
    spacings = slot_spacings(artifact, primaries, geometry)
    fields = i3(0, 1, 2)
    for slot, leaf_value in enumerate(primaries):
        output_lower = one + cell_lower[slot]
        output_upper = one + cell_upper[slot]
        cartesian_curl_reference(
            completed[int(leaf_value) : int(leaf_value) + 1],
            output_lower,
            output_upper,
            fields,
            spacings[slot : slot + 1],
            curl_values[slot : slot + 1],
            fields,
            cell_lower[slot],
        )
        accumulate_field_sum_reference(
            curl_values[slot : slot + 1],
            cell_lower[slot],
            cell_upper[slot],
            reduction_component,
            accumulator,
        )


def expected_rhc_managed_bytes(
    capacity: int,
    block_shape: tuple[int, int, int],
) -> int:
    field_count = 3
    padded_volume = int(
        np.prod(tuple(value + 2 for value in block_shape), dtype=np.int64)
    )
    coarse_volume = int(
        np.prod(tuple(value + 1 for value in block_shape), dtype=np.int64)
    )
    return (
        capacity * (8 * field_count * padded_volume + 1854)
        + 8 * field_count * coarse_volume
        + 11778
        + 8 * field_count
    )


def expected_lfe_managed_bytes(
    capacity: int,
    primary_count: int,
    block_shape: tuple[int, int, int],
) -> int:
    return expected_rhc_managed_bytes(capacity, block_shape) + 72 * primary_count + 112


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def assert_only_boxes_changed(
    values: np.ndarray,
    before: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
) -> None:
    selected = np.zeros(values.shape, dtype=np.bool_)
    for slot in range(values.shape[0]):
        box = tuple(
            slice(int(cell_lower[slot, axis]), int(cell_upper[slot, axis]))
            for axis in range(3)
        )
        selected[(slot, slice(None), *box)] = True
    assert np.array_equal(
        values.view(np.uint64)[~selected], before.view(np.uint64)[~selected]
    )


def test_empty_and_single_partial_selection() -> None:
    artifact = make_artifact((1, 1, 1))
    block_shape = (4, 4, 4)
    geometry = geometry_inputs(artifact, block_shape)
    backing = axis_coded_backing(1, 3, block_shape)
    modes = np.zeros((3, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    fields = i3(0, 1, 2)

    empty_state = ReaderState(backing)
    empty_curl = np.empty((0, 3, *block_shape), dtype=np.float64)
    empty_accumulator = np.asarray([np.float64(-0.0)], dtype=np.float64)
    accumulator_before = empty_accumulator.copy()
    stats = execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            recording_reader(empty_state),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            fields,
            geometry,
            modes,
            normals,
            0,
            empty_curl,
            0,
            empty_accumulator,
        )
    )
    assert isinstance(stats, SelectedCurlExecutionStats)
    assert stats[:12] == (0,) * 12
    assert stats.managed_array_bytes == expected_lfe_managed_bytes(
        0, 0, block_shape
    )
    assert len(empty_state.calls) == 1
    assert empty_state.calls[0][0].size == 0
    assert_bits_equal(empty_accumulator, accumulator_before)

    primaries = i3(0)
    lower = np.asarray([[1, 0, 2]], dtype=np.int64)
    upper = np.asarray([[4, 3, 4]], dtype=np.int64)
    actual = np.full((1, 3, *block_shape), -701.0, dtype=np.float64)
    before = actual.copy()
    expected = actual.copy()
    actual_accumulator = np.asarray([1.25], dtype=np.float64)
    expected_accumulator = actual_accumulator.copy()
    state = ReaderState(backing)
    stats = execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            recording_reader(state),
            primaries,
            lower,
            upper,
            fields,
            geometry,
            modes,
            normals,
            1,
            actual,
            2,
            actual_accumulator,
        )
    )
    reference_lfe(
        artifact,
        backing,
        primaries,
        lower,
        upper,
        fields,
        geometry,
        modes,
        normals,
        expected,
        2,
        expected_accumulator,
    )
    assert_bits_equal(actual, expected)
    assert_bits_equal(actual_accumulator, expected_accumulator)
    assert_only_boxes_changed(actual, before, lower, upper)
    output_cells = int(np.prod(upper[0] - lower[0], dtype=np.int64))
    assert stats[:8] == (1, output_cells, 3 * output_cells, 1, 1, 1, 1, 1)
    assert stats.selected_load_count == 1
    assert stats.support_load_count == 0
    assert stats.maximum_selected_slots == 1
    assert stats.logical_reader_bytes == 3 * 4**3 * 8
    assert stats.managed_array_bytes == expected_lfe_managed_bytes(
        1, 1, block_shape
    )
    assert len(state.calls) == 2
    assert state.calls[0][0].size == 0


@pytest.mark.parametrize(
    "magnetic_fields",
    (i3(4, 1, 3), i3(2, 2, 0)),
    ids=("reordered", "repeated"),
)
def test_mixed_refinement_all_relations_and_pbc_match_independent_reference(
    magnetic_fields: np.ndarray,
) -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    block_shape = (4, 4, 4)
    geometry = geometry_inputs(artifact, block_shape)
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 5, block_shape)
    backing_before = backing.copy()
    primaries = i3(0, 1, 8)
    lower = np.asarray(((0, 0, 0), (0, 0, 0), (1, 1, 1)), dtype=np.int64)
    upper = np.asarray(((4, 4, 4), (4, 4, 4), (3, 3, 3)), dtype=np.int64)
    modes = np.zeros((3, 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 2] = 3
    modes[2, 4] = 1
    normals = i3(-1, 1, -1)
    actual = np.full((3, 3, *block_shape), -811.0, dtype=np.float64)
    before = actual.copy()
    expected = actual.copy()
    actual_accumulator = np.asarray([np.float64(-0.0)], dtype=np.float64)
    expected_accumulator = actual_accumulator.copy()
    state = ReaderState(backing)

    stats = execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            recording_reader(state),
            primaries,
            lower,
            upper,
            magnetic_fields,
            geometry,
            modes,
            normals,
            leaf_count,
            actual,
            0,
            actual_accumulator,
        )
    )
    reference_lfe(
        artifact,
        backing,
        primaries,
        lower,
        upper,
        magnetic_fields,
        geometry,
        modes,
        normals,
        expected,
        0,
        expected_accumulator,
    )

    assert_bits_equal(actual, expected)
    assert_bits_equal(actual_accumulator, expected_accumulator)
    assert_bits_equal(backing, backing_before)
    assert_only_boxes_changed(actual, before, lower, upper)
    kinds, masks, _, _ = balanced_refined_relations(
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        primaries,
        CANONICAL_DIRECTIONS,
    )
    assert set(int(value) for value in np.unique(kinds)) == {1, 2, 3, 4}
    assert np.any(masks != 0)
    assert set(int(value) for value in np.unique(modes)) == {0, 1, 2, 3}
    output_cells = 64 + 64 + 8
    assert stats.primary_count == 3
    assert stats.output_cell_count == output_cells
    assert stats.output_value_count == 3 * output_cells
    assert stats.chunk_count == stats.reader_call_count == stats.consumer_call_count == 1
    assert stats.operator_call_count == stats.reduction_call_count == 2
    assert stats.selected_load_count == 9
    assert stats.support_load_count == 6
    assert stats.maximum_selected_slots == 9
    assert stats.logical_reader_bytes == 9 * 3 * 4**3 * 8
    assert stats.managed_array_bytes == expected_lfe_managed_bytes(
        leaf_count, 3, block_shape
    )
    assert len(state.calls) == 2
    assert state.calls[0][0].size == 0
    assert np.array_equal(state.calls[1][1], magnetic_fields)


def test_affine_curl_is_constant_and_manual_sum_is_exact() -> None:
    artifact = make_artifact((1, 1, 1))
    block_shape = (6, 6, 6)
    geometry = geometry_inputs(artifact, block_shape)

    def affine(x, y, z):
        return (2.0 * y + 3.0 * z, 7.0 * x + 5.0 * z, 11.0 * x + 13.0 * y)

    backing = coordinate_backing(
        artifact, block_shape, geometry[0], geometry[1], affine
    )
    primaries = i3(0)
    lower = np.asarray([[1, 1, 1]], dtype=np.int64)
    upper = np.asarray([[5, 5, 5]], dtype=np.int64)
    output = np.full((1, 3, *block_shape), -907.0, dtype=np.float64)
    before = output.copy()
    accumulator = np.asarray([2.0], dtype=np.float64)
    stats = execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            array_block_reader(backing),
            primaries,
            lower,
            upper,
            i3(0, 1, 2),
            geometry,
            np.zeros((3, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            1,
            output,
            1,
            accumulator,
        )
    )
    box = (slice(None), slice(None), slice(1, 5), slice(1, 5), slice(1, 5))
    expected_components = np.asarray((8.0, -8.0, 5.0), dtype=np.float64)
    np.testing.assert_array_equal(
        output[box],
        np.broadcast_to(expected_components[None, :, None, None, None], (1, 3, 4, 4, 4)),
    )
    assert accumulator[0] == 2.0 - 8.0 * 4**3
    assert stats.output_cell_count == 4**3
    assert_only_boxes_changed(output, before, lower, upper)


def test_smooth_curl_converges_at_second_order() -> None:
    errors: list[float] = []
    for count in (8, 16, 32):
        artifact = make_artifact((1, 1, 1))
        block_shape = (count, count, count)
        geometry = geometry_inputs(artifact, block_shape, unit_domain=True)

        def smooth(_x, y, _z):
            zero = np.zeros((count, count, count), dtype=np.float64)
            bz = np.broadcast_to(np.sin(y), (count, count, count))
            return zero, zero, bz

        backing = coordinate_backing(
            artifact, block_shape, geometry[0], geometry[1], smooth
        )
        lower = np.asarray([[1, 1, 1]], dtype=np.int64)
        upper = np.asarray([[count - 1, count - 1, count - 1]], dtype=np.int64)
        output = np.full((1, 3, *block_shape), np.nan, dtype=np.float64)
        accumulator = np.zeros(1, dtype=np.float64)
        execute_selected_refined_curl_from_blocks(
            *lfe_arguments(
                artifact,
                array_block_reader(backing),
                i3(0),
                lower,
                upper,
                i3(0, 1, 2),
                geometry,
                np.zeros((3, 6), dtype=np.uint8),
                i3(-1, -1, -1),
                1,
                output,
                0,
                accumulator,
            )
        )
        centers = (np.arange(count, dtype=np.float64) + 0.5) / count
        expected = np.cos(centers[1:-1])[None, :, None]
        actual = output[0, 0, 1:-1, 1:-1, 1:-1]
        errors.append(float(np.max(np.abs(actual - expected))))
    orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in range(2)]
    assert min(orders) >= 1.8


def test_multi_chunk_capacity_invariant_values_sum_stats_and_equal_run_grouping() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    assert leaf_count == 71
    block_shape = (4, 4, 4)
    geometry = geometry_inputs(artifact, block_shape)
    backing = axis_coded_backing(leaf_count, 3, block_shape)
    primaries = np.arange(leaf_count, dtype=np.int64)
    lower = np.zeros((leaf_count, 3), dtype=np.int64)
    upper = np.full((leaf_count, 3), 4, dtype=np.int64)
    fields = i3(0, 1, 2)
    modes = np.zeros((3, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    bounded = np.full((leaf_count, 3, *block_shape), -1009.0, dtype=np.float64)
    resident = bounded.copy()
    bounded_accumulator = np.asarray([1.5], dtype=np.float64)
    resident_accumulator = bounded_accumulator.copy()
    bounded_reader_state = ReaderState(backing)
    resident_reader_state = ReaderState(backing)

    bounded_stats = execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            recording_reader(bounded_reader_state),
            primaries,
            lower,
            upper,
            fields,
            geometry,
            modes,
            normals,
            57,
            bounded,
            2,
            bounded_accumulator,
        )
    )
    resident_stats = execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            recording_reader(resident_reader_state),
            primaries,
            lower,
            upper,
            fields,
            geometry,
            modes,
            normals,
            leaf_count,
            resident,
            2,
            resident_accumulator,
        )
    )

    assert_bits_equal(bounded, resident)
    assert_bits_equal(bounded_accumulator, resident_accumulator)
    assert bounded_stats.chunk_count > 1
    assert resident_stats.chunk_count == 1
    for stats, state, capacity in (
        (bounded_stats, bounded_reader_state, 57),
        (resident_stats, resident_reader_state, leaf_count),
    ):
        real_calls = state.calls[1:]
        assert stats.reader_call_count == stats.chunk_count == len(real_calls)
        assert stats.consumer_call_count == stats.chunk_count
        assert stats.operator_call_count == stats.reduction_call_count == stats.chunk_count
        assert stats.selected_load_count == sum(call[0].size for call in real_calls)
        assert stats.support_load_count == stats.selected_load_count - leaf_count
        assert stats.maximum_selected_slots == max(call[0].size for call in real_calls)
        assert stats.logical_reader_bytes == stats.selected_load_count * 3 * 4**3 * 8
        assert stats.output_cell_count == leaf_count * 4**3
        assert stats.output_value_count == 3 * stats.output_cell_count
        assert stats.managed_array_bytes == expected_lfe_managed_bytes(
            capacity, leaf_count, block_shape
        )


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("type", TypeError, "int64"),
        ("selection", ValueError, "strictly increasing"),
        ("box", ValueError, "nonempty and contained"),
        ("field", ValueError, "out of range"),
        ("max_level", ValueError, "maximum forest node level"),
        ("boundary", ValueError, "unknown mode code"),
        ("capacity", ValueError, "universal all-26 closure"),
        ("output", ValueError, "curl_values must have shape"),
        ("accumulator", ValueError, "writable"),
    ],
)
def test_preflight_errors_precede_reader_and_output_mutation(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    block_shape = (4, 4, 4)
    geometry = geometry_inputs(artifact, block_shape)
    backing = axis_coded_backing(9, 3, block_shape)
    state = ReaderState(backing)
    primaries = i3(0, 1, 8)
    lower = np.zeros((3, 3), dtype=np.int64)
    upper = np.full((3, 3), 4, dtype=np.int64)
    fields = i3(0, 1, 2)
    modes = np.zeros((3, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    capacity = 9
    curl_values = np.full((3, 3, *block_shape), -1117.0, dtype=np.float64)
    accumulator = np.asarray([np.float64(-0.0)], dtype=np.float64)

    if failure == "type":
        fields = fields.astype(np.int32)
    elif failure == "selection":
        primaries = i3(0, 0, 8)
    elif failure == "box":
        upper = upper.copy()
        upper[-1, -1] = 5
    elif failure == "field":
        fields = i3(0, 1, 3)
    elif failure == "max_level":
        artifact = replace(artifact, max_level=artifact.max_level + 1)
    elif failure == "boundary":
        modes = modes.copy()
        modes[-1, -1] = 4
    elif failure == "capacity":
        capacity = 8
    elif failure == "output":
        curl_values = np.full((3, 2, *block_shape), -1117.0, dtype=np.float64)
    else:
        accumulator.flags.writeable = False

    curl_before = curl_values.copy()
    accumulator_before = accumulator.copy()
    with pytest.raises(error, match=match):
        execute_selected_refined_curl_from_blocks(
            *lfe_arguments(
                artifact,
                recording_reader(state),
                primaries,
                lower,
                upper,
                fields,
                geometry,
                modes,
                normals,
                capacity,
                curl_values,
                0,
                accumulator,
            )
        )
    assert state.calls == []
    assert_bits_equal(curl_values, curl_before)
    assert_bits_equal(accumulator, accumulator_before)


def test_output_aliases_fail_before_reader_or_mutation() -> None:
    artifact = make_artifact((1, 1, 1))
    block_shape = (4, 4, 4)
    geometry = geometry_inputs(artifact, block_shape)
    lower = np.zeros((1, 3), dtype=np.int64)
    upper = np.full((1, 3), 4, dtype=np.int64)
    modes = np.zeros((3, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)

    backing = axis_coded_backing(1, 3, block_shape)
    backing_before = backing.copy()
    state = ReaderState(backing)
    accumulator = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError, match="overlap reader memory"):
        execute_selected_refined_curl_from_blocks(
            *lfe_arguments(
                artifact,
                recording_reader(state),
                i3(0),
                lower,
                upper,
                i3(0, 1, 2),
                geometry,
                modes,
                normals,
                1,
                backing,
                0,
                accumulator,
            )
        )
    assert state.calls == []
    assert_bits_equal(backing, backing_before)

    backing = axis_coded_backing(1, 3, block_shape)
    state = ReaderState(backing)
    curl_values = np.full((1, 3, *block_shape), -1219.0, dtype=np.float64)
    before = curl_values.copy()
    overlapping_accumulator = curl_values.reshape(-1)[:1]
    with pytest.raises(ValueError, match="must not overlap"):
        execute_selected_refined_curl_from_blocks(
            *lfe_arguments(
                artifact,
                recording_reader(state),
                i3(0),
                lower,
                upper,
                i3(0, 1, 2),
                geometry,
                modes,
                normals,
                1,
                curl_values,
                0,
                overlapping_accumulator,
            )
        )
    assert state.calls == []
    assert_bits_equal(curl_values, before)


def test_later_reader_failure_preserves_completed_primary_prefix_and_sum() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    block_shape = (4, 4, 4)
    geometry = geometry_inputs(artifact, block_shape)
    backing = axis_coded_backing(leaf_count, 3, block_shape)
    primaries = np.arange(leaf_count, dtype=np.int64)
    lower = np.zeros((leaf_count, 3), dtype=np.int64)
    upper = np.full((leaf_count, 3), 4, dtype=np.int64)
    fields = i3(0, 1, 2)
    modes = np.zeros((3, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)

    expected = np.full((leaf_count, 3, *block_shape), np.nan, dtype=np.float64)
    expected_accumulator = np.asarray([3.25], dtype=np.float64)
    execute_selected_refined_curl_from_blocks(
        *lfe_arguments(
            artifact,
            array_block_reader(backing),
            primaries,
            lower,
            upper,
            fields,
            geometry,
            modes,
            normals,
            57,
            expected,
            1,
            expected_accumulator,
        )
    )

    actual = np.full_like(expected, np.nan)
    initial_accumulator = np.asarray([3.25], dtype=np.float64)
    actual_accumulator = initial_accumulator.copy()
    state = ReaderState(backing, fail_nonempty_call=2)
    with pytest.raises(OSError, match="injected LFE reader failure"):
        execute_selected_refined_curl_from_blocks(
            *lfe_arguments(
                artifact,
                recording_reader(state),
                primaries,
                lower,
                upper,
                fields,
                geometry,
                modes,
                normals,
                57,
                actual,
                1,
                actual_accumulator,
            )
        )
    completed_rows = np.all(np.isfinite(actual), axis=(1, 2, 3, 4))
    completed_count = int(np.count_nonzero(completed_rows))
    assert 0 < completed_count < leaf_count
    assert np.all(completed_rows[:completed_count])
    assert not np.any(completed_rows[completed_count:])
    assert_bits_equal(actual[:completed_count], expected[:completed_count])
    assert np.all(np.isnan(actual[completed_count:]))

    expected_prefix_accumulator = initial_accumulator.copy()
    for slot in range(completed_count):
        accumulate_field_sum_reference(
            expected[slot : slot + 1],
            lower[slot],
            upper[slot],
            1,
            expected_prefix_accumulator,
        )
    assert_bits_equal(actual_accumulator, expected_prefix_accumulator)
    assert state.nonempty_calls == 2
    assert len(state.calls) == 3
