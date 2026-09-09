from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest
import simesh_rewrite.refined_halo as refined_halo_module

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import (
    array_block_reader,
    array_block_writer,
    make_block_reader,
    make_block_writer,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_halo import (
    RefinedHaloExecutionStats,
    execute_selected_refined_halos_from_blocks,
)
from simesh_rewrite.refined_halo_reference import (
    fill_selected_refined_halos_reference,
)
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.relations_reference import balanced_refined_relations_reference
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class Artifact:
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    rank_to_coord: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    parent_node_ids: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    flags: np.ndarray


def make_artifact(
    root_shape: tuple[int, int, int],
    refined_roots: set[tuple[int, int, int]],
) -> Artifact:
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        refined = tuple(int(value) for value in coordinate) in refined_roots
        flags.append(not refined)
        if refined:
            flags.extend([True] * 8)
    flag_array = np.asarray(flags, dtype=np.bool_)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flag_array)
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
    return Artifact(
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
        flag_array,
    )


def axis_coded_backing(
    leaf_count: int,
    field_count: int,
    block_shape: tuple[int, int, int] = (4, 4, 4),
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    result = np.empty(
        (leaf_count, field_count, *block_shape), dtype=np.float64
    )
    for leaf in range(leaf_count):
        for field_index in range(field_count):
            result[leaf, field_index] = (
                100000.0 * leaf
                + 10000.0 * field_index
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return result


def execute_args(
    artifact: Artifact,
    reader,
    writer,
    primaries: np.ndarray,
    fields: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    modes: np.ndarray,
    normals: np.ndarray,
    capacity: int,
) -> tuple:
    return (
        reader,
        writer,
        primaries,
        fields,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        lower,
        upper,
        modes,
        normals,
        capacity,
    )


def reference_into(
    artifact: Artifact,
    backing: np.ndarray,
    output: np.ndarray,
    primaries: np.ndarray,
    fields: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    modes: np.ndarray,
    normals: np.ndarray,
) -> None:
    fill_selected_refined_halos_reference(
        backing,
        primaries,
        fields,
        artifact.root_shape,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        lower,
        upper,
        modes,
        normals,
        output,
    )


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def expected_managed_bytes(
    slot_capacity: int,
    field_count: int,
    block_shape: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
) -> int:
    padded_volume = int(np.prod(padded_shape, dtype=np.int64))
    coarse_volume = int(
        np.prod(tuple(value + 1 for value in block_shape), dtype=np.int64)
    )
    return (
        slot_capacity * (8 * field_count * padded_volume + 1854)
        + 8 * field_count * coarse_volume
        + 11778
        + 8 * field_count
    )


def test_tiny_all_action_array_execution_matches_independent_reference() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 3)
    backing_before = backing.copy()
    primaries = np.asarray([0, 1, 8], dtype=np.int64)
    fields = np.asarray([2, 0, 2], dtype=np.int64)
    lower = i3(2, 2, 2)
    upper = i3(2, 2, 2)
    modes = np.zeros((3, 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 2] = 3
    modes[2, 4] = 1
    normals = i3(-1, 1, -1)
    for value in (backing, primaries, fields, lower, upper, modes, normals):
        value.setflags(write=False)
    output = np.full((leaf_count, 3, 8, 8, 8), -777.0, dtype=np.float64)
    expected = output.copy()

    stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(output),
            primaries,
            fields,
            lower,
            upper,
            modes,
            normals,
            leaf_count,
        )
    )
    reference_into(
        artifact,
        backing,
        expected,
        primaries,
        fields,
        lower,
        upper,
        modes,
        normals,
    )

    assert isinstance(stats, RefinedHaloExecutionStats)
    assert stats[:6] == (3, 1, 1, 1, 9, 9)
    assert stats.managed_array_bytes == expected_managed_bytes(
        leaf_count, 3, (4, 4, 4), (8, 8, 8)
    )
    assert_bits_equal(output, expected)
    assert_bits_equal(backing, backing_before)
    unselected = sorted(set(range(leaf_count)) - set(map(int, primaries)))
    assert np.all(output[unselected] == -777.0)

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
    assert np.any((masks != 0) & (kinds == 2))
    assert np.any((masks != 0) & (kinds == 3))
    assert np.any((masks != 0) & (kinds == 4))


def test_production_and_checked_preflight_are_exactly_equivalent(
    monkeypatch,
) -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 2)
    primaries = np.asarray([0, 1, 8], dtype=np.int64)
    fields = np.asarray([1, 0], dtype=np.int64)
    halo = i3(2, 2, 2)
    modes = np.zeros((2, 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 2] = 3
    normals = i3(-1, 1, -1)
    checked = np.full((leaf_count, 2, 8, 8, 8), np.nan, dtype=np.float64)
    production = checked.copy()
    production_preflight = refined_halo_module._preflight_chunk_actions

    monkeypatch.setattr(
        refined_halo_module,
        "_preflight_chunk_actions",
        refined_halo_module._preflight_chunk_actions_checked_reference,
    )
    checked_stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(checked),
            primaries,
            fields,
            halo,
            halo,
            modes,
            normals,
            leaf_count,
        )
    )
    monkeypatch.setattr(
        refined_halo_module,
        "_preflight_chunk_actions",
        production_preflight,
    )
    production_stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(production),
            primaries,
            fields,
            halo,
            halo,
            modes,
            normals,
            leaf_count,
        )
    )
    assert production_stats == checked_stats
    assert_bits_equal(production, checked)


def test_capacity_57_and_resident_71_are_bitwise_equal_with_exact_stats() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    assert leaf_count == 71
    backing = np.arange(leaf_count * 64, dtype=np.float64).reshape(
        leaf_count, 1, 4, 4, 4
    )
    primaries = np.arange(leaf_count, dtype=np.int64)
    fields = np.asarray([0], dtype=np.int64)
    halo = i3(2, 2, 2)
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    bounded = np.full((leaf_count, 1, 8, 8, 8), np.nan, dtype=np.float64)
    resident = bounded.copy()

    bounded_stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(bounded),
            primaries,
            fields,
            halo,
            halo,
            modes,
            normals,
            57,
        )
    )
    resident_stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(resident),
            primaries,
            fields,
            halo,
            halo,
            modes,
            normals,
            leaf_count,
        )
    )

    assert_bits_equal(bounded, resident)
    assert bounded_stats[:6] == (71, 2, 2, 2, 106, 55)
    assert resident_stats[:6] == (71, 1, 1, 1, 71, 71)
    bytes_per_slot = 8 * 8 * 8 * 8 + 1854
    assert (
        resident_stats.managed_array_bytes - bounded_stats.managed_array_bytes
        == (71 - 57) * bytes_per_slot
    )


def test_distant_sparse_selection_reads_exact_independent_relation_union() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    primaries = np.asarray([0, 70], dtype=np.int64)
    fields = np.asarray([0], dtype=np.int64)
    halo = i3(2, 2, 2)
    output = np.full((leaf_count, 1, 8, 8, 8), -211.0, dtype=np.float64)
    reader_state = ReaderState(backing)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)

    _, _, counts, source_ids = balanced_refined_relations_reference(
        artifact.root_shape,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        primaries,
        CANONICAL_DIRECTIONS,
    )
    expected_primaries: list[int] = []
    expected_support: list[int] = []
    for primary_row, primary_value in enumerate(primaries):
        primary = int(primary_value)
        if primary in expected_support:
            expected_support.remove(primary)
        expected_primaries.append(primary)
        for direction_row in range(26):
            for source in range(int(counts[primary_row, direction_row])):
                leaf_id = int(source_ids[primary_row, direction_row, source])
                if (
                    leaf_id not in expected_primaries
                    and leaf_id not in expected_support
                ):
                    expected_support.append(leaf_id)
    expected_read_ids = np.asarray(
        expected_primaries + expected_support, dtype=np.int64
    )
    assert expected_read_ids.size <= 57

    stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            reader,
            writer,
            primaries,
            fields,
            halo,
            halo,
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            57,
        )
    )
    assert stats[:6] == (2, 1, 1, 1, expected_read_ids.size, expected_read_ids.size)
    assert np.array_equal(reader_state.calls[1][0], expected_read_ids)
    assert np.array_equal(writer_state.calls[1][0], primaries)
    omitted_gap = next(
        leaf_id
        for leaf_id in range(1, 70)
        if leaf_id not in set(map(int, expected_read_ids))
    )
    assert omitted_gap == 8
    assert omitted_gap not in set(map(int, reader_state.calls[1][0]))
    assert np.all(output[omitted_gap] == -211.0)


@dataclass
class ReaderState:
    backing: np.ndarray
    calls: list[tuple[np.ndarray, ...]] = field(default_factory=list)
    fail_nonempty_call: int | None = None
    nonempty_calls: int = 0


@dataclass
class WriterState:
    backing: np.ndarray
    calls: list[tuple[np.ndarray, ...]] = field(default_factory=list)
    fail_nonempty_call: int | None = None
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
    state.calls.append(
        (
            block_ids.copy(),
            field_ids.copy(),
            source_lower.copy(),
            source_upper.copy(),
            destination_lower.copy(),
        )
    )
    if block_ids.size:
        state.nonempty_calls += 1
        if state.nonempty_calls == state.fail_nonempty_call:
            raise OSError("injected reader failure")
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def recording_write(
    state: WriterState,
    source: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls.append(
        (
            block_ids.copy(),
            field_ids.copy(),
            source_lower.copy(),
            source_upper.copy(),
            destination_lower.copy(),
        )
    )
    if block_ids.size:
        state.nonempty_calls += 1
        if state.nonempty_calls == state.fail_nonempty_call:
            raise OSError("injected writer failure")
    scatter_blocks_from(
        source,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        state.backing,
        destination_lower,
    )


def adapters(reader_state: ReaderState, writer_state: WriterState):
    return (
        make_block_reader(
            reader_state,
            reader_state.backing.shape,
            recording_read,
            memory_arrays=(reader_state.backing,),
        ),
        make_block_writer(
            writer_state,
            writer_state.backing.shape,
            recording_write,
            memory_arrays=(writer_state.backing,),
        ),
    )


def test_sparse_asymmetric_custom_adapters_preserve_order_and_fields() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 3)
    primaries = np.asarray([0, 8], dtype=np.int64)
    fields = np.asarray([2, 0, 2], dtype=np.int64)
    lower = i3(1, 0, 2)
    upper = i3(0, 2, 1)
    padded = (5, 6, 7)
    modes = np.zeros((3, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    output = np.full((leaf_count, 3, *padded), -313.0, dtype=np.float64)
    expected = output.copy()
    reader_state = ReaderState(backing)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)

    stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            reader,
            writer,
            primaries,
            fields,
            lower,
            upper,
            modes,
            normals,
            leaf_count,
        )
    )
    reference_into(
        artifact,
        backing,
        expected,
        primaries,
        fields,
        lower,
        upper,
        modes,
        normals,
    )

    assert_bits_equal(output, expected)
    assert stats.reader_calls == stats.writer_calls == stats.chunk_count == 1
    assert len(reader_state.calls) == len(writer_state.calls) == 2
    assert reader_state.calls[0][0].size == writer_state.calls[0][0].size == 0
    assert np.array_equal(reader_state.calls[1][1], fields)
    assert np.array_equal(writer_state.calls[1][0], primaries)
    assert np.array_equal(writer_state.calls[1][1], np.arange(3))
    assert np.array_equal(reader_state.calls[1][2], i3(0, 0, 0))
    assert np.array_equal(reader_state.calls[1][3], i3(4, 4, 4))
    assert np.array_equal(reader_state.calls[1][4], lower)
    assert np.array_equal(writer_state.calls[1][2], i3(0, 0, 0))
    assert np.array_equal(writer_state.calls[1][3], i3(*padded))
    assert np.array_equal(writer_state.calls[1][4], i3(0, 0, 0))
    assert all(
        leaf_id in set(map(int, reader_state.calls[1][0]))
        for leaf_id in primaries
    )


def test_empty_selection_executes_only_empty_adapter_conformance() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    output = np.full((leaf_count, 1, 8, 8, 8), 29.0, dtype=np.float64)
    before = output.copy()
    reader_state = ReaderState(backing)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)
    stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            reader,
            writer,
            np.empty(0, dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            i3(2, 2, 2),
            i3(2, 2, 2),
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            0,
        )
    )
    assert stats[:6] == (0, 0, 0, 0, 0, 0)
    assert stats.managed_array_bytes == expected_managed_bytes(
        0, 1, (4, 4, 4), (8, 8, 8)
    )
    assert len(reader_state.calls) == len(writer_state.calls) == 1
    assert reader_state.calls[0][0].size == writer_state.calls[0][0].size == 0
    assert_bits_equal(output, before)


def test_empty_field_selection_is_valid_and_still_visits_primary_chunk() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    output = np.empty((leaf_count, 0, 8, 8, 8), dtype=np.float64)
    stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(output),
            np.asarray([0, 1, 8], dtype=np.int64),
            np.empty(0, dtype=np.int64),
            i3(2, 2, 2),
            i3(2, 2, 2),
            np.empty((0, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            leaf_count,
        )
    )
    assert stats[:6] == (3, 1, 1, 1, 9, 9)
    assert stats.managed_array_bytes == expected_managed_bytes(
        leaf_count, 0, (4, 4, 4), (8, 8, 8)
    )


def test_reader_failure_leaves_writer_unchanged() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    output = np.full((leaf_count, 1, 8, 8, 8), 41.0, dtype=np.float64)
    before = output.copy()
    reader_state = ReaderState(backing, fail_nonempty_call=1)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)
    with pytest.raises(OSError, match="reader"):
        execute_selected_refined_halos_from_blocks(
            *execute_args(
                artifact,
                reader,
                writer,
                np.asarray([0, 1, 8], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                i3(2, 2, 2),
                i3(2, 2, 2),
                np.zeros((1, 6), dtype=np.uint8),
                i3(-1, -1, -1),
                leaf_count,
            )
        )
    assert writer_state.nonempty_calls == 0
    assert_bits_equal(output, before)


@pytest.mark.parametrize(
    ("action", "bad_slot"),
    [("SAME", -1), ("SAME", "upper"), ("FINER", -1), ("FINER", "upper")],
)
def test_invalid_same_and_finer_slots_fail_before_reader(
    monkeypatch, action, bad_slot
) -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    output = np.full((leaf_count, 1, 8, 8, 8), 83.0, dtype=np.float64)
    output_before = output.copy()
    reader_state = ReaderState(backing)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)
    original = refined_halo_module.resolve_refined_relation_source_slots_unchecked

    def corrupt_slots(selected, counts, source_ids, source_slots):
        original(selected, counts, source_ids, source_slots)
        primary, direction = (0, 13) if action == "SAME" else (2, 12)
        source_slots[primary, direction, 0] = (
            selected.shape[0] if bad_slot == "upper" else bad_slot
        )

    monkeypatch.setattr(
        refined_halo_module,
        "resolve_refined_relation_source_slots_unchecked",
        corrupt_slots,
    )
    with pytest.raises(RuntimeError, match=f"{action} action source slot"):
        execute_selected_refined_halos_from_blocks(
            *execute_args(
                artifact,
                reader,
                writer,
                np.asarray([0, 1, 8], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                i3(2, 2, 2),
                i3(2, 2, 2),
                np.zeros((1, 6), dtype=np.uint8),
                i3(-1, -1, -1),
                leaf_count,
            )
        )
    assert reader_state.nonempty_calls == writer_state.nonempty_calls == 0
    assert len(reader_state.calls) == len(writer_state.calls) == 1
    assert_bits_equal(output, output_before)


def test_second_writer_failure_preserves_completed_prefix() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = np.arange(leaf_count * 64, dtype=np.float64).reshape(
        leaf_count, 1, 4, 4, 4
    )
    output = np.full((leaf_count, 1, 8, 8, 8), -919.0, dtype=np.float64)
    reader_state = ReaderState(backing)
    writer_state = WriterState(output, fail_nonempty_call=2)
    reader, writer = adapters(reader_state, writer_state)
    with pytest.raises(OSError, match="writer"):
        execute_selected_refined_halos_from_blocks(
            *execute_args(
                artifact,
                reader,
                writer,
                np.arange(leaf_count, dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                i3(2, 2, 2),
                i3(2, 2, 2),
                np.zeros((1, 6), dtype=np.uint8),
                i3(-1, -1, -1),
                57,
            )
        )
    assert reader_state.nonempty_calls == 2
    assert writer_state.nonempty_calls == 2
    first_written_ids = writer_state.calls[1][0]
    assert np.array_equal(first_written_ids, np.arange(39))
    assert np.all(output[first_written_ids] != -919.0)
    assert np.all(output[39:] == -919.0)


def test_safe_current_dyadic_refined_halos_are_bitwise_equal() -> None:
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh

    artifact = make_artifact((2, 2, 2), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 2)
    current_forest = AMRForest(3, 2, 2, 2, artifact.flags.astype(np.int32))
    block = np.asarray([4, 4, 4], dtype=np.uint32)
    current_mesh = AMRMesh(
        3,
        block,
        block * 2,
        np.zeros(3),
        np.ones(3),
        np.uint32(2),
        np.uint32(2),
        current_forest,
    )
    current_mesh.load_interior_data(backing)
    current_mesh.apply_ghost_cells()
    expected = np.transpose(current_mesh.padded_view(), (0, 4, 1, 2, 3)).copy()

    output = np.empty_like(expected)
    primary_ids = np.arange(leaf_count, dtype=np.int64)
    halo = i3(2, 2, 2)
    stats = execute_selected_refined_halos_from_blocks(
        *execute_args(
            artifact,
            array_block_reader(backing),
            array_block_writer(output),
            primary_ids,
            np.asarray([0, 1], dtype=np.int64),
            halo,
            halo,
            np.zeros((2, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            leaf_count,
        )
    )
    assert stats[:6] == (leaf_count, 1, 1, 1, leaf_count, leaf_count)
    assert_bits_equal(output, expected)


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        ("capacity", ValueError, "universal"),
        ("order", ValueError, "strictly increasing"),
        ("field", ValueError, "out of range"),
        ("reach", ValueError, "half a block"),
        ("writer_shape", ValueError, "writer shape"),
        ("noinflow", ValueError, "no-inflow"),
        ("alias", ValueError, "overlap"),
    ],
)
def test_static_request_errors_precede_nonempty_io(mutation, error, match) -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    output = np.full((leaf_count, 1, 8, 8, 8), 53.0, dtype=np.float64)
    output_before = output.copy()
    reader_state = ReaderState(backing)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)
    primaries = np.asarray([0, 1, 8], dtype=np.int64)
    fields = np.asarray([0], dtype=np.int64)
    lower = i3(2, 2, 2)
    upper = i3(2, 2, 2)
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    capacity = leaf_count

    if mutation == "capacity":
        capacity = leaf_count - 1
    elif mutation == "order":
        primaries = np.asarray([1, 0, 8], dtype=np.int64)
    elif mutation == "field":
        fields = np.asarray([1], dtype=np.int64)
    elif mutation == "reach":
        lower = i3(3, 2, 2)
    elif mutation == "writer_shape":
        bad = np.empty((leaf_count, 1, 8, 8, 7), dtype=np.float64)
        writer_state = WriterState(bad)
        _, writer = adapters(reader_state, writer_state)
    elif mutation == "noinflow":
        modes[0, 0] = 3
    else:
        writer = make_block_writer(
            ReaderState(backing),
            (leaf_count, 1, 8, 8, 8),
            recording_write,
            memory_arrays=(backing,),
        )

    with pytest.raises(error, match=match):
        execute_selected_refined_halos_from_blocks(
            *execute_args(
                artifact,
                reader,
                writer,
                primaries,
                fields,
                lower,
                upper,
                modes,
                normals,
                capacity,
            )
        )
    assert reader_state.nonempty_calls == 0
    assert writer_state.nonempty_calls == 0
    if mutation != "writer_shape":
        assert_bits_equal(output, output_before)


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        ("capacity_bool", TypeError, "integer"),
        ("capacity_negative", ValueError, "non-negative"),
        ("capacity_above_leaf_count", ValueError, "exceeds leaf count"),
        ("primary_dtype", TypeError, "dtype int64"),
        ("primary_duplicate", ValueError, "strictly increasing"),
        ("primary_range", ValueError, "strictly increasing in range"),
        ("forest_dtype", TypeError, "node_levels must have dtype int64"),
    ],
)
def test_static_scalar_selection_and_forest_representations_do_no_nonempty_io(
    mutation, error, match
) -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    output = np.full((leaf_count, 1, 8, 8, 8), 67.0, dtype=np.float64)
    output_before = output.copy()
    reader_state = ReaderState(backing)
    writer_state = WriterState(output)
    reader, writer = adapters(reader_state, writer_state)
    primaries = np.asarray([0, 1, 8], dtype=np.int64)
    capacity: object = leaf_count

    if mutation == "capacity_bool":
        capacity = True
    elif mutation == "capacity_negative":
        capacity = -1
    elif mutation == "capacity_above_leaf_count":
        capacity = leaf_count + 1
    elif mutation == "primary_dtype":
        primaries = primaries.astype(np.int32)
    elif mutation == "primary_duplicate":
        primaries = np.asarray([0, 0], dtype=np.int64)
    elif mutation == "primary_range":
        primaries = np.asarray([0, leaf_count], dtype=np.int64)
    else:
        artifact = replace(
            artifact, node_levels=artifact.node_levels.astype(np.int32)
        )

    with pytest.raises(error, match=match):
        execute_selected_refined_halos_from_blocks(
            *execute_args(
                artifact,
                reader,
                writer,
                primaries,
                np.asarray([0], dtype=np.int64),
                i3(2, 2, 2),
                i3(2, 2, 2),
                np.zeros((1, 6), dtype=np.uint8),
                i3(-1, -1, -1),
                capacity,  # type: ignore[arg-type]
            )
        )
    assert reader_state.nonempty_calls == 0
    assert writer_state.nonempty_calls == 0
    assert_bits_equal(output, output_before)
