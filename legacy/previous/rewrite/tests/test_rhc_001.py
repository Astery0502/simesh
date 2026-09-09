from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field

import numpy as np
import pytest
import simesh_rewrite.refined_halo as refined_halo_module

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import (
    array_block_reader,
    array_block_writer,
    make_block_reader,
)
from simesh_rewrite.completed_primary import (
    CompletedPrimaryConsumer,
    CompletedPrimaryExecutionStats,
    _require_completed_primary_consumer,
    execute_selected_refined_halos_with_consumer,
    make_completed_primary_consumer,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_halo import execute_selected_refined_halos_from_blocks
from simesh_rewrite.storage import gather_blocks_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


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


def make_artifact(
    root_shape: tuple[int, int, int],
    refined_roots: set[tuple[int, int, int]] = frozenset(),
) -> Artifact:
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        refined = tuple(int(value) for value in coordinate) in refined_roots
        flags.append(not refined)
        if refined:
            flags.extend([True] * 8)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        np.asarray(flags, dtype=np.bool_),
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
    return Artifact(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def axis_coded_backing(
    leaf_count: int,
    field_count: int,
    block_shape: tuple[int, int, int] = (4, 4, 4),
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    result = np.empty(
        (leaf_count, field_count, *block_shape),
        dtype=np.float64,
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


@dataclass
class ReaderState:
    backing: np.ndarray
    fail_nonempty_call: int | None = None
    calls: list[np.ndarray] = field(default_factory=list)
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
    state.calls.append(block_ids.copy())
    if block_ids.size:
        state.nonempty_calls += 1
        if state.nonempty_calls == state.fail_nonempty_call:
            raise OSError("injected RHC reader failure")
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


def execution_arguments(
    artifact: Artifact,
    reader,
    consumer: CompletedPrimaryConsumer,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    slot_capacity: int,
    *,
    halo: np.ndarray | None = None,
) -> tuple:
    if halo is None:
        halo = i3(1, 1, 1)
    return (
        reader,
        consumer,
        primary_leaf_ids,
        field_ids,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        halo,
        halo,
        np.zeros((field_ids.shape[0], 6), dtype=np.uint8),
        i3(-1, -1, -1),
        slot_capacity,
    )


def writer_arguments(
    artifact: Artifact,
    reader,
    writer,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    slot_capacity: int,
    *,
    halo: np.ndarray | None = None,
) -> tuple:
    if halo is None:
        halo = i3(1, 1, 1)
    return (
        reader,
        writer,
        primary_leaf_ids,
        field_ids,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        halo,
        halo,
        np.zeros((field_ids.shape[0], 6), dtype=np.uint8),
        i3(-1, -1, -1),
        slot_capacity,
    )


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


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def noop_consumer(*_arguments) -> None:
    return None


def test_descriptor_factory_and_direct_descriptors_are_validated() -> None:
    output = np.empty(1, dtype=np.float64)
    descriptor = make_completed_primary_consumer(
        {"token": 7},
        noop_consumer,
        output_arrays=[output],
    )
    assert isinstance(descriptor, CompletedPrimaryConsumer)
    assert descriptor.state == {"token": 7}
    assert descriptor.consume_completed is noop_consumer
    assert descriptor.output_arrays == (output,)
    with pytest.raises(FrozenInstanceError):
        descriptor.state = None  # type: ignore[misc]

    with pytest.raises(TypeError, match="callable"):
        make_completed_primary_consumer(None, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        make_completed_primary_consumer(
            None,
            noop_consumer,
            output_arrays=output,
        )
    with pytest.raises(TypeError, match="NumPy arrays"):
        make_completed_primary_consumer(
            None,
            noop_consumer,
            output_arrays=(object(),),  # type: ignore[arg-type]
        )
    output.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        make_completed_primary_consumer(
            None,
            noop_consumer,
            output_arrays=(output,),
        )

    with pytest.raises(TypeError, match="CompletedPrimaryConsumer"):
        _require_completed_primary_consumer(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        _require_completed_primary_consumer(
            CompletedPrimaryConsumer(None, None, ())  # type: ignore[arg-type]
        )
    normalized = _require_completed_primary_consumer(
        CompletedPrimaryConsumer(None, noop_consumer, [])  # type: ignore[arg-type]
    )
    assert normalized.output_arrays == ()


@dataclass
class CopyConsumerState:
    output: np.ndarray
    expected_primary_ids: np.ndarray
    offsets: list[int] = field(default_factory=list)
    ids: list[np.ndarray] = field(default_factory=list)
    selected_sizes: list[int] = field(default_factory=list)


def copy_completed_primaries(
    state: CopyConsumerState,
    primary_offset: int,
    primary_leaf_ids: np.ndarray,
    payload: np.ndarray,
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
) -> None:
    for value in (
        primary_leaf_ids,
        payload,
        valid_lower,
        valid_upper,
        interior_lower,
        interior_upper,
    ):
        assert isinstance(value, np.ndarray)
        assert not value.flags.writeable
    stop = primary_offset + primary_leaf_ids.shape[0]
    assert np.array_equal(
        primary_leaf_ids,
        state.expected_primary_ids[primary_offset:stop],
    )
    assert valid_lower.tolist() == [0, 0, 0]
    assert valid_upper.tolist() == [6, 6, 6]
    assert interior_lower.tolist() == [1, 1, 1]
    assert interior_upper.tolist() == [5, 5, 5]
    state.output[primary_leaf_ids] = payload
    state.offsets.append(primary_offset)
    state.ids.append(primary_leaf_ids.copy())
    state.selected_sizes.append(payload.shape[0])


def test_bounded_callbacks_are_ordered_exactly_once_and_match_public_rhe() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    assert leaf_count == 71
    backing = axis_coded_backing(leaf_count, 2)
    primaries = np.arange(leaf_count, dtype=np.int64)
    fields = np.asarray([1, 0], dtype=np.int64)
    capacity = 57
    padded_shape = (6, 6, 6)

    expected = np.full(
        (leaf_count, fields.shape[0], *padded_shape),
        -701.0,
        dtype=np.float64,
    )
    public_stats = execute_selected_refined_halos_from_blocks(
        *writer_arguments(
            artifact,
            array_block_reader(backing),
            array_block_writer(expected),
            primaries,
            fields,
            capacity,
        )
    )

    actual = np.full_like(expected, -701.0)
    reader_state = ReaderState(backing)
    state = CopyConsumerState(actual, primaries)
    consumer = make_completed_primary_consumer(
        state,
        copy_completed_primaries,
        output_arrays=(actual,),
    )
    stats = execute_selected_refined_halos_with_consumer(
        *execution_arguments(
            artifact,
            recording_reader(reader_state),
            consumer,
            primaries,
            fields,
            capacity,
        )
    )

    assert isinstance(stats, CompletedPrimaryExecutionStats)
    assert stats.primary_count == leaf_count
    assert stats.chunk_count > 1
    assert stats.reader_call_count == stats.chunk_count
    assert stats.consumer_call_count == stats.chunk_count
    assert stats.selected_load_count == sum(
        call.size for call in reader_state.calls[1:]
    )
    assert stats.maximum_selected_slots == max(
        call.size for call in reader_state.calls[1:]
    )
    assert stats.maximum_selected_slots <= capacity
    assert stats.managed_array_bytes == expected_managed_bytes(
        capacity,
        fields.shape[0],
        (4, 4, 4),
        padded_shape,
    )
    assert stats.primary_count == public_stats.primary_count
    assert stats.chunk_count == public_stats.chunk_count
    assert stats.reader_call_count == public_stats.reader_calls
    assert stats.selected_load_count == public_stats.selected_load_count
    assert stats.maximum_selected_slots == public_stats.maximum_selected_slots
    assert stats.managed_array_bytes == public_stats.managed_array_bytes

    assert len(reader_state.calls) == stats.reader_call_count + 1
    assert reader_state.calls[0].size == 0
    assert state.offsets[0] == 0
    assert state.offsets == list(
        np.cumsum([0, *(ids.size for ids in state.ids[:-1])])
    )
    assert np.array_equal(np.concatenate(state.ids), primaries)
    assert sum(state.selected_sizes) == leaf_count
    assert_bits_equal(actual, expected)


def test_empty_selection_calls_only_reader_conformance_and_not_consumer() -> None:
    artifact = make_artifact((1, 1, 1))
    backing = axis_coded_backing(1, 1)
    reader_state = ReaderState(backing)
    output = np.asarray([np.float64(-0.0)], dtype=np.float64)
    before = output.copy()
    consumer_calls: list[int] = []

    def consume(*_arguments) -> None:
        consumer_calls.append(1)

    stats = execute_selected_refined_halos_with_consumer(
        *execution_arguments(
            artifact,
            recording_reader(reader_state),
            make_completed_primary_consumer(
                None,
                consume,
                output_arrays=(output,),
            ),
            np.empty(0, dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            0,
        )
    )

    assert stats[:6] == (0, 0, 0, 0, 0, 0)
    assert stats.managed_array_bytes == expected_managed_bytes(
        0,
        1,
        (4, 4, 4),
        (6, 6, 6),
    )
    assert len(reader_state.calls) == 1
    assert reader_state.calls[0].size == 0
    assert consumer_calls == []
    assert_bits_equal(output, before)


def test_consumer_output_aliases_fail_before_any_reader_call() -> None:
    artifact = make_artifact((1, 1, 1))
    backing = axis_coded_backing(1, 1)
    reader_state = ReaderState(backing)
    consumer = make_completed_primary_consumer(
        None,
        noop_consumer,
        output_arrays=(backing,),
    )
    with pytest.raises(ValueError, match="overlap reader or metadata"):
        execute_selected_refined_halos_with_consumer(
            *execution_arguments(
                artifact,
                recording_reader(reader_state),
                consumer,
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                1,
            )
        )
    assert reader_state.calls == []

    shared = np.empty(4, dtype=np.float64)
    consumer = make_completed_primary_consumer(
        None,
        noop_consumer,
        output_arrays=(shared[:3], shared[1:]),
    )
    with pytest.raises(ValueError, match="pairwise nonoverlapping"):
        execute_selected_refined_halos_with_consumer(
            *execution_arguments(
                artifact,
                recording_reader(reader_state),
                consumer,
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                1,
            )
        )
    assert reader_state.calls == []


def test_later_chunk_preflight_failure_precedes_even_empty_reader_call(
    monkeypatch,
) -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    reader_state = ReaderState(backing)
    output = np.full(leaf_count, -811.0, dtype=np.float64)
    before = output.copy()
    consumer_calls: list[int] = []
    original = refined_halo_module._preflight_chunk_actions
    preflight_calls = 0

    def fail_second_preflight(*arguments, **keywords) -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls == 2:
            raise ValueError("injected later chunk preflight failure")
        original(*arguments, **keywords)

    def consume(*_arguments) -> None:
        consumer_calls.append(1)

    monkeypatch.setattr(
        refined_halo_module,
        "_preflight_chunk_actions",
        fail_second_preflight,
    )
    with pytest.raises(ValueError, match="later chunk preflight"):
        execute_selected_refined_halos_with_consumer(
            *execution_arguments(
                artifact,
                recording_reader(reader_state),
                make_completed_primary_consumer(
                    None,
                    consume,
                    output_arrays=(output,),
                ),
                np.arange(leaf_count, dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                57,
            )
        )

    assert preflight_calls == 2
    assert reader_state.calls == []
    assert consumer_calls == []
    assert_bits_equal(output, before)


@dataclass
class MarkingState:
    marks: np.ndarray
    calls: list[np.ndarray] = field(default_factory=list)


def mark_completed(
    state: MarkingState,
    _primary_offset: int,
    primary_leaf_ids: np.ndarray,
    _payload: np.ndarray,
    _valid_lower: np.ndarray,
    _valid_upper: np.ndarray,
    _interior_lower: np.ndarray,
    _interior_upper: np.ndarray,
) -> None:
    state.marks[primary_leaf_ids] = 1
    state.calls.append(primary_leaf_ids.copy())


def test_later_reader_failure_preserves_only_completed_consumer_prefix() -> None:
    artifact = make_artifact((4, 4, 4), {(1, 1, 1)})
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1)
    reader_state = ReaderState(backing, fail_nonempty_call=2)
    marks = np.zeros(leaf_count, dtype=np.float64)
    state = MarkingState(marks)
    consumer = make_completed_primary_consumer(
        state,
        mark_completed,
        output_arrays=(marks,),
    )

    with pytest.raises(OSError, match="injected RHC reader failure"):
        execute_selected_refined_halos_with_consumer(
            *execution_arguments(
                artifact,
                recording_reader(reader_state),
                consumer,
                np.arange(leaf_count, dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                57,
            )
        )

    assert reader_state.nonempty_calls == 2
    assert len(state.calls) == 1
    completed = state.calls[0]
    assert np.array_equal(completed, np.arange(completed.size))
    assert np.all(marks[completed] == 1.0)
    assert np.all(marks[completed.size :] == 0.0)


@pytest.mark.parametrize(
    ("mode", "error", "match"),
    [
        ("raise", RuntimeError, "injected consumer failure"),
        ("return", TypeError, "must return None"),
    ],
)
def test_consumer_failures_propagate_after_current_callback_mutation(
    mode: str,
    error: type[Exception],
    match: str,
) -> None:
    artifact = make_artifact((1, 1, 1))
    backing = axis_coded_backing(1, 1)
    reader_state = ReaderState(backing)
    marker = np.zeros(1, dtype=np.float64)

    def consume(
        _state,
        _primary_offset,
        primary_leaf_ids,
        _payload,
        _valid_lower,
        _valid_upper,
        _interior_lower,
        _interior_upper,
    ):
        marker[primary_leaf_ids] = 1.0
        if mode == "raise":
            raise RuntimeError("injected consumer failure")
        return 17

    with pytest.raises(error, match=match):
        execute_selected_refined_halos_with_consumer(
            *execution_arguments(
                artifact,
                recording_reader(reader_state),
                make_completed_primary_consumer(
                    None,
                    consume,
                    output_arrays=(marker,),
                ),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                1,
            )
        )

    assert len(reader_state.calls) == 2
    assert reader_state.nonempty_calls == 1
    assert marker.tolist() == [1.0]
