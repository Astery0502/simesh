from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from simesh_rewrite.blockio import (
    BlockReader,
    array_block_reader,
    array_block_writer,
    make_block_reader,
    make_block_writer,
    read_blocks_into,
    write_blocks_from,
)
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def patterned_storage(shape=(5, 4, 4, 5, 6)) -> np.ndarray:
    block, field, x, y, z = np.indices(shape, dtype=np.float64)
    return 10000.0 * block + 1000.0 * field + 100.0 * x + 10.0 * y + z


@dataclass
class CountingStorage:
    data: np.ndarray
    calls: int = 0


def counting_read(
    state: CountingStorage,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls += 1
    gather_blocks_into(
        state.data,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def counting_write(
    state: CountingStorage,
    source: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls += 1
    scatter_blocks_from(
        source,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        state.data,
        destination_lower,
    )


def test_array_adapters_match_sto_001_and_expose_alias_arrays() -> None:
    backing = patterned_storage()
    destination = np.full((2, 2, 5, 5, 6), -1.0)
    expected = destination.copy()
    block_ids = np.array([3, 1], dtype=np.int64)
    field_ids = np.array([2, 0], dtype=np.int64)
    lower = i3(1, 1, 0)
    upper = i3(4, 5, 6)
    destination_lower = i3(0, 1, 0)

    gather_blocks_into(
        backing,
        lower,
        upper,
        block_ids,
        field_ids,
        expected,
        destination_lower,
    )
    reader = array_block_reader(backing)
    assert reader.state is backing
    assert reader.memory_arrays == (backing,)
    read_blocks_into(
        reader,
        lower,
        upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )
    assert np.array_equal(destination.view(np.uint64), expected.view(np.uint64))

    sink = np.full_like(backing, -9.0)
    expected_sink = sink.copy()
    scatter_blocks_from(
        destination,
        destination_lower,
        destination_lower + (upper - lower),
        block_ids,
        field_ids,
        expected_sink,
        lower,
    )
    writer = array_block_writer(sink)
    write_blocks_from(
        writer,
        destination,
        destination_lower,
        destination_lower + (upper - lower),
        block_ids,
        field_ids,
        lower,
    )
    assert np.array_equal(sink.view(np.uint64), expected_sink.view(np.uint64))


def test_explicit_nonarray_state_is_functionally_substitutable() -> None:
    backing = patterned_storage((4, 3, 3, 3, 4))
    state = CountingStorage(backing)
    reader = make_block_reader(
        state,
        backing.shape,
        counting_read,
        memory_arrays=(backing,),
    )
    destination = np.full((2, 2, 3, 3, 4), np.nan)
    block_ids = np.array([2, 0], dtype=np.int64)
    field_ids = np.array([1, 2], dtype=np.int64)
    read_blocks_into(
        reader,
        i3(0, 0, 0),
        i3(3, 3, 4),
        block_ids,
        field_ids,
        destination,
        i3(0, 0, 0),
    )
    expected = np.ascontiguousarray(backing[block_ids][:, field_ids])
    assert np.array_equal(destination, expected)
    assert state.calls == 1

    sink_state = CountingStorage(np.full_like(backing, -5.0))
    writer = make_block_writer(
        sink_state,
        sink_state.data.shape,
        counting_write,
        memory_arrays=(sink_state.data,),
    )
    write_blocks_from(
        writer,
        destination,
        i3(0, 0, 0),
        i3(3, 3, 4),
        block_ids,
        field_ids,
        i3(0, 0, 0),
    )
    assert np.array_equal(sink_state.data[2, 1], backing[2, 1])
    assert np.array_equal(sink_state.data[0, 2], backing[0, 2])
    assert sink_state.calls == 1


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ((1, 2, 3, 4), ValueError),
        ((1, 2, 3, 4, 0), ValueError),
        ((1, 2, 3, True, 5), TypeError),
    ],
)
def test_descriptor_shape_validation(shape, error) -> None:
    with pytest.raises(error):
        make_block_reader(object(), shape, lambda *args: None)


def test_directly_constructed_invalid_descriptor_is_revalidated() -> None:
    reader = BlockReader(object(), (1, 1, 1, 1, 0), lambda *args: None)
    with pytest.raises(ValueError, match="spatial extents"):
        read_blocks_into(
            reader,
            i3(0, 0, 0),
            i3(0, 0, 0),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 0, 1, 1, 1)),
            i3(0, 0, 0),
        )


def test_callbacks_must_return_none() -> None:
    reader = make_block_reader(
        object(),
        (0, 0, 1, 1, 1),
        lambda *args: 7,
    )
    with pytest.raises(TypeError, match="must return None"):
        read_blocks_into(
            reader,
            i3(0, 0, 0),
            i3(0, 0, 0),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 0, 1, 1, 1)),
            i3(0, 0, 0),
        )


def test_array_writer_rejects_read_only_storage() -> None:
    backing = np.zeros((1, 1, 1, 1, 1))
    backing.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        array_block_writer(backing)
