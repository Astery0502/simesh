"""Functional block reader/writer adapters for canonical payload transfers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from simesh._amr.foundation import _require_payload
from simesh._amr.storage import gather_blocks_into, scatter_blocks_from


BlockShape: TypeAlias = tuple[int, int, int, int, int]
BlockReadInto: TypeAlias = Callable[
    [
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    None,
]
BlockWriteFrom: TypeAlias = Callable[
    [
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    None,
]


@dataclass(frozen=True, slots=True)
class BlockReader:
    """Plain immutable description of one canonical block read function."""

    state: Any
    shape: BlockShape
    read_into: BlockReadInto
    memory_arrays: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True, slots=True)
class BlockWriter:
    """Plain immutable description of one canonical block write function."""

    state: Any
    shape: BlockShape
    write_from: BlockWriteFrom
    memory_arrays: tuple[np.ndarray, ...] = ()


def _require_shape(name: str, shape: Sequence[int]) -> BlockShape:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise TypeError(f"{name} shape must be a sequence")
    if len(shape) != 5:
        raise ValueError(f"{name} shape must have five entries")
    normalized: list[int] = []
    for axis, value in enumerate(shape):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} shape entry {axis} must be an integer")
        value = int(value)
        if value < 0 or (axis >= 2 and value == 0):
            raise ValueError(
                f"{name} block/field extents must be nonnegative and "
                "spatial extents must be positive"
            )
        normalized.append(value)
    return tuple(normalized)  # type: ignore[return-value]


def _require_memory_arrays(
    name: str,
    memory_arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if isinstance(memory_arrays, np.ndarray) or not isinstance(
        memory_arrays, Sequence
    ):
        raise TypeError(f"{name} memory_arrays must be a sequence of arrays")
    normalized = tuple(memory_arrays)
    if any(not isinstance(value, np.ndarray) for value in normalized):
        raise TypeError(f"{name} memory_arrays entries must be NumPy arrays")
    return normalized


def make_block_reader(
    state: Any,
    shape: Sequence[int],
    read_into: BlockReadInto,
    *,
    memory_arrays: Sequence[np.ndarray] = (),
) -> BlockReader:
    """Create an explicit coarse-grained block reader descriptor."""
    if not callable(read_into):
        raise TypeError("read_into must be callable")
    return BlockReader(
        state=state,
        shape=_require_shape("reader", shape),
        read_into=read_into,
        memory_arrays=_require_memory_arrays("reader", memory_arrays),
    )


def make_block_writer(
    state: Any,
    shape: Sequence[int],
    write_from: BlockWriteFrom,
    *,
    memory_arrays: Sequence[np.ndarray] = (),
) -> BlockWriter:
    """Create an explicit coarse-grained block writer descriptor."""
    if not callable(write_from):
        raise TypeError("write_from must be callable")
    return BlockWriter(
        state=state,
        shape=_require_shape("writer", shape),
        write_from=write_from,
        memory_arrays=_require_memory_arrays("writer", memory_arrays),
    )


def _array_read_into(
    state: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    gather_blocks_into(
        state,
        source_valid_lower,
        source_valid_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def _array_write_from(
    state: np.ndarray,
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    scatter_blocks_from(
        source,
        source_valid_lower,
        source_valid_upper,
        block_ids,
        field_ids,
        state,
        destination_lower,
    )


def array_block_reader(backing: np.ndarray) -> BlockReader:
    """Adapt a resident array or ``numpy.memmap`` as a block reader."""
    backing = _require_payload("backing", backing, writable=False)
    return make_block_reader(
        backing,
        backing.shape,
        _array_read_into,
        memory_arrays=(backing,),
    )


def array_block_writer(backing: np.ndarray) -> BlockWriter:
    """Adapt a writable resident array or ``numpy.memmap`` as a block writer."""
    backing = _require_payload("backing", backing, writable=True)
    return make_block_writer(
        backing,
        backing.shape,
        _array_write_from,
        memory_arrays=(backing,),
    )


def _require_block_reader(reader: BlockReader) -> BlockReader:
    if not isinstance(reader, BlockReader):
        raise TypeError("reader must be a BlockReader")
    # Revalidate descriptors constructed directly rather than by the factory.
    shape = _require_shape("reader", reader.shape)
    if not callable(reader.read_into):
        raise TypeError("reader read_into must be callable")
    memory_arrays = _require_memory_arrays("reader", reader.memory_arrays)
    if (
        isinstance(reader.shape, tuple)
        and shape == reader.shape
        and isinstance(reader.memory_arrays, tuple)
    ):
        return reader
    return BlockReader(reader.state, shape, reader.read_into, memory_arrays)


def _require_block_writer(writer: BlockWriter) -> BlockWriter:
    if not isinstance(writer, BlockWriter):
        raise TypeError("writer must be a BlockWriter")
    shape = _require_shape("writer", writer.shape)
    if not callable(writer.write_from):
        raise TypeError("writer write_from must be callable")
    memory_arrays = _require_memory_arrays("writer", writer.memory_arrays)
    if (
        isinstance(writer.shape, tuple)
        and shape == writer.shape
        and isinstance(writer.memory_arrays, tuple)
    ):
        return writer
    return BlockWriter(writer.state, shape, writer.write_from, memory_arrays)


def read_blocks_into(
    reader: BlockReader,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    """Invoke one reader transfer at the Python chunk boundary."""
    reader = _require_block_reader(reader)
    result = reader.read_into(
        reader.state,
        source_valid_lower,
        source_valid_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )
    if result is not None:
        raise TypeError("block reader callbacks must return None")


def write_blocks_from(
    writer: BlockWriter,
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    """Invoke one writer transfer at the Python chunk boundary."""
    writer = _require_block_writer(writer)
    result = writer.write_from(
        writer.state,
        source,
        source_valid_lower,
        source_valid_upper,
        block_ids,
        field_ids,
        destination_lower,
    )
    if result is not None:
        raise TypeError("block writer callbacks must return None")
