"""RHC-001 synchronous completed refined-primary consumption."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .blockio import BlockReader
from .refined_halo import _execute_selected_refined_halos_with_consumer


@dataclass(frozen=True, slots=True)
class CompletedPrimaryConsumer:
    """Explicit state and one synchronous completed-primary callback."""

    state: Any
    consume_completed: Callable[..., None]
    output_arrays: tuple[np.ndarray, ...] = ()


class CompletedPrimaryExecutionStats(NamedTuple):
    primary_count: int
    chunk_count: int
    reader_call_count: int
    consumer_call_count: int
    selected_load_count: int
    maximum_selected_slots: int
    managed_array_bytes: int


def _require_output_arrays(
    output_arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if isinstance(output_arrays, np.ndarray) or not isinstance(
        output_arrays, Sequence
    ):
        raise TypeError("output_arrays must be a sequence of NumPy arrays")
    normalized = tuple(output_arrays)
    for value in normalized:
        if not isinstance(value, np.ndarray):
            raise TypeError("output_arrays entries must be NumPy arrays")
        if not value.flags.writeable:
            raise ValueError("consumer output arrays must be writable")
    return normalized


def make_completed_primary_consumer(
    state: Any,
    consume_completed: Callable[..., None],
    *,
    output_arrays: Sequence[np.ndarray] = (),
) -> CompletedPrimaryConsumer:
    """Create one frozen completed-primary consumer descriptor."""
    if not callable(consume_completed):
        raise TypeError("consume_completed must be callable")
    return CompletedPrimaryConsumer(
        state,
        consume_completed,
        _require_output_arrays(output_arrays),
    )


def _require_completed_primary_consumer(
    consumer: CompletedPrimaryConsumer,
) -> CompletedPrimaryConsumer:
    if not isinstance(consumer, CompletedPrimaryConsumer):
        raise TypeError("consumer must be a CompletedPrimaryConsumer")
    if not callable(consumer.consume_completed):
        raise TypeError("consumer consume_completed must be callable")
    output_arrays = _require_output_arrays(consumer.output_arrays)
    if output_arrays is consumer.output_arrays:
        return consumer
    return CompletedPrimaryConsumer(
        consumer.state,
        consumer.consume_completed,
        output_arrays,
    )


def execute_selected_refined_halos_with_consumer(
    reader: BlockReader,
    consumer: CompletedPrimaryConsumer,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    lower_halo: np.ndarray,
    upper_halo: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    slot_capacity: int,
    *,
    _additional_managed_array_bytes: int = 0,
) -> CompletedPrimaryExecutionStats:
    """Run bounded RHE and synchronously consume each completed primary prefix."""
    consumer = _require_completed_primary_consumer(consumer)

    def invoke(
        primary_offset: int,
        primary_ids: np.ndarray,
        payload: np.ndarray,
        valid_lower: np.ndarray,
        valid_upper: np.ndarray,
        interior_lower: np.ndarray,
        interior_upper: np.ndarray,
    ) -> None:
        return consumer.consume_completed(
            consumer.state,
            primary_offset,
            primary_ids,
            payload,
            valid_lower,
            valid_upper,
            interior_lower,
            interior_upper,
        )

    rhe_stats = _execute_selected_refined_halos_with_consumer(
        reader,
        primary_leaf_ids,
        field_ids,
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        lower_halo,
        upper_halo,
        boundary_modes,
        normal_field_slots,
        slot_capacity,
        invoke,
        consumer_output_arrays=consumer.output_arrays,
        additional_managed_array_bytes=_additional_managed_array_bytes,
    )
    return CompletedPrimaryExecutionStats(
        rhe_stats.primary_count,
        rhe_stats.chunk_count,
        rhe_stats.reader_calls,
        rhe_stats.chunk_count,
        rhe_stats.selected_load_count,
        rhe_stats.maximum_selected_slots,
        rhe_stats.managed_array_bytes,
    )
