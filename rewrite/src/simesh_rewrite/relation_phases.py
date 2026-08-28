"""Validated RPH-001 refined relation child-phase codes."""

from __future__ import annotations

import numpy as np

from ._relation_phases import (
    fill_refined_relation_phase_codes_unchecked,
    validate_refined_relation_phases_unchecked,
    validate_selected_phase_nodes_unchecked,
)
from ._relation_slots import duplicate_selected_leaf_id_unchecked
from ._storage import validate_indices_unchecked
from .foundation import INDEX_DTYPE
from .relation_slots import _require_selected_leaf_ids


def _require_index_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous vector")
    return value


def _require_node_coords(value: np.ndarray, node_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("node_coords must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("node_coords must have dtype int64")
    if value.shape != (node_count, 3):
        raise ValueError(
            f"node_coords must have shape {(node_count, 3)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("node_coords must be C-contiguous")
    return value


def _require_phase_directions(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("directions must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("directions must have dtype int64")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError(f"directions must have shape (D,3), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError("directions must be C-contiguous")
    return value


def _require_uint8_matrix(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must have dtype uint8")
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_source_slots(value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("source_slots must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("source_slots must have dtype int64")
    if value.shape != shape:
        raise ValueError(f"source_slots must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError("source_slots must be C-contiguous")
    return value


def _require_phase_output(value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("fine_phase_codes must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("fine_phase_codes must have dtype uint8")
    if value.shape != shape:
        raise ValueError(
            f"fine_phase_codes must have shape {shape}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("fine_phase_codes must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError("fine_phase_codes must be writable")
    return value


def _phase_status_message(status: int) -> str:
    if status == 1:
        return "direction component is outside [-1,1]"
    if status == 2:
        return "direction row is center"
    if status == 3:
        return "physical mask uses unsupported bits"
    if status == 4:
        return "physical mask marks a zero direction axis"
    if status == 5:
        return "relation kind is invalid"
    if status == 6:
        return "physical mask and relation kind are inconsistent"
    if status == 7:
        return "source count is inconsistent with relation phase geometry"
    if status == 8:
        return "active source slot is out of range"
    if status == 9:
        return "inactive source slot must be -1"
    if status == 10:
        return "source level is inconsistent with relation kind"
    if status == 11:
        return "coarser source coordinate is inconsistent with reduced direction"
    if status == 12:
        return "finer source phase is incompatible with direction"
    if status == 13:
        return "finer source phases are not in canonical order"
    return f"unexpected phase validation status {status}"


def fill_refined_relation_phase_codes(
    selected_leaf_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    directions: np.ndarray,
    relation_kinds: np.ndarray,
    physical_masks: np.ndarray,
    source_counts: np.ndarray,
    source_slots: np.ndarray,
    fine_phase_codes: np.ndarray,
) -> None:
    """Fill exact COARSER-primary and FINER-source phase codes."""
    selected_leaf_ids = _require_selected_leaf_ids(selected_leaf_ids)
    node_levels = _require_index_vector("node_levels", node_levels)
    node_coords = _require_node_coords(node_coords, node_levels.shape[0])
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    directions = _require_phase_directions(directions)
    relation_kinds = _require_uint8_matrix("relation_kinds", relation_kinds)
    relation_shape = relation_kinds.shape
    physical_masks = _require_uint8_matrix(
        "physical_masks", physical_masks, relation_shape
    )
    source_counts = _require_uint8_matrix(
        "source_counts", source_counts, relation_shape
    )
    if relation_shape[1] != directions.shape[0]:
        raise ValueError("relation direction axis must match directions")
    tensor_shape = (*relation_shape, 4)
    source_slots = _require_source_slots(source_slots, tensor_shape)
    fine_phase_codes = _require_phase_output(fine_phase_codes, tensor_shape)

    if relation_shape[0] > selected_leaf_ids.shape[0]:
        raise ValueError("accepted relation rows exceed selected slots")
    invalid_selected = int(
        validate_indices_unchecked(selected_leaf_ids, leaf_node_ids.shape[0])
    )
    if invalid_selected >= 0:
        raise ValueError(
            f"selected_leaf_ids entry {invalid_selected} is out of range"
        )
    duplicate = int(
        duplicate_selected_leaf_id_unchecked(selected_leaf_ids)
    )
    if duplicate >= 0:
        raise ValueError(
            f"selected_leaf_ids entry {duplicate} duplicates an earlier ID"
        )
    status, slot, axis = validate_selected_phase_nodes_unchecked(
        selected_leaf_ids,
        node_levels,
        node_coords,
        leaf_node_ids,
    )
    if status == 1:
        raise ValueError(f"selected slot {slot} maps to an invalid node")
    if status == 2:
        raise ValueError(f"selected slot {slot} has an invalid node level")
    if status == 3:
        raise ValueError(
            f"selected slot {slot} has a negative node coordinate on axis {axis}"
        )
    if status != 0:
        raise RuntimeError(f"unexpected selected phase validation status {status}")

    status, row, direction, source, axis = (
        validate_refined_relation_phases_unchecked(
            selected_leaf_ids,
            node_levels,
            node_coords,
            leaf_node_ids,
            directions,
            relation_kinds,
            physical_masks,
            source_counts,
            source_slots,
        )
    )
    if status != 0:
        message = _phase_status_message(status)
        raise ValueError(
            f"{message} at row {row}, direction {direction}, "
            f"source {source}, axis {axis}"
        )

    if any(
        np.shares_memory(fine_phase_codes, value)
        for value in (
            selected_leaf_ids,
            node_levels,
            node_coords,
            leaf_node_ids,
            directions,
            relation_kinds,
            physical_masks,
            source_counts,
            source_slots,
        )
    ):
        raise ValueError("fine_phase_codes must not overlap inputs")

    fill_refined_relation_phase_codes_unchecked(
        selected_leaf_ids,
        node_coords,
        leaf_node_ids,
        relation_kinds,
        source_counts,
        source_slots,
        fine_phase_codes,
    )
