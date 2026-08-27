"""Validated TOP-001 non-periodic level-1 face topology."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._topology import (
    fill_level1_face_neighbors_unchecked,
    validate_level1_maps_unchecked,
)
from .foundation import INDEX_DTYPE, _require_index_triplet
from .morton import _root_volume


PHYSICAL_BOUNDARY_ID: Final = -1


def _require_mapping_input(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_neighbor_output(
    face_neighbor_ids: np.ndarray,
    volume: int,
) -> np.ndarray:
    if not isinstance(face_neighbor_ids, np.ndarray):
        raise TypeError("face_neighbor_ids must be a NumPy array")
    if face_neighbor_ids.dtype != INDEX_DTYPE:
        raise TypeError("face_neighbor_ids must have dtype int64")
    if face_neighbor_ids.shape != (volume, 6):
        raise ValueError(
            f"face_neighbor_ids must have shape {(volume, 6)}, "
            f"got {face_neighbor_ids.shape}"
        )
    if not face_neighbor_ids.flags.c_contiguous:
        raise ValueError("face_neighbor_ids must be C-contiguous")
    if not face_neighbor_ids.flags.writeable:
        raise ValueError("face_neighbor_ids must be writable")
    return face_neighbor_ids


def fill_level1_face_neighbors(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    face_neighbor_ids: np.ndarray,
) -> None:
    """Validate MOR maps, then fill all six non-periodic face neighbors."""
    root_shape = _require_index_triplet("root_shape", root_shape)
    volume = _root_volume(root_shape)
    forward_shape = tuple(int(value) for value in root_shape)
    coord_to_rank = _require_mapping_input(
        "coord_to_rank",
        coord_to_rank,
        forward_shape,
    )
    rank_to_coord = _require_mapping_input(
        "rank_to_coord",
        rank_to_coord,
        (volume, 3),
    )
    face_neighbor_ids = _require_neighbor_output(face_neighbor_ids, volume)
    if any(
        np.shares_memory(face_neighbor_ids, source)
        for source in (root_shape, coord_to_rank, rank_to_coord)
    ):
        raise ValueError("face_neighbor_ids must not overlap topology inputs")

    invalid_rank = int(
        validate_level1_maps_unchecked(
            root_shape,
            coord_to_rank,
            rank_to_coord,
        )
    )
    if invalid_rank >= 0:
        raise ValueError(
            f"Morton maps are not a dense in-box inverse at rank {invalid_rank}"
        )

    fill_level1_face_neighbors_unchecked(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        face_neighbor_ids,
    )


def level1_face_neighbors(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
) -> np.ndarray:
    """Allocate and return the validated six-face neighbor table."""
    volume = _root_volume(root_shape)
    face_neighbor_ids = np.empty((volume, 6), dtype=np.int64)
    fill_level1_face_neighbors(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        face_neighbor_ids,
    )
    return face_neighbor_ids
