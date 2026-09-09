"""Independent Python reference for TOP-001 face topology."""

from __future__ import annotations

import numpy as np


FACE_OFFSETS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def level1_face_neighbors_reference(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
) -> np.ndarray:
    shape = tuple(int(value) for value in root_shape)
    volume = shape[0] * shape[1] * shape[2]
    neighbors = np.empty((volume, 6), dtype=np.int64)
    for rank, coordinate in enumerate(rank_to_coord):
        x, y, z = (int(value) for value in coordinate)
        for face, (dx, dy, dz) in enumerate(FACE_OFFSETS):
            neighbor = (x + dx, y + dy, z + dz)
            if all(0 <= neighbor[axis] < shape[axis] for axis in range(3)):
                neighbors[rank, face] = coord_to_rank[neighbor]
            else:
                neighbors[rank, face] = -1
    return neighbors
