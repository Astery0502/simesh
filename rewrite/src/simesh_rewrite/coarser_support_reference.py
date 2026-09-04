"""Independent Python-integer reference for CSP-001."""

from __future__ import annotations

from itertools import product

import numpy as np


PLAN_CAPACITY = 18
SOURCE_COARSE = np.uint8(0)
SOURCE_FINE = np.uint8(1)
NO_SOURCE = np.uint8(255)

RELATION_PHYSICAL = 1
RELATION_COARSER = 2
RELATION_SAME = 3
RELATION_FINER = 4


def _direction_index(direction: tuple[int, int, int]) -> int:
    column = (
        (direction[2] + 1) * 9
        + (direction[1] + 1) * 3
        + direction[0]
        + 1
    )
    if column == 13:
        raise ValueError("center has no all-26 direction row")
    return column if column < 13 else column - 1


def _empty_plan() -> tuple[np.ndarray, ...]:
    slots = np.full(PLAN_CAPACITY, -1, dtype=np.int64)
    source_is_fine = np.full(PLAN_CAPACITY, NO_SOURCE, dtype=np.uint8)
    masks = np.zeros(PLAN_CAPACITY, dtype=np.uint8)
    matrices = tuple(
        np.zeros((PLAN_CAPACITY, 3), dtype=np.int64) for _ in range(10)
    )
    return slots, source_is_fine, masks, *matrices


def coarser_slope_support_plan_reference(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    selected_count: int,
    primary_slot: int,
    primary_phase_code: int,
    reduced_direction: np.ndarray,
    directions: np.ndarray,
    relation_kinds: np.ndarray,
    physical_masks: np.ndarray,
    source_counts: np.ndarray,
    source_slots: np.ndarray,
    fine_target_lower: np.ndarray,
    fine_target_upper: np.ndarray,
    coarse_source_lower: np.ndarray,
    coarse_source_upper: np.ndarray,
    workspace_source_lower: np.ndarray,
    workspace_source_upper: np.ndarray,
    workspace_required_lower: np.ndarray,
    workspace_required_upper: np.ndarray,
    workspace_coarse_origin: np.ndarray,
) -> tuple[int, int, *tuple[np.ndarray, ...]]:
    """Allocate the exact fixed-capacity CSP plan for one accepted CWP row.

    Inputs are expected to satisfy the checked CSP contract.  The reference
    deliberately uses Python integers, lists, and Cartesian products rather
    than the production fill kernel.
    """
    del selected_count, fine_target_lower, fine_target_upper
    lower = tuple(int(value) for value in interior_lower)
    upper = tuple(int(value) for value in interior_upper)
    block = tuple(upper[axis] - lower[axis] for axis in range(3))
    half = tuple(value // 2 for value in block)
    origin = tuple(int(value) for value in workspace_coarse_origin)
    required_lower = tuple(int(value) for value in workspace_required_lower)
    required_upper = tuple(int(value) for value in workspace_required_upper)
    workspace_lower = tuple(int(value) for value in workspace_source_lower)
    workspace_upper = tuple(int(value) for value in workspace_source_upper)
    phase_bits = tuple((int(primary_phase_code) >> axis) & 1 for axis in range(3))

    per_axis: list[list[tuple[int, int, int]]] = []
    for axis in range(3):
        pieces: list[tuple[int, int, int]] = []
        for component in (-1, 0, 1):
            start = max(
                required_lower[axis], origin[axis] + component * half[axis]
            )
            stop = min(
                required_upper[axis],
                origin[axis] + (component + 1) * half[axis],
            )
            if start < stop:
                pieces.append((component, start, stop))
        per_axis.append(pieces)

    raw: list[dict[str, object]] = []
    for z_piece, y_piece, x_piece in product(
        per_axis[2], per_axis[1], per_axis[0]
    ):
        pieces = (x_piece, y_piece, z_piece)
        direction = tuple(piece[0] for piece in pieces)
        target_lower = tuple(piece[1] for piece in pieces)
        target_upper = tuple(piece[2] for piece in pieces)
        inside_workspace = all(
            target_lower[axis] >= workspace_lower[axis]
            and target_upper[axis] <= workspace_upper[axis]
            for axis in range(3)
        )
        if inside_workspace:
            continue

        if direction == (0, 0, 0):
            kind = 0
            mask = 0
            direction_row = -1
        else:
            direction_row = _direction_index(direction)
            kind = int(relation_kinds[direction_row])
            mask = int(physical_masks[direction_row])
        raw.append(
            {
                "direction": direction,
                "direction_row": direction_row,
                "kind": kind,
                "mask": mask,
                "target_lower": target_lower,
                "target_upper": target_upper,
            }
        )

    plan = _empty_plan()
    (
        plan_source_slots,
        plan_source_is_fine,
        plan_physical_masks,
        plan_directions,
        plan_source_lower,
        plan_source_upper,
        plan_base_lower,
        plan_base_upper,
        plan_target_lower,
        plan_target_upper,
        plan_logical_interior_lower,
        plan_logical_interior_upper,
        plan_storage_logical_offsets,
    ) = plan

    reduced = tuple(int(value) for value in reduced_direction)
    reduced_row = _direction_index(reduced)
    record = 0
    plan_source_slots[record] = int(source_slots[reduced_row, 0])
    plan_source_is_fine[record] = SOURCE_COARSE
    plan_directions[record] = reduced
    plan_source_lower[record] = coarse_source_lower
    plan_source_upper[record] = coarse_source_upper
    plan_base_lower[record] = workspace_source_lower
    plan_base_upper[record] = workspace_source_upper
    plan_target_lower[record] = workspace_source_lower
    plan_target_upper[record] = workspace_source_upper
    record += 1

    for physical in (False, True):
        for item in raw:
            mask = int(item["mask"])
            if bool(mask) != physical:
                continue
            direction = tuple(int(value) for value in item["direction"])
            target_lower = tuple(int(value) for value in item["target_lower"])
            target_upper = tuple(int(value) for value in item["target_upper"])
            kind = int(item["kind"])
            direction_row = int(item["direction_row"])

            plan_physical_masks[record] = mask
            plan_directions[record] = direction
            plan_target_lower[record] = target_lower
            plan_target_upper[record] = target_upper

            if not physical:
                if kind == 0:
                    plan_source_slots[record] = primary_slot
                    plan_source_is_fine[record] = SOURCE_FINE
                elif kind == RELATION_SAME:
                    plan_source_slots[record] = source_slots[direction_row, 0]
                    plan_source_is_fine[record] = SOURCE_FINE
                elif kind == RELATION_COARSER:
                    plan_source_slots[record] = source_slots[direction_row, 0]
                    plan_source_is_fine[record] = SOURCE_COARSE
                else:
                    raise ValueError("reference received an invalid unmasked owner")

                if plan_source_is_fine[record] == SOURCE_FINE:
                    for axis in range(3):
                        band_lower = (
                            origin[axis] + direction[axis] * half[axis]
                        )
                        plan_source_lower[record, axis] = (
                            lower[axis]
                            + 2 * (target_lower[axis] - band_lower)
                        )
                        plan_source_upper[record, axis] = (
                            lower[axis]
                            + 2 * (target_upper[axis] - band_lower)
                        )
                else:
                    for axis in range(3):
                        quotient = (
                            phase_bits[axis] + direction[axis]
                        ) // 2
                        source_phase = (
                            phase_bits[axis]
                            + direction[axis]
                            - 2 * quotient
                        )
                        source_band_lower = (
                            lower[axis]
                            + source_phase * half[axis]
                        )
                        target_band_lower = (
                            origin[axis] + direction[axis] * half[axis]
                        )
                        plan_source_lower[record, axis] = source_band_lower + (
                            target_lower[axis] - target_band_lower
                        )
                        plan_source_upper[record, axis] = source_band_lower + (
                            target_upper[axis] - target_band_lower
                        )
                plan_base_lower[record] = target_lower
                plan_base_upper[record] = target_upper
            else:
                offset = tuple(
                    -direction[axis] if mask & (1 << axis) else 0
                    for axis in range(3)
                )
                base_lower = tuple(
                    target_lower[axis] + offset[axis] for axis in range(3)
                )
                base_upper = tuple(
                    target_upper[axis] + offset[axis] for axis in range(3)
                )
                plan_base_lower[record] = base_lower
                plan_base_upper[record] = base_upper

                owners = [
                    owner
                    for owner in range(record)
                    if int(plan_physical_masks[owner]) == 0
                    and all(
                        base_lower[axis] >= int(plan_target_lower[owner, axis])
                        and base_upper[axis] <= int(plan_target_upper[owner, axis])
                        for axis in range(3)
                    )
                ]
                if len(owners) != 1:
                    raise ValueError("reference physical base owner is not unique")
                owner = owners[0]
                if plan_source_is_fine[owner] == SOURCE_FINE:
                    plan_logical_interior_lower[record] = 0
                    plan_logical_interior_upper[record] = half
                    for axis in range(3):
                        plan_storage_logical_offsets[record, axis] = (
                            (
                                int(plan_source_lower[owner, axis])
                                - lower[axis]
                            )
                            // 2
                            - int(plan_target_lower[owner, axis])
                        )
                else:
                    plan_logical_interior_lower[record] = lower
                    plan_logical_interior_upper[record] = upper
                    plan_storage_logical_offsets[record] = (
                        plan_source_lower[owner] - plan_target_lower[owner]
                    )
            record += 1

    transfer_count = 1 + sum(int(item["mask"]) == 0 for item in raw)
    return transfer_count, record, *plan
