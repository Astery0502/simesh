"""Validated CSP-001 COARSER slope-support planning."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from ._coarser_support import fill_coarser_slope_support_plan_unchecked
from .foundation import INDEX_DTYPE, _require_index_triplet
from .workspace import _require_nonnegative_integer


PLAN_CAPACITY = 18
SOURCE_COARSE = np.uint8(0)
SOURCE_FINE = np.uint8(1)
NO_SOURCE = np.uint8(255)

RELATION_PHYSICAL = 1
RELATION_COARSER = 2
RELATION_SAME = 3
RELATION_FINER = 4

_INDEX_MIN = int(np.iinfo(np.int64).min)
_INDEX_MAX = int(np.iinfo(np.int64).max)

CANONICAL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)
CANONICAL_DIRECTIONS.setflags(write=False)


def _checked_int64(name: str, value: int) -> int:
    if value < _INDEX_MIN or value > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return value


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


def _require_array(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, ...],
    *,
    writable: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_triplet(name: str, value: np.ndarray) -> np.ndarray:
    try:
        return _require_index_triplet(name, value)
    except ValueError as error:
        raise ValueError(str(error)) from None


def _validate_relation_row(
    directions: np.ndarray,
    relation_kinds: np.ndarray,
    physical_masks: np.ndarray,
    source_counts: np.ndarray,
    source_slots: np.ndarray,
    selected_count: int,
) -> None:
    if not np.array_equal(directions, CANONICAL_DIRECTIONS):
        raise ValueError("directions must be the exact canonical x-fast all-26 rows")

    for direction_row in range(26):
        direction = tuple(int(value) for value in directions[direction_row])
        kind = int(relation_kinds[direction_row])
        mask = int(physical_masks[direction_row])
        count = int(source_counts[direction_row])
        if mask & ~7:
            raise ValueError("physical mask uses unsupported bits")
        reduced_noncenter = False
        neutral_axes = 0
        for axis in range(3):
            if mask & (1 << axis):
                if direction[axis] == 0:
                    raise ValueError("physical mask marks a zero direction axis")
                neutral_axes += 1
            elif direction[axis] == 0:
                neutral_axes += 1
            else:
                reduced_noncenter = True
        if kind < RELATION_PHYSICAL or kind > RELATION_FINER:
            raise ValueError("relation kind is outside [1,4]")
        if kind == RELATION_PHYSICAL:
            if reduced_noncenter or count != 0:
                raise ValueError("PHYSICAL relation representation is inconsistent")
        else:
            if not reduced_noncenter:
                raise ValueError("nonphysical relation has center reduced direction")
            expected = (1 << neutral_axes) if kind == RELATION_FINER else 1
            if count != expected or count > 4:
                raise ValueError("relation source count is inconsistent")
        for source in range(4):
            slot = int(source_slots[direction_row, source])
            if source < count:
                if slot < 0 or slot >= selected_count:
                    raise ValueError("active relation source slot is out of range")
            elif slot != -1:
                raise ValueError("inactive relation source slot must be -1")


def _expected_cwp(
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    direction: tuple[int, int, int],
    phase: int,
    target_lower: tuple[int, int, int],
    target_upper: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    outputs: list[list[int]] = [[] for _ in range(7)]
    for axis in range(3):
        block = upper[axis] - lower[axis]
        bit = (phase >> axis) & 1
        component = direction[axis]
        logical_origin = _checked_int64(
            "logical coarse origin",
            lower[axis]
            + bit * (block // 2)
            - ((bit + component) // 2) * block,
        )
        center_lower = _checked_int64(
            "coarse center lower",
            logical_origin + (target_lower[axis] - lower[axis]) // 2,
        )
        center_upper = _checked_int64(
            "coarse center upper",
            logical_origin + (target_upper[axis] - 1 - lower[axis]) // 2 + 1,
        )
        required_lower = _checked_int64(
            "coarse required lower", center_lower - 1
        )
        required_upper = _checked_int64(
            "coarse required upper", center_upper + 1
        )
        source_lower = max(required_lower, lower[axis])
        source_upper = min(required_upper, upper[axis])
        base = min(required_lower, logical_origin)
        values = (
            source_lower,
            source_upper,
            _checked_int64("workspace source lower", source_lower - base),
            _checked_int64("workspace source upper", source_upper - base),
            _checked_int64("workspace required lower", required_lower - base),
            _checked_int64("workspace required upper", required_upper - base),
            _checked_int64("workspace coarse origin", logical_origin - base),
        )
        for output, value in zip(outputs, values, strict=True):
            output.append(value)
    return tuple(tuple(output) for output in outputs)


def _box_contains(
    outer_lower: tuple[int, int, int],
    outer_upper: tuple[int, int, int],
    inner_lower: tuple[int, int, int],
    inner_upper: tuple[int, int, int],
) -> bool:
    return all(
        outer_lower[axis] <= inner_lower[axis]
        and inner_upper[axis] <= outer_upper[axis]
        for axis in range(3)
    )


def _boxes_disjoint(
    left_lower: tuple[int, int, int],
    left_upper: tuple[int, int, int],
    right_lower: tuple[int, int, int],
    right_upper: tuple[int, int, int],
) -> bool:
    return any(
        left_upper[axis] <= right_lower[axis]
        or right_upper[axis] <= left_lower[axis]
        for axis in range(3)
    )


def _box_volume(
    lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> int:
    volume = 1
    for axis in range(3):
        volume *= upper[axis] - lower[axis]
    return volume


def _validate_plan_geometry(
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    primary_slot: int,
    phase: int,
    reduced: tuple[int, int, int],
    relation_kinds: np.ndarray,
    physical_masks: np.ndarray,
    source_slots: np.ndarray,
    coarse_source_lower: tuple[int, int, int],
    coarse_source_upper: tuple[int, int, int],
    workspace_source_lower: tuple[int, int, int],
    workspace_source_upper: tuple[int, int, int],
    required_lower: tuple[int, int, int],
    required_upper: tuple[int, int, int],
    origin: tuple[int, int, int],
) -> None:
    block = tuple(upper[axis] - lower[axis] for axis in range(3))
    half = tuple(value // 2 for value in block)
    phase_bits = tuple((phase >> axis) & 1 for axis in range(3))
    per_axis: list[list[tuple[int, int, int]]] = []
    for axis in range(3):
        pieces: list[tuple[int, int, int]] = []
        for component in (-1, 0, 1):
            band_lower = _checked_int64(
                "support band lower", origin[axis] + component * half[axis]
            )
            band_upper = _checked_int64(
                "support band upper", origin[axis] + (component + 1) * half[axis]
            )
            start = max(required_lower[axis], band_lower)
            stop = min(required_upper[axis], band_upper)
            if start < stop:
                pieces.append((component, start, stop))
        if (
            not pieces
            or pieces[0][1] != required_lower[axis]
            or pieces[-1][2] != required_upper[axis]
            or any(
                pieces[index][2] != pieces[index + 1][1]
                for index in range(len(pieces) - 1)
            )
        ):
            raise ValueError("required rectangle exceeds the all-26 half-block lattice")
        per_axis.append(pieces)

    transfers: list[dict[str, object]] = [
        {
            "target_lower": workspace_source_lower,
            "target_upper": workspace_source_upper,
            "source_lower": coarse_source_lower,
            "source_is_fine": False,
        }
    ]
    physical_records: list[dict[str, object]] = []
    raw_target_volume = 0
    merged_volume = 0
    outside_targets: list[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ] = []

    for z_piece, y_piece, x_piece in product(
        per_axis[2], per_axis[1], per_axis[0]
    ):
        pieces = (x_piece, y_piece, z_piece)
        direction = tuple(piece[0] for piece in pieces)
        target_lower = tuple(piece[1] for piece in pieces)
        target_upper = tuple(piece[2] for piece in pieces)
        volume = _box_volume(target_lower, target_upper)
        raw_target_volume += volume
        inside_workspace = _box_contains(
            workspace_source_lower,
            workspace_source_upper,
            target_lower,
            target_upper,
        )
        disjoint_workspace = _boxes_disjoint(
            workspace_source_lower,
            workspace_source_upper,
            target_lower,
            target_upper,
        )
        if inside_workspace:
            merged_volume += volume
            continue
        if not disjoint_workspace:
            raise ValueError("support tile partially overlaps CWP workspace source")
        outside_targets.append((target_lower, target_upper))

        if direction == (0, 0, 0):
            kind = 0
            mask = 0
            direction_row = -1
        else:
            direction_row = _direction_index(direction)
            kind = int(relation_kinds[direction_row])
            mask = int(physical_masks[direction_row])
        if kind == RELATION_FINER:
            raise ValueError("CSP support owner must not be FINER")

        if mask == 0:
            if kind == 0:
                slot = primary_slot
                source_is_fine = True
            elif kind == RELATION_SAME:
                slot = int(source_slots[direction_row, 0])
                source_is_fine = True
            elif kind == RELATION_COARSER:
                slot = int(source_slots[direction_row, 0])
                source_is_fine = False
            else:
                raise ValueError("unmasked support owner is invalid")
            del slot
            source_lower_values: list[int] = []
            source_upper_values: list[int] = []
            for axis in range(3):
                if source_is_fine:
                    band_lower = _checked_int64(
                        "fine support band lower",
                        origin[axis] + direction[axis] * half[axis],
                    )
                    source_lower = (
                        lower[axis]
                        + 2 * (target_lower[axis] - band_lower)
                    )
                    source_upper = (
                        lower[axis]
                        + 2 * (target_upper[axis] - band_lower)
                    )
                else:
                    quotient = (
                        phase_bits[axis] + direction[axis]
                    ) // 2
                    source_phase = (
                        phase_bits[axis]
                        + direction[axis]
                        - 2 * quotient
                    )
                    source_band_lower = _checked_int64(
                        "coarse source band lower",
                        lower[axis]
                        + source_phase * half[axis],
                    )
                    target_band_lower = _checked_int64(
                        "coarse target band lower",
                        origin[axis] + direction[axis] * half[axis],
                    )
                    source_lower = source_band_lower + (
                        target_lower[axis] - target_band_lower
                    )
                    source_upper = source_band_lower + (
                        target_upper[axis] - target_band_lower
                    )
                source_lower = _checked_int64("plan source lower", source_lower)
                source_upper = _checked_int64("plan source upper", source_upper)
                expected_extent = (
                    2 if source_is_fine else 1
                ) * (target_upper[axis] - target_lower[axis])
                if (
                    source_lower < lower[axis]
                    or source_upper > upper[axis]
                    or source_upper - source_lower != expected_extent
                ):
                    raise ValueError("support source box is outside the block interior")
                source_lower_values.append(source_lower)
                source_upper_values.append(source_upper)
            transfers.append(
                {
                    "target_lower": target_lower,
                    "target_upper": target_upper,
                    "source_lower": tuple(source_lower_values),
                    "source_is_fine": source_is_fine,
                }
            )
        else:
            for axis in range(3):
                if mask & (1 << axis) and (
                    target_upper[axis] - target_lower[axis] != 1
                ):
                    raise ValueError("physical support target depth must be one")
            physical_records.append(
                {
                    "direction": direction,
                    "mask": mask,
                    "target_lower": target_lower,
                    "target_upper": target_upper,
                }
            )

    required_volume = _box_volume(required_lower, required_upper)
    workspace_volume = _box_volume(
        workspace_source_lower, workspace_source_upper
    )
    if raw_target_volume != required_volume or merged_volume != workspace_volume:
        raise ValueError("support target partition does not cover CWP required box")
    if 1 + len(outside_targets) > PLAN_CAPACITY:
        raise ValueError("support plan exceeds fixed 18-record capacity")

    for record in physical_records:
        direction = record["direction"]
        mask = int(record["mask"])
        target_lower = record["target_lower"]
        target_upper = record["target_upper"]
        base_lower = tuple(
            int(target_lower[axis])
            - (int(direction[axis]) if mask & (1 << axis) else 0)
            for axis in range(3)
        )
        base_upper = tuple(
            int(target_upper[axis])
            - (int(direction[axis]) if mask & (1 << axis) else 0)
            for axis in range(3)
        )
        owners = [
            owner
            for owner in transfers
            if _box_contains(
                owner["target_lower"],
                owner["target_upper"],
                base_lower,
                base_upper,
            )
        ]
        if len(owners) != 1:
            raise ValueError("physical support base owner is not unique")
        owner = owners[0]
        owner_source_lower = owner["source_lower"]
        owner_target_lower = owner["target_lower"]
        owner_is_fine = bool(owner["source_is_fine"])
        logical_lower = (0, 0, 0) if owner_is_fine else lower
        logical_upper = half if owner_is_fine else upper
        offsets: list[int] = []
        for axis in range(3):
            if owner_is_fine:
                delta = int(owner_source_lower[axis]) - lower[axis]
                if delta % 2:
                    raise ValueError("fine physical base source offset must be even")
                offset = delta // 2 - int(owner_target_lower[axis])
            else:
                offset = int(owner_source_lower[axis]) - int(
                    owner_target_lower[axis]
                )
            offset = _checked_int64("storage logical offset", offset)
            offsets.append(offset)
            mapped_lower = _checked_int64(
                "mapped physical target lower", int(target_lower[axis]) + offset
            )
            mapped_upper = _checked_int64(
                "mapped physical target upper", int(target_upper[axis]) + offset
            )
            if mask & (1 << axis):
                if int(direction[axis]) < 0:
                    if mapped_upper != logical_lower[axis]:
                        raise ValueError("physical lower target is not adjacent")
                elif mapped_lower != logical_upper[axis]:
                    raise ValueError("physical upper target is not adjacent")
        del offsets


def fill_coarser_slope_support_plan(
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
    plan_source_slots: np.ndarray,
    plan_source_is_fine: np.ndarray,
    plan_physical_masks: np.ndarray,
    plan_directions: np.ndarray,
    plan_source_lower: np.ndarray,
    plan_source_upper: np.ndarray,
    plan_base_lower: np.ndarray,
    plan_base_upper: np.ndarray,
    plan_target_lower: np.ndarray,
    plan_target_upper: np.ndarray,
    plan_logical_interior_lower: np.ndarray,
    plan_logical_interior_upper: np.ndarray,
    plan_storage_logical_offsets: np.ndarray,
) -> tuple[int, int]:
    """Fill one exact fixed-capacity CSP plan after complete preflight."""
    interior_lower = _require_triplet("interior_lower", interior_lower)
    interior_upper = _require_triplet("interior_upper", interior_upper)
    selected_count = _require_nonnegative_integer("selected_count", selected_count)
    primary_slot = _require_nonnegative_integer("primary_slot", primary_slot)
    primary_phase_code = _require_nonnegative_integer(
        "primary_phase_code", primary_phase_code
    )
    reduced_direction = _require_triplet("reduced_direction", reduced_direction)
    directions = _require_array(
        "directions", directions, INDEX_DTYPE, (26, 3)
    )
    relation_kinds = _require_array(
        "relation_kinds", relation_kinds, np.dtype(np.uint8), (26,)
    )
    physical_masks = _require_array(
        "physical_masks", physical_masks, np.dtype(np.uint8), (26,)
    )
    source_counts = _require_array(
        "source_counts", source_counts, np.dtype(np.uint8), (26,)
    )
    source_slots = _require_array(
        "source_slots", source_slots, INDEX_DTYPE, (26, 4)
    )
    box_inputs = tuple(
        _require_triplet(name, value)
        for name, value in (
            ("fine_target_lower", fine_target_lower),
            ("fine_target_upper", fine_target_upper),
            ("coarse_source_lower", coarse_source_lower),
            ("coarse_source_upper", coarse_source_upper),
            ("workspace_source_lower", workspace_source_lower),
            ("workspace_source_upper", workspace_source_upper),
            ("workspace_required_lower", workspace_required_lower),
            ("workspace_required_upper", workspace_required_upper),
            ("workspace_coarse_origin", workspace_coarse_origin),
        )
    )
    (
        fine_target_lower,
        fine_target_upper,
        coarse_source_lower,
        coarse_source_upper,
        workspace_source_lower,
        workspace_source_upper,
        workspace_required_lower,
        workspace_required_upper,
        workspace_coarse_origin,
    ) = box_inputs

    output_specs = (
        ("plan_source_slots", plan_source_slots, INDEX_DTYPE, (18,)),
        (
            "plan_source_is_fine",
            plan_source_is_fine,
            np.dtype(np.uint8),
            (18,),
        ),
        (
            "plan_physical_masks",
            plan_physical_masks,
            np.dtype(np.uint8),
            (18,),
        ),
        ("plan_directions", plan_directions, INDEX_DTYPE, (18, 3)),
        ("plan_source_lower", plan_source_lower, INDEX_DTYPE, (18, 3)),
        ("plan_source_upper", plan_source_upper, INDEX_DTYPE, (18, 3)),
        ("plan_base_lower", plan_base_lower, INDEX_DTYPE, (18, 3)),
        ("plan_base_upper", plan_base_upper, INDEX_DTYPE, (18, 3)),
        ("plan_target_lower", plan_target_lower, INDEX_DTYPE, (18, 3)),
        ("plan_target_upper", plan_target_upper, INDEX_DTYPE, (18, 3)),
        (
            "plan_logical_interior_lower",
            plan_logical_interior_lower,
            INDEX_DTYPE,
            (18, 3),
        ),
        (
            "plan_logical_interior_upper",
            plan_logical_interior_upper,
            INDEX_DTYPE,
            (18, 3),
        ),
        (
            "plan_storage_logical_offsets",
            plan_storage_logical_offsets,
            INDEX_DTYPE,
            (18, 3),
        ),
    )
    outputs = tuple(
        _require_array(name, value, dtype, shape, writable=True)
        for name, value, dtype, shape in output_specs
    )

    if primary_phase_code > 7:
        raise ValueError("primary_phase_code must be in [0,7]")
    if primary_slot >= selected_count:
        raise ValueError("primary_slot must be inside selected slots")
    lower = tuple(int(value) for value in interior_lower)
    upper = tuple(int(value) for value in interior_upper)
    reduced = tuple(int(value) for value in reduced_direction)
    for axis in range(3):
        block = upper[axis] - lower[axis]
        if lower[axis] < 0 or block < 4 or block % 2:
            raise ValueError("interior extents must be positive even values at least four")
        if reduced[axis] < -1 or reduced[axis] > 1:
            raise ValueError("reduced direction component is outside [-1,1]")
    if reduced == (0, 0, 0):
        raise ValueError("reduced direction must be noncenter")

    _validate_relation_row(
        directions,
        relation_kinds,
        physical_masks,
        source_counts,
        source_slots,
        selected_count,
    )
    reduced_row = _direction_index(reduced)
    if (
        int(relation_kinds[reduced_row]) != RELATION_COARSER
        or int(physical_masks[reduced_row]) != 0
        or int(source_counts[reduced_row]) != 1
    ):
        raise ValueError("reduced direction must identify an unmasked COARSER row")

    target_lower = tuple(int(value) for value in fine_target_lower)
    target_upper = tuple(int(value) for value in fine_target_upper)
    for axis in range(3):
        block = upper[axis] - lower[axis]
        half = block // 2
        start = target_lower[axis]
        stop = target_upper[axis]
        component = reduced[axis]
        if start < 0 or start >= stop:
            raise ValueError("fine target must be nonempty, ordered, and nonnegative")
        if component == 0:
            valid = start == lower[axis] and stop == upper[axis]
        elif component < 0:
            valid = (
                stop == lower[axis]
                and lower[axis] - half <= start < lower[axis]
            )
        else:
            valid = (
                start == upper[axis]
                and upper[axis] < stop <= upper[axis] + half
            )
        if not valid:
            raise ValueError("fine target exceeds the active CWP B/2 reach gate")

    expected_cwp = _expected_cwp(
        lower,
        upper,
        reduced,
        primary_phase_code,
        target_lower,
        target_upper,
    )
    actual_cwp = tuple(
        tuple(int(value) for value in array)
        for array in (
            coarse_source_lower,
            coarse_source_upper,
            workspace_source_lower,
            workspace_source_upper,
            workspace_required_lower,
            workspace_required_upper,
            workspace_coarse_origin,
        )
    )
    if expected_cwp != actual_cwp:
        raise ValueError("supplied boxes are not the exact CWP row")

    _validate_plan_geometry(
        lower,
        upper,
        primary_slot,
        primary_phase_code,
        reduced,
        relation_kinds,
        physical_masks,
        source_slots,
        actual_cwp[0],
        actual_cwp[1],
        actual_cwp[2],
        actual_cwp[3],
        actual_cwp[4],
        actual_cwp[5],
        actual_cwp[6],
    )

    inputs = (
        interior_lower,
        interior_upper,
        reduced_direction,
        directions,
        relation_kinds,
        physical_masks,
        source_counts,
        source_slots,
        *box_inputs,
    )
    if any(
        np.shares_memory(output, value) for output in outputs for value in inputs
    ) or any(
        np.shares_memory(left, right) for left, right in combinations(outputs, 2)
    ):
        raise ValueError("plan outputs must not overlap inputs or each other")

    transfer_count, record_count = fill_coarser_slope_support_plan_unchecked(
        interior_lower,
        interior_upper,
        primary_slot,
        primary_phase_code,
        reduced_direction,
        relation_kinds,
        physical_masks,
        source_slots,
        coarse_source_lower,
        coarse_source_upper,
        workspace_source_lower,
        workspace_source_upper,
        workspace_required_lower,
        workspace_required_upper,
        workspace_coarse_origin,
        *outputs,
    )
    return int(transfer_count), int(record_count)
