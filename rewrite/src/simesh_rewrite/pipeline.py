"""Concrete two-pass bounded executor for Cartesian level-1 M0."""

from __future__ import annotations

import numpy as np

from ._chunking import minimum_halo_closed_slots_unchecked
from ._geometry import validate_selected_geometry_unchecked
from ._halos import (
    duplicate_block_id_index_unchecked,
    missing_halo_closure_primary_unchecked,
)
from ._pipeline import validate_level1_mor_top_unchecked
from ._sampling import (
    validate_selected_level1_placement_unchecked,
    validate_trilinear_stencils_unchecked,
)
from .chunking import plan_level1_chunk, plan_level1_halo_chunk
from .foundation import _require_index_triplet, _require_payload
from .geometry import _require_float_triplet
from .halos import fill_physical_halos, fill_same_level_halos
from .morton import _root_volume
from .operators import central_difference_into, scaled_difference_into
from .reductions import accumulate_field_sum, finalize_field_sum
from .sampling import (
    _require_uniform_grid,
    _validated_spacing,
    place_level1_blocks,
    sample_level1_trilinear,
    sample_level1_zero_order,
)
from .storage import (
    _require_index_vector,
    gather_blocks_into,
    scatter_blocks_from,
)
from .topology import _require_mapping_input


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _require_budget(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("budget_bytes must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("budget_bytes must be nonnegative")
    if value > _INDEX_MAX:
        raise OverflowError("budget_bytes does not fit in int64")
    return value


def _require_face_table(face_neighbor_ids: np.ndarray, block_count: int) -> np.ndarray:
    if not isinstance(face_neighbor_ids, np.ndarray):
        raise TypeError("face_neighbor_ids must be a NumPy array")
    if face_neighbor_ids.dtype != np.dtype(np.int64):
        raise TypeError("face_neighbor_ids must have dtype int64")
    if face_neighbor_ids.shape != (block_count, 6):
        raise ValueError(
            f"face_neighbor_ids must have shape {(block_count, 6)}, "
            f"got {face_neighbor_ids.shape}"
        )
    if not face_neighbor_ids.flags.c_contiguous:
        raise ValueError("face_neighbor_ids must be C-contiguous")
    return face_neighbor_ids


def _product(name: str, shape: np.ndarray) -> int:
    result = 1
    for value in shape:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} entries must be positive")
        if result > _INDEX_MAX // value:
            raise OverflowError(f"{name} volume does not fit in int64")
        result *= value
    return result


def _require_result_backing(
    name: str,
    value: np.ndarray,
    block_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    value = _require_payload(name, value, writable=True)
    expected = (block_count, 1, *block_shape)
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    return value


def _validate_result_nonoverlap(
    backing: np.ndarray,
    results: tuple[np.ndarray, ...],
    metadata: tuple[np.ndarray, ...],
) -> None:
    for index, result in enumerate(results):
        if np.shares_memory(result, backing) or any(
            np.shares_memory(result, value) for value in metadata
        ):
            raise ValueError("pipeline results must not overlap inputs or metadata")
        for other in results[index + 1 :]:
            if np.shares_memory(result, other):
                raise ValueError("pipeline results must be pairwise nonoverlapping")


def execute_level1_m0(
    backing: np.ndarray,
    field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    pointwise_left_field: int,
    pointwise_right_field: int,
    pointwise_scale: float,
    stencil_field: int,
    stencil_axis: int,
    reduction_field: int,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    budget_bytes: int,
    pointwise_backing: np.ndarray,
    stencil_backing: np.ndarray,
    native_grid: np.ndarray,
    zero_grid: np.ndarray,
    trilinear_grid: np.ndarray,
) -> tuple[float, int]:
    """Execute the complete bounded Cartesian level-1 M0 workflow."""
    budget_bytes = _require_budget(budget_bytes)
    backing = _require_payload("backing", backing, writable=False)
    field_ids = _require_index_vector("field_ids", field_ids)
    if field_ids.shape[0] == 0:
        raise ValueError("field_ids must select at least one field")
    domain_lower = _require_float_triplet("domain_lower", domain_lower)
    domain_upper = _require_float_triplet("domain_upper", domain_upper)
    domain_cell_counts = _require_index_triplet(
        "domain_cell_counts", domain_cell_counts
    )
    block_cell_counts = _require_index_triplet(
        "block_cell_counts", block_cell_counts
    )
    if np.any(domain_cell_counts <= 0) or np.any(block_cell_counts <= 0):
        raise ValueError("cell counts must be positive")
    if np.any(domain_cell_counts % block_cell_counts != 0):
        raise ValueError("domain_cell_counts must be divisible by block_cell_counts")
    root_shape = np.ascontiguousarray(
        domain_cell_counts // block_cell_counts,
        dtype=np.int64,
    )
    block_count = _root_volume(root_shape)
    if backing.shape[0] != block_count:
        raise ValueError("backing block axis must equal root-grid volume")
    block_shape = tuple(int(value) for value in block_cell_counts)
    if backing.shape[2:] != block_shape:
        raise ValueError("backing spatial shape must equal block_cell_counts")
    coord_to_rank = _require_mapping_input(
        "coord_to_rank",
        coord_to_rank,
        tuple(int(value) for value in root_shape),
    )
    rank_to_coord = _require_mapping_input(
        "rank_to_coord",
        rank_to_coord,
        (block_count, 3),
    )
    face_neighbor_ids = _require_face_table(face_neighbor_ids, block_count)

    pointwise_backing = _require_result_backing(
        "pointwise_backing",
        pointwise_backing,
        block_count,
        block_shape,
    )
    stencil_backing = _require_result_backing(
        "stencil_backing",
        stencil_backing,
        block_count,
        block_shape,
    )
    native_grid = _require_uniform_grid(native_grid)
    zero_grid = _require_uniform_grid(zero_grid)
    trilinear_grid = _require_uniform_grid(trilinear_grid)
    if (
        native_grid.shape[0] != field_ids.shape[0]
        or zero_grid.shape[0] != field_ids.shape[0]
        or trilinear_grid.shape[0] != field_ids.shape[0]
    ):
        raise ValueError("all grid field extents must match field_ids")

    sample_lower = _require_float_triplet("sample_lower", sample_lower)
    sample_upper = _require_float_triplet("sample_upper", sample_upper)
    normal_field_slots = _require_index_triplet(
        "normal_field_slots",
        normal_field_slots,
    )
    metadata = (
        field_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
        sample_lower,
        sample_upper,
    )
    results = (
        pointwise_backing,
        stencil_backing,
        native_grid,
        zero_grid,
        trilinear_grid,
    )
    _validate_result_nonoverlap(backing, results, metadata)

    if np.any(block_cell_counts > _INDEX_MAX - 2):
        raise OverflowError("padded block shape does not fit in int64")
    padded_shape = np.ascontiguousarray(block_cell_counts + 2, dtype=np.int64)
    block_volume = _product("block_cell_counts", block_cell_counts)
    padded_volume = _product("padded_shape", padded_shape)
    field_count = int(field_ids.shape[0])
    values_per_slot = field_count * padded_volume + block_volume + 1
    if values_per_slot > _INDEX_MAX // 8:
        raise OverflowError("managed bytes per slot do not fit in int64")
    bytes_per_slot = 8 * values_per_slot

    zero = np.zeros(3, dtype=np.int64)
    interior_lower = np.ones(3, dtype=np.int64)
    interior_upper = np.ascontiguousarray(
        interior_lower + block_cell_counts,
        dtype=np.int64,
    )
    field_zero = np.array([0], dtype=np.int64)
    cell_spacing = np.ascontiguousarray(
        (domain_upper - domain_lower) / domain_cell_counts,
        dtype=np.float64,
    )
    empty_ids = np.empty(0, dtype=np.int64)
    empty_payload = np.empty(
        (0, field_count, *(int(value) for value in padded_shape)),
        dtype=np.float64,
    )
    empty_output = np.empty((0, 1, *block_shape), dtype=np.float64)
    preflight_state = np.array([0.0], dtype=np.float64)

    # Fixed lower-boundary preflight before topology proof or bulk allocation.
    gather_blocks_into(
        backing,
        zero,
        block_cell_counts,
        empty_ids,
        field_ids,
        empty_payload,
        interior_lower,
    )
    scaled_difference_into(
        empty_payload,
        interior_lower,
        interior_upper,
        pointwise_left_field,
        pointwise_right_field,
        pointwise_scale,
        empty_output,
        0,
        zero,
    )
    scatter_blocks_from(
        empty_output,
        zero,
        block_cell_counts,
        empty_ids,
        field_zero,
        pointwise_backing,
        zero,
    )
    place_level1_blocks(
        empty_payload,
        interior_lower,
        interior_upper,
        empty_ids,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
        native_grid,
    )
    sample_level1_zero_order(
        empty_payload,
        interior_lower,
        interior_upper,
        empty_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        zero_grid,
    )
    accumulate_field_sum(
        empty_payload,
        interior_lower,
        interior_upper,
        reduction_field,
        preflight_state,
    )
    fill_physical_halos(
        empty_payload,
        interior_lower,
        interior_upper,
        empty_ids,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )
    fill_same_level_halos(
        empty_payload,
        interior_lower,
        interior_upper,
        empty_ids,
        0,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )
    central_difference_into(
        empty_payload,
        zero,
        padded_shape,
        interior_lower,
        interior_upper,
        stencil_field,
        stencil_axis,
        cell_spacing,
        empty_output,
        0,
        zero,
    )
    scatter_blocks_from(
        empty_output,
        zero,
        block_cell_counts,
        empty_ids,
        field_zero,
        stencil_backing,
        zero,
    )
    sample_level1_trilinear(
        empty_payload,
        zero,
        padded_shape,
        interior_lower,
        interior_upper,
        empty_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        trilinear_grid,
    )

    native_spacing = _validated_spacing(
        "native",
        domain_lower,
        domain_upper,
        domain_cell_counts,
    )
    trilinear_counts = np.ascontiguousarray(
        trilinear_grid.shape[1:],
        dtype=np.int64,
    )
    trilinear_spacing = _validated_spacing(
        "output",
        sample_lower,
        sample_upper,
        trilinear_counts,
    )

    invalid_topology = int(
        validate_level1_mor_top_unchecked(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            face_neighbor_ids,
        )
    )
    if invalid_topology >= 0:
        raise ValueError(
            f"invalid Morton/topology semantics at block {invalid_topology}"
        )

    capacity = 0 if budget_bytes < 8 else min(
        block_count,
        (budget_bytes - 8) // bytes_per_slot,
    )
    minimum_capacity = int(
        minimum_halo_closed_slots_unchecked(face_neighbor_ids)
    )
    if capacity < minimum_capacity:
        raise ValueError(
            "budget cannot fit the minimum full-halo closure and reduction state"
        )

    ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty(
        (capacity, field_count, *(int(value) for value in padded_shape)),
        dtype=np.float64,
    )
    output_workspace = np.empty(
        (capacity, 1, *block_shape),
        dtype=np.float64,
    )
    sum_state = np.array([0.0], dtype=np.float64)
    managed_bytes = (
        ids.nbytes + payload.nbytes + output_workspace.nbytes + sum_state.nbytes
    )
    if managed_bytes != 8 * capacity * values_per_slot + 8:
        raise RuntimeError("internal managed-byte accounting mismatch")
    if managed_bytes > budget_bytes:
        raise RuntimeError("internal workspace exceeds budget")

    # Actual-ID dry planning and metadata validation.
    first = 0
    while first < block_count:
        primary_count, _ = plan_level1_chunk(
            first,
            face_neighbor_ids,
            False,
            ids,
        )
        primary_ids = ids[:primary_count]
        if validate_selected_level1_placement_unchecked(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            primary_ids,
        ) >= 0 or validate_selected_geometry_unchecked(
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            rank_to_coord,
            primary_ids,
            native_spacing,
        ) >= 0:
            raise ValueError("invalid selected geometry in no-closure preflight")
        first += primary_count

    first = 0
    while first < block_count:
        primary_count, selected_count = plan_level1_halo_chunk(
            first,
            face_neighbor_ids,
            ids,
        )
        selected_ids = ids[:selected_count]
        primary_ids = ids[:primary_count]
        if duplicate_block_id_index_unchecked(selected_ids) >= 0:
            raise ValueError("full-halo preflight selected duplicate block IDs")
        if missing_halo_closure_primary_unchecked(
            primary_count,
            selected_ids,
            face_neighbor_ids,
        ) >= 0:
            raise ValueError("full-halo preflight lacks primary closure")
        if validate_selected_geometry_unchecked(
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            rank_to_coord,
            primary_ids,
            native_spacing,
        ) >= 0:
            raise ValueError("invalid selected geometry in full-halo preflight")
        if validate_trilinear_stencils_unchecked(
            primary_ids,
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            rank_to_coord,
            native_spacing,
            sample_lower,
            trilinear_spacing,
            trilinear_grid,
        ) >= 0:
            raise ValueError("invalid trilinear stencil in full-halo preflight")
        first += primary_count

    # Pass one.
    sum_state[0] = 0.0
    first = 0
    while first < block_count:
        primary_count, selected_count = plan_level1_chunk(
            first,
            face_neighbor_ids,
            False,
            ids,
        )
        primary_ids = ids[:primary_count]
        gather_blocks_into(
            backing,
            zero,
            block_cell_counts,
            primary_ids,
            field_ids,
            payload[:primary_count],
            interior_lower,
        )
        scaled_difference_into(
            payload[:primary_count],
            interior_lower,
            interior_upper,
            pointwise_left_field,
            pointwise_right_field,
            pointwise_scale,
            output_workspace[:primary_count],
            0,
            zero,
        )
        scatter_blocks_from(
            output_workspace[:primary_count],
            zero,
            block_cell_counts,
            primary_ids,
            field_zero,
            pointwise_backing,
            zero,
        )
        place_level1_blocks(
            payload[:primary_count],
            interior_lower,
            interior_upper,
            primary_ids,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            rank_to_coord,
            native_grid,
        )
        sample_level1_zero_order(
            payload[:primary_count],
            interior_lower,
            interior_upper,
            primary_ids,
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            zero_grid,
        )
        accumulate_field_sum(
            payload[:primary_count],
            interior_lower,
            interior_upper,
            reduction_field,
            sum_state,
        )
        first += primary_count

    # Pass two.
    first = 0
    while first < block_count:
        primary_count, selected_count = plan_level1_halo_chunk(
            first,
            face_neighbor_ids,
            ids,
        )
        selected_ids = ids[:selected_count]
        primary_ids = ids[:primary_count]
        gather_blocks_into(
            backing,
            zero,
            block_cell_counts,
            selected_ids,
            field_ids,
            payload[:selected_count],
            interior_lower,
        )
        fill_physical_halos(
            payload[:selected_count],
            interior_lower,
            interior_upper,
            selected_ids,
            face_neighbor_ids,
            boundary_modes,
            normal_field_slots,
        )
        fill_same_level_halos(
            payload[:selected_count],
            interior_lower,
            interior_upper,
            selected_ids,
            primary_count,
            face_neighbor_ids,
            boundary_modes,
            normal_field_slots,
        )
        central_difference_into(
            payload[:primary_count],
            zero,
            padded_shape,
            interior_lower,
            interior_upper,
            stencil_field,
            stencil_axis,
            cell_spacing,
            output_workspace[:primary_count],
            0,
            zero,
        )
        scatter_blocks_from(
            output_workspace[:primary_count],
            zero,
            block_cell_counts,
            primary_ids,
            field_zero,
            stencil_backing,
            zero,
        )
        sample_level1_trilinear(
            payload[:primary_count],
            zero,
            padded_shape,
            interior_lower,
            interior_upper,
            primary_ids,
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            trilinear_grid,
        )
        first += primary_count

    return finalize_field_sum(sum_state), capacity
