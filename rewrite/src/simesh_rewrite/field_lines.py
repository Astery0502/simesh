"""SLE-001 cached native refined field-line execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ._field_line_rhs import field_line_rhs_unchecked
from ._rk4 import (
    field_line_rk4_finish_unchecked,
    field_line_rk4_stage_unchecked,
)
from .completed_halo_sampling import (
    CachedVectorSamplingStats,
    CompletedHaloSamplingSession,
    UnrepresentableRefinedSampleError,
    _require_session,
    _sample_refined_trilinear_vectors_cached,
)
from .field_line_rhs import _require_direction_signs
from .field_line_termination import (
    FieldLineStage,
    FieldLineTermination,
    _classify_field_line_candidate_unchecked,
    _classify_field_line_stage_state_unchecked,
    _field_line_termination_from_rhs_status_unchecked,
)
from .rk4 import _require_step_size


_INDEX_MAX = int(np.iinfo(np.int64).max)


class FieldLineExecutionStats(NamedTuple):
    seed_count: int
    interior_seed_count: int
    accepted_point_count: int
    attempted_step_count: int
    accepted_step_count: int
    stage_sample_count: int
    sampler_call_count: int
    sampler_preflight_rejection_count: int
    hint_candidate_count: int
    hint_hit_count: int
    hierarchy_fallback_count: int
    cache_lookup_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_eviction_count: int
    halo_fill_count: int
    reader_call_count: int
    selected_load_count: int
    support_load_count: int
    maximum_selected_slots: int
    logical_reader_bytes: int
    seed_outside_count: int
    max_steps_count: int
    domain_exit_count: int
    zero_field_count: int
    nonfinite_field_count: int
    unrepresentable_norm_count: int
    unrepresentable_sample_count: int
    nonfinite_state_count: int
    no_progress_count: int
    sampler_call_managed_peak_bytes: int
    executor_managed_array_bytes: int
    session_managed_array_bytes: int


@dataclass(slots=True)
class _ExecutionCounters:
    attempted_steps: int = 0
    accepted_steps: int = 0
    stage_samples: int = 0
    sampler_calls: int = 0
    sampler_rejections: int = 0
    hint_candidates: int = 0
    hint_hits: int = 0
    hierarchy_fallbacks: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    halo_fills: int = 0
    reader_calls: int = 0
    selected_loads: int = 0
    support_loads: int = 0
    maximum_selected_slots: int = 0
    logical_reader_bytes: int = 0
    sampler_managed_peak: int = 0


def _checked_add(name: str, left: int, right: int) -> int:
    result = left + right
    if result < 0 or result > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return result


def _checked_multiply(name: str, left: int, right: int) -> int:
    if left < 0 or right < 0:
        raise ValueError(f"{name} factors must be nonnegative")
    if left and right > _INDEX_MAX // left:
        raise OverflowError(f"{name} does not fit in int64")
    return left * right


def _require_max_steps(value: int) -> int:
    if type(value) is not int:
        raise TypeError("max_steps must be an exact Python int")
    if value < 0:
        raise ValueError("max_steps must be nonnegative")
    if value >= _INDEX_MAX:
        raise OverflowError("max_steps + 1 does not fit in int64")
    return value


def _require_seeds(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("seeds must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise TypeError("seeds must have dtype float64")
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"seeds must have shape (seed, 3), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError("seeds must be C-contiguous")
    for seed in range(value.shape[0]):
        for axis in range(3):
            if not np.isfinite(float(value[seed, axis])):
                raise ValueError(
                    f"seed coordinates must be finite; seed {seed}, axis {axis} is not"
                )
    return value


def _require_output(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _validate_outputs_and_aliases(
    state,
    seeds: np.ndarray,
    direction_signs: np.ndarray,
    max_steps: int,
    positions: np.ndarray,
    integrals: np.ndarray,
    point_counts: np.ndarray,
    termination_codes: np.ndarray,
    termination_stages: np.ndarray,
) -> tuple[np.ndarray, ...]:
    seed_count = int(seeds.shape[0])
    point_capacity = max_steps + 1
    positions = _require_output(
        "positions",
        positions,
        np.dtype(np.float64),
        (seed_count, point_capacity, 3),
    )
    integrals = _require_output(
        "integrals",
        integrals,
        np.dtype(np.float64),
        (seed_count, point_capacity),
    )
    point_counts = _require_output(
        "point_counts", point_counts, np.dtype(np.int64), (seed_count,)
    )
    termination_codes = _require_output(
        "termination_codes",
        termination_codes,
        np.dtype(np.uint8),
        (seed_count,),
    )
    termination_stages = _require_output(
        "termination_stages",
        termination_stages,
        np.dtype(np.uint8),
        (seed_count,),
    )
    outputs = (
        positions,
        integrals,
        point_counts,
        termination_codes,
        termination_stages,
    )
    dynamic_inputs = (seeds, direction_signs)
    for dynamic_input in dynamic_inputs:
        if any(
            np.shares_memory(dynamic_input, value)
            for value in state.owned_arrays
        ):
            raise ValueError(
                "field-line inputs must not overlap mutable session memory"
            )
    readonly = (*dynamic_inputs, *state.borrowed_arrays, *state.owned_arrays)
    for index, output in enumerate(outputs):
        if any(np.shares_memory(output, value) for value in readonly):
            raise ValueError(
                "field-line outputs must not overlap inputs or session memory"
            )
        if any(np.shares_memory(output, prior) for prior in outputs[:index]):
            raise ValueError("field-line outputs must be pairwise nonoverlapping")
    return outputs


def _copy_compact_row(array: np.ndarray, source: int, target: int) -> None:
    if source == target:
        return
    if array.ndim == 1:
        array[target] = array[source]
        return
    for column in range(array.shape[1]):
        array[target, column] = array[source, column]


def _mark_termination(
    seed_id: int,
    code: int,
    stage: int,
    termination_codes: np.ndarray,
    termination_stages: np.ndarray,
) -> None:
    termination_codes[seed_id] = code
    termination_stages[seed_id] = stage


def _compact_stage_states(
    active_ids: np.ndarray,
    active_count: int,
    stage_states: np.ndarray,
    carried: tuple[np.ndarray, ...],
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    stage_code: int,
    termination_codes: np.ndarray,
    termination_stages: np.ndarray,
) -> int:
    write = 0
    arrays = (*carried, stage_states)
    for row in range(active_count):
        code = _classify_field_line_stage_state_unchecked(
            stage_states[row], domain_lower, domain_upper
        )
        seed_id = int(active_ids[row])
        if code == int(FieldLineTermination.ACTIVE):
            if write != row:
                active_ids[write] = seed_id
                for array in arrays:
                    _copy_compact_row(array, row, write)
            write += 1
        else:
            _mark_termination(
                seed_id,
                code,
                stage_code,
                termination_codes,
                termination_stages,
            )
    return write


def _remove_compact_row(
    active_ids: np.ndarray,
    active_count: int,
    remove: int,
    arrays: tuple[np.ndarray, ...],
) -> int:
    for row in range(remove, active_count - 1):
        active_ids[row] = active_ids[row + 1]
        for array in arrays:
            _copy_compact_row(array, row + 1, row)
    return active_count - 1


def _accumulate_sampling_stats(
    counters: _ExecutionCounters,
    stats: CachedVectorSamplingStats,
) -> None:
    counters.stage_samples = _checked_add(
        "stage sample count", counters.stage_samples, stats.inside_point_count
    )
    counters.hint_candidates = _checked_add(
        "hint candidate count", counters.hint_candidates, stats.hint_candidate_count
    )
    counters.hint_hits = _checked_add(
        "hint hit count", counters.hint_hits, stats.hint_hit_count
    )
    counters.hierarchy_fallbacks = _checked_add(
        "hierarchy fallback count",
        counters.hierarchy_fallbacks,
        stats.hierarchy_fallback_count,
    )
    counters.cache_lookups = _checked_add(
        "cache lookup count", counters.cache_lookups, stats.cache_lookup_count
    )
    counters.cache_hits = _checked_add(
        "cache hit count", counters.cache_hits, stats.cache_hit_count
    )
    counters.cache_misses = _checked_add(
        "cache miss count", counters.cache_misses, stats.cache_miss_count
    )
    counters.cache_evictions = _checked_add(
        "cache eviction count",
        counters.cache_evictions,
        stats.cache_eviction_count,
    )
    counters.halo_fills = _checked_add(
        "halo fill count", counters.halo_fills, stats.halo_fill_count
    )
    counters.reader_calls = _checked_add(
        "reader call count", counters.reader_calls, stats.reader_call_count
    )
    counters.selected_loads = _checked_add(
        "selected load count", counters.selected_loads, stats.selected_load_count
    )
    counters.support_loads = _checked_add(
        "support load count", counters.support_loads, stats.support_load_count
    )
    counters.maximum_selected_slots = max(
        counters.maximum_selected_slots, stats.maximum_selected_slots
    )
    counters.logical_reader_bytes = _checked_add(
        "logical reader bytes",
        counters.logical_reader_bytes,
        stats.logical_reader_bytes,
    )
    counters.sampler_managed_peak = max(
        counters.sampler_managed_peak, stats.call_managed_array_bytes
    )


def _prepare_sample_arrays(
    active_ids: np.ndarray,
    active_count: int,
    stage_states: np.ndarray,
    last_owner_ids: np.ndarray,
    direction_signs: np.ndarray,
    sample_points: np.ndarray,
    sample_hints: np.ndarray,
    sample_signs: np.ndarray,
) -> None:
    for row in range(active_count):
        seed_id = int(active_ids[row])
        for axis in range(3):
            sample_points[row, axis] = stage_states[row, axis]
        sample_hints[row] = last_owner_ids[seed_id]
        sample_signs[row] = direction_signs[seed_id]


def _sample_rhs_stage(
    state,
    active_ids: np.ndarray,
    active_count: int,
    stage_states: np.ndarray,
    carried_before_rhs: tuple[np.ndarray, ...],
    target_rhs: np.ndarray,
    last_owner_ids: np.ndarray,
    direction_signs: np.ndarray,
    sample_points: np.ndarray,
    sample_hints: np.ndarray,
    sample_values: np.ndarray,
    sample_owner_ids: np.ndarray,
    sample_signs: np.ndarray,
    rhs_statuses: np.ndarray,
    stage_code: int,
    termination_codes: np.ndarray,
    termination_stages: np.ndarray,
    counters: _ExecutionCounters,
) -> int:
    active_count = _compact_stage_states(
        active_ids,
        active_count,
        stage_states,
        carried_before_rhs,
        state.domain_lower,
        state.domain_upper,
        stage_code,
        termination_codes,
        termination_stages,
    )
    carried_with_state = (*carried_before_rhs, stage_states)
    while active_count:
        _prepare_sample_arrays(
            active_ids,
            active_count,
            stage_states,
            last_owner_ids,
            direction_signs,
            sample_points,
            sample_hints,
            sample_signs,
        )
        counters.sampler_calls = _checked_add(
            "sampler call count", counters.sampler_calls, 1
        )
        try:
            stats = _sample_refined_trilinear_vectors_cached(
                state,
                sample_points[:active_count],
                sample_hints[:active_count],
                sample_values[:active_count],
                sample_owner_ids[:active_count],
            )
        except UnrepresentableRefinedSampleError as error:
            bad_row = error.point_index
            if bad_row < 0 or bad_row >= active_count:
                raise RuntimeError(
                    "CHS reported an invalid compact point index"
                ) from error
            seed_id = int(active_ids[bad_row])
            _mark_termination(
                seed_id,
                int(FieldLineTermination.UNREPRESENTABLE_SAMPLE),
                stage_code,
                termination_codes,
                termination_stages,
            )
            counters.sampler_rejections = _checked_add(
                "sampler preflight rejection count",
                counters.sampler_rejections,
                1,
            )
            counters.sampler_managed_peak = max(
                counters.sampler_managed_peak,
                error.call_managed_array_bytes,
            )
            active_count = _remove_compact_row(
                active_ids,
                active_count,
                bad_row,
                carried_with_state,
            )
            continue
        _accumulate_sampling_stats(counters, stats)
        break

    if not active_count:
        return 0

    for row in range(active_count):
        seed_id = int(active_ids[row])
        last_owner_ids[seed_id] = sample_owner_ids[row]
    field_line_rhs_unchecked(
        sample_values[:active_count],
        sample_signs[:active_count],
        target_rhs[:active_count],
        rhs_statuses[:active_count],
    )

    write = 0
    carried_after_rhs = (*carried_before_rhs, target_rhs)
    for row in range(active_count):
        rhs_status = int(rhs_statuses[row])
        code = _field_line_termination_from_rhs_status_unchecked(rhs_status)
        seed_id = int(active_ids[row])
        if code == int(FieldLineTermination.ACTIVE):
            if write != row:
                active_ids[write] = seed_id
                for array in carried_after_rhs:
                    _copy_compact_row(array, row, write)
            write += 1
        else:
            _mark_termination(
                seed_id,
                code,
                stage_code,
                termination_codes,
                termination_stages,
            )
    return write


def execute_refined_field_lines(
    sampling_session: CompletedHaloSamplingSession,
    seeds: np.ndarray,
    direction_signs: np.ndarray,
    step_size: float,
    max_steps: int,
    positions: np.ndarray,
    integrals: np.ndarray,
    point_counts: np.ndarray,
    termination_codes: np.ndarray,
    termination_stages: np.ndarray,
) -> FieldLineExecutionStats:
    """Execute fixed-step field lines through one explicit CHS session."""
    state = _require_session(sampling_session)
    if state.active:
        raise ValueError("completed halo sampling session is already active")
    seeds = _require_seeds(seeds)
    seed_count = int(seeds.shape[0])
    direction_signs = _require_direction_signs(direction_signs, seed_count)
    step_size = _require_step_size(step_size)
    max_steps = _require_max_steps(max_steps)
    (
        positions,
        integrals,
        point_counts,
        termination_codes,
        termination_stages,
    ) = _validate_outputs_and_aliases(
        state,
        seeds,
        direction_signs,
        max_steps,
        positions,
        integrals,
        point_counts,
        termination_codes,
        termination_stages,
    )

    expected_executor_bytes = _checked_multiply(
        "executor managed raw-array bytes", seed_count, 314
    )
    active_ids = np.empty(seed_count, dtype=np.int64)
    next_active_ids = np.empty(seed_count, dtype=np.int64)
    current_states = np.empty((seed_count, 4), dtype=np.float64)
    last_owner_ids = np.full(seed_count, -1, dtype=np.int64)
    base_states = np.empty((seed_count, 4), dtype=np.float64)
    k1 = np.empty_like(base_states)
    k2 = np.empty_like(base_states)
    k3 = np.empty_like(base_states)
    k4 = np.empty_like(base_states)
    stage_states = np.empty_like(base_states)
    sample_points = np.empty((seed_count, 3), dtype=np.float64)
    sample_values = np.empty_like(sample_points)
    sample_hints = np.empty(seed_count, dtype=np.int64)
    sample_owner_ids = np.empty(seed_count, dtype=np.int64)
    sample_signs = np.empty(seed_count, dtype=np.int8)
    rhs_statuses = np.empty(seed_count, dtype=np.uint8)
    managed_arrays = (
        active_ids,
        next_active_ids,
        current_states,
        last_owner_ids,
        base_states,
        k1,
        k2,
        k3,
        k4,
        stage_states,
        sample_points,
        sample_values,
        sample_hints,
        sample_owner_ids,
        sample_signs,
        rhs_statuses,
    )
    executor_managed_array_bytes = sum(
        int(value.nbytes) for value in managed_arrays
    )
    if executor_managed_array_bytes != expected_executor_bytes:
        raise RuntimeError("executor raw-array byte accounting mismatch")

    active_count = 0
    interior_seed_count = 0
    counters = _ExecutionCounters()
    state.active = True
    try:
        for seed_id in range(seed_count):
            inside = all(
                float(state.domain_lower[axis])
                <= float(seeds[seed_id, axis])
                < float(state.domain_upper[axis])
                for axis in range(3)
            )
            point_counts[seed_id] = 0
            termination_stages[seed_id] = int(FieldLineStage.CONTROL)
            if not inside:
                termination_codes[seed_id] = int(
                    FieldLineTermination.SEED_OUTSIDE
                )
                continue
            for axis in range(3):
                coordinate = seeds[seed_id, axis]
                positions[seed_id, 0, axis] = coordinate
                current_states[seed_id, axis] = coordinate
            integrals[seed_id, 0] = np.float64(0.0)
            current_states[seed_id, 3] = np.float64(0.0)
            point_counts[seed_id] = 1
            termination_codes[seed_id] = int(FieldLineTermination.ACTIVE)
            active_ids[active_count] = seed_id
            active_count += 1
            interior_seed_count += 1

        for _step_index in range(max_steps):
            if not active_count:
                break
            counters.attempted_steps = _checked_add(
                "attempted step count", counters.attempted_steps, active_count
            )
            for row in range(active_count):
                seed_id = int(active_ids[row])
                for component in range(4):
                    base_states[row, component] = current_states[
                        seed_id, component
                    ]

            active_count = _sample_rhs_stage(
                state,
                active_ids,
                active_count,
                base_states,
                (base_states,),
                k1,
                last_owner_ids,
                direction_signs,
                sample_points,
                sample_hints,
                sample_values,
                sample_owner_ids,
                sample_signs,
                rhs_statuses,
                int(FieldLineStage.K1),
                termination_codes,
                termination_stages,
                counters,
            )
            if not active_count:
                continue

            field_line_rk4_stage_unchecked(
                base_states[:active_count],
                k1[:active_count],
                step_size,
                True,
                stage_states[:active_count],
            )
            active_count = _sample_rhs_stage(
                state,
                active_ids,
                active_count,
                stage_states,
                (base_states, k1),
                k2,
                last_owner_ids,
                direction_signs,
                sample_points,
                sample_hints,
                sample_values,
                sample_owner_ids,
                sample_signs,
                rhs_statuses,
                int(FieldLineStage.K2),
                termination_codes,
                termination_stages,
                counters,
            )
            if not active_count:
                continue

            field_line_rk4_stage_unchecked(
                base_states[:active_count],
                k2[:active_count],
                step_size,
                True,
                stage_states[:active_count],
            )
            active_count = _sample_rhs_stage(
                state,
                active_ids,
                active_count,
                stage_states,
                (base_states, k1, k2),
                k3,
                last_owner_ids,
                direction_signs,
                sample_points,
                sample_hints,
                sample_values,
                sample_owner_ids,
                sample_signs,
                rhs_statuses,
                int(FieldLineStage.K3),
                termination_codes,
                termination_stages,
                counters,
            )
            if not active_count:
                continue

            field_line_rk4_stage_unchecked(
                base_states[:active_count],
                k3[:active_count],
                step_size,
                False,
                stage_states[:active_count],
            )
            active_count = _sample_rhs_stage(
                state,
                active_ids,
                active_count,
                stage_states,
                (base_states, k1, k2, k3),
                k4,
                last_owner_ids,
                direction_signs,
                sample_points,
                sample_hints,
                sample_values,
                sample_owner_ids,
                sample_signs,
                rhs_statuses,
                int(FieldLineStage.K4),
                termination_codes,
                termination_stages,
                counters,
            )
            if not active_count:
                continue

            field_line_rk4_finish_unchecked(
                base_states[:active_count],
                k1[:active_count],
                k2[:active_count],
                k3[:active_count],
                k4[:active_count],
                step_size,
                stage_states[:active_count],
            )
            next_count = 0
            for row in range(active_count):
                seed_id = int(active_ids[row])
                code = _classify_field_line_candidate_unchecked(
                    base_states[row],
                    stage_states[row],
                    state.domain_lower,
                    state.domain_upper,
                )
                if code != int(FieldLineTermination.ACTIVE):
                    _mark_termination(
                        seed_id,
                        code,
                        int(FieldLineStage.CANDIDATE),
                        termination_codes,
                        termination_stages,
                    )
                    continue
                destination = int(point_counts[seed_id])
                for axis in range(3):
                    coordinate = stage_states[row, axis]
                    positions[seed_id, destination, axis] = coordinate
                    current_states[seed_id, axis] = coordinate
                integral = stage_states[row, 3]
                integrals[seed_id, destination] = integral
                current_states[seed_id, 3] = integral
                point_counts[seed_id] = destination + 1
                next_active_ids[next_count] = seed_id
                next_count += 1
                counters.accepted_steps = _checked_add(
                    "accepted step count", counters.accepted_steps, 1
                )
            for row in range(next_count):
                active_ids[row] = next_active_ids[row]
            active_count = next_count

        for row in range(active_count):
            seed_id = int(active_ids[row])
            _mark_termination(
                seed_id,
                int(FieldLineTermination.MAX_STEPS),
                int(FieldLineStage.CONTROL),
                termination_codes,
                termination_stages,
            )
    finally:
        state.active = False

    termination_histogram = [0] * (int(FieldLineTermination.NO_PROGRESS) + 1)
    accepted_point_count = 0
    for seed_id in range(seed_count):
        termination_histogram[int(termination_codes[seed_id])] += 1
        accepted_point_count = _checked_add(
            "accepted point count",
            accepted_point_count,
            int(point_counts[seed_id]),
        )
    if accepted_point_count != interior_seed_count + counters.accepted_steps:
        raise RuntimeError("accepted point accounting mismatch")
    if sum(termination_histogram[1:]) != seed_count:
        raise RuntimeError("termination histogram contains an active seed")

    return FieldLineExecutionStats(
        seed_count,
        interior_seed_count,
        accepted_point_count,
        counters.attempted_steps,
        counters.accepted_steps,
        counters.stage_samples,
        counters.sampler_calls,
        counters.sampler_rejections,
        counters.hint_candidates,
        counters.hint_hits,
        counters.hierarchy_fallbacks,
        counters.cache_lookups,
        counters.cache_hits,
        counters.cache_misses,
        counters.cache_evictions,
        counters.halo_fills,
        counters.reader_calls,
        counters.selected_loads,
        counters.support_loads,
        counters.maximum_selected_slots,
        counters.logical_reader_bytes,
        termination_histogram[int(FieldLineTermination.SEED_OUTSIDE)],
        termination_histogram[int(FieldLineTermination.MAX_STEPS)],
        termination_histogram[int(FieldLineTermination.DOMAIN_EXIT)],
        termination_histogram[int(FieldLineTermination.ZERO_FIELD)],
        termination_histogram[int(FieldLineTermination.NONFINITE_FIELD)],
        termination_histogram[int(FieldLineTermination.UNREPRESENTABLE_NORM)],
        termination_histogram[int(FieldLineTermination.UNREPRESENTABLE_SAMPLE)],
        termination_histogram[int(FieldLineTermination.NONFINITE_STATE)],
        termination_histogram[int(FieldLineTermination.NO_PROGRESS)],
        counters.sampler_managed_peak,
        executor_managed_array_bytes,
        state.session_managed_array_bytes,
    )
