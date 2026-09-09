from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import pytest

import simesh_rewrite.field_lines as sle_module
from simesh.amrvac.datio import header_template, write_datfile_from_sfc
from simesh_rewrite.amrvac_dat import bind_amrvac_v5_forest, read_amrvac_v5_index
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.blockio import array_block_reader, make_block_reader
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.completed_halo_sampling import (
    CachedVectorSamplingStats,
    UnrepresentableRefinedSampleError,
    clear_completed_halo_sampling_session,
    make_completed_halo_sampling_session,
    sample_refined_trilinear_vectors_cached,
)
from simesh_rewrite.field_line_termination import (
    FieldLineStage,
    FieldLineTermination,
)
from simesh_rewrite.field_lines import execute_refined_field_lines
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.storage import gather_blocks_into


SENTINEL_BITS = np.uint64(0x7FF8000000005E01)
SENTINEL = np.asarray([SENTINEL_BITS], dtype=np.uint64).view(np.float64)[0]


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class Artifact:
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    max_level: int
    lower: np.ndarray
    upper: np.ndarray
    domain_counts: np.ndarray
    block_counts: np.ndarray
    backing: np.ndarray


def make_artifact(
    field_kind: str = "constant",
    *,
    root_shape: tuple[int, int, int] = (2, 1, 1),
    block_shape: tuple[int, int, int] = (4, 4, 4),
) -> Artifact:
    roots = np.asarray(root_shape, dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(roots)
    flags = np.asarray(
        [False, *([True] * 8), *([True] * (int(np.prod(roots)) - 1))],
        dtype=np.bool_,
    )
    forest = refined_forest(roots, coord_to_rank, rank_to_coord, flags)
    max_level = validate_refined_forest_arrays(
        roots,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    block_counts = np.asarray(block_shape, dtype=np.int64)
    lower = np.zeros(3, dtype=np.float64)
    upper = roots.astype(np.float64)
    domain_counts = roots * block_counts
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        lower,
        upper,
        roots,
        domain_counts,
        block_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )
    backing = np.empty(
        (leaf_ids.size, 3, *block_shape), dtype=np.float64
    )
    grid = np.indices(block_shape, dtype=np.float64)
    for leaf in range(leaf_ids.size):
        coordinates = [
            bounds[leaf, 0, axis]
            + (grid[axis] + np.float64(0.5)) * spacing[leaf, axis]
            for axis in range(3)
        ]
        if field_kind == "constant":
            backing[leaf, 0].fill(2.0)
            backing[leaf, 1].fill(0.0)
            backing[leaf, 2].fill(0.0)
        elif field_kind == "oblique":
            backing[leaf, 0].fill(1.0)
            backing[leaf, 1].fill(0.25)
            backing[leaf, 2].fill(0.125)
        elif field_kind == "rotation":
            backing[leaf, 0] = -(coordinates[1] - 0.5)
            backing[leaf, 1] = coordinates[0] - 1.5
            backing[leaf, 2].fill(0.0)
        else:
            raise ValueError(f"unknown field kind {field_kind}")
    return Artifact(
        roots,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        max_level,
        lower,
        upper,
        domain_counts,
        block_counts,
        backing,
    )


@dataclass
class ReaderState:
    backing: np.ndarray
    calls: list[np.ndarray] = field(default_factory=list)
    nonempty_calls: int = 0
    bytes_read: int = 0
    fail_nonempty_call: int | None = None

    def reset(self, *, keep_failure: bool = False) -> None:
        self.calls.clear()
        self.nonempty_calls = 0
        self.bytes_read = 0
        if not keep_failure:
            self.fail_nonempty_call = None


def counting_read(
    state: ReaderState,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls.append(block_ids.copy())
    if block_ids.size:
        state.nonempty_calls += 1
        if state.nonempty_calls == state.fail_nonempty_call:
            raise OSError("injected SLE reader failure")
        state.bytes_read += (
            int(block_ids.size)
            * int(field_ids.size)
            * int(np.prod(source_upper - source_lower, dtype=np.int64))
            * 8
        )
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def counting_reader(state: ReaderState):
    return make_block_reader(
        state,
        state.backing.shape,
        counting_read,
        memory_arrays=(state.backing,),
    )


def boundary_configuration(*, mixed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    modes = np.zeros((3, 6), dtype=np.uint8)
    if mixed:
        modes[0, 0] = 2
        modes[1, 2] = 3
        modes[2, 4] = 1
    return modes, i3(-1, 1, -1)


def cache_entry_bytes(artifact: Artifact) -> int:
    return (
        3 * int(np.prod(artifact.block_counts + 2, dtype=np.int64)) * 8 + 16
    )


def make_session(
    artifact: Artifact,
    reader,
    capacity: int,
    *,
    mixed_boundaries: bool = False,
):
    modes, normals = boundary_configuration(mixed=mixed_boundaries)
    session = make_completed_halo_sampling_session(
        reader,
        i3(0, 1, 2),
        artifact.lower,
        artifact.upper,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.max_level,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        modes,
        normals,
        len(artifact.leaf_node_ids),
        capacity * cache_entry_bytes(artifact),
    )
    assert session.cache_capacity == capacity
    return session


def outputs(seed_count: int, max_steps: int) -> tuple[np.ndarray, ...]:
    return (
        np.full((seed_count, max_steps + 1, 3), SENTINEL),
        np.full((seed_count, max_steps + 1), SENTINEL),
        np.full(seed_count, -801, dtype=np.int64),
        np.full(seed_count, 251, dtype=np.uint8),
        np.full(seed_count, 252, dtype=np.uint8),
    )


def run_lines(session, seeds, signs, step: float, max_steps: int):
    result = outputs(len(seeds), max_steps)
    stats = execute_refined_field_lines(
        session, seeds, signs, step, max_steps, *result
    )
    return result, stats


def bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.uint8).copy()


def snapshot_session(session) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    state = session._state
    return (
        bits(state.cache_payload),
        state.cache_leaf_ids.copy(),
        state.cache_recency.copy(),
        state.clock,
    )


def assert_session_snapshot(session, before) -> None:
    state = session._state
    payload, keys, recency, clock = before
    assert np.array_equal(bits(state.cache_payload), payload)
    assert np.array_equal(state.cache_leaf_ids, keys)
    assert np.array_equal(state.cache_recency, recency)
    assert state.clock == clock


def assert_accounting(stats, result, session, reader_state: ReaderState | None = None) -> None:
    positions, integrals, counts, codes, stages = result
    del positions, integrals, stages
    histogram_fields = (
        "seed_outside_count",
        "max_steps_count",
        "domain_exit_count",
        "zero_field_count",
        "nonfinite_field_count",
        "unrepresentable_norm_count",
        "unrepresentable_sample_count",
        "nonfinite_state_count",
        "no_progress_count",
    )
    histogram = np.bincount(codes, minlength=10)
    for code, name in enumerate(histogram_fields, start=1):
        assert getattr(stats, name) == int(histogram[code])
    assert sum(getattr(stats, name) for name in histogram_fields) == stats.seed_count
    assert stats.accepted_point_count == int(np.sum(counts, dtype=np.int64))
    assert stats.accepted_point_count == stats.interior_seed_count + stats.accepted_step_count
    assert stats.executor_managed_array_bytes == 314 * stats.seed_count
    assert stats.session_managed_array_bytes == session.session_managed_array_bytes
    if reader_state is not None:
        assert stats.reader_call_count == reader_state.nonempty_calls
        assert stats.logical_reader_bytes == reader_state.bytes_read
        assert stats.selected_load_count == sum(len(call) for call in reader_state.calls)
        assert stats.support_load_count == stats.selected_load_count - stats.halo_fill_count
        assert stats.maximum_selected_slots == max(
            (len(call) for call in reader_state.calls), default=0
        )


def assert_tail_preserved(result) -> None:
    positions, integrals, counts, _, _ = result
    for seed, count_value in enumerate(counts):
        count = int(count_value)
        assert np.all(positions[seed, count:].view(np.uint64) == SENTINEL_BITS)
        assert np.all(integrals[seed, count:].view(np.uint64) == SENTINEL_BITS)


def test_empty_batch_has_exact_zero_stats_and_no_mutation_or_io() -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 1)
    reader_state.reset()
    result, stats = run_lines(
        session,
        np.empty((0, 3), dtype=np.float64),
        np.empty(0, dtype=np.int8),
        0.125,
        3,
    )
    assert stats.seed_count == 0
    assert stats.interior_seed_count == 0
    assert stats.accepted_point_count == 0
    assert stats.attempted_step_count == 0
    assert stats.accepted_step_count == 0
    assert stats.stage_sample_count == 0
    assert stats.sampler_call_count == 0
    assert stats.sampler_preflight_rejection_count == 0
    assert stats.executor_managed_array_bytes == 0
    assert reader_state.calls == []
    assert_accounting(stats, result, session, reader_state)


def test_finite_exterior_and_exact_upper_faces_preserve_all_value_slots() -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 1)
    reader_state.reset()
    seeds = np.asarray(
        (
            (-np.finfo(np.float64).eps, 0.5, 0.5),
            (2.0, 0.5, 0.5),
            (0.5, 1.0, 0.5),
            (0.5, 0.5, 1.0),
        ),
        dtype=np.float64,
    )
    result, stats = run_lines(
        session, seeds, np.ones(4, dtype=np.int8), 0.125, 2
    )
    positions, integrals, counts, codes, stages = result
    assert counts.tolist() == [0, 0, 0, 0]
    assert codes.tolist() == [int(FieldLineTermination.SEED_OUTSIDE)] * 4
    assert stages.tolist() == [int(FieldLineStage.CONTROL)] * 4
    assert np.all(positions.view(np.uint64) == SENTINEL_BITS)
    assert np.all(integrals.view(np.uint64) == SENTINEL_BITS)
    assert stats.seed_outside_count == 4
    assert stats.sampler_call_count == 0
    assert reader_state.calls == []
    assert_accounting(stats, result, session, reader_state)


def test_max_steps_zero_initializes_only_interior_prefixes() -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 1)
    reader_state.reset()
    seeds = np.asarray(((0.0, 0.25, 0.25), (2.0, 0.25, 0.25)), dtype=np.float64)
    result, stats = run_lines(session, seeds, np.asarray((1, -1), dtype=np.int8), 0.25, 0)
    positions, integrals, counts, codes, stages = result
    assert counts.tolist() == [1, 0]
    assert np.array_equal(positions[0, 0].view(np.uint64), seeds[0].view(np.uint64))
    assert integrals[0, 0].view(np.uint64) == np.float64(0.0).view(np.uint64)
    assert np.all(positions[1].view(np.uint64) == SENTINEL_BITS)
    assert np.all(integrals[1].view(np.uint64) == SENTINEL_BITS)
    assert codes.tolist() == [
        int(FieldLineTermination.MAX_STEPS),
        int(FieldLineTermination.SEED_OUTSIDE),
    ]
    assert stages.tolist() == [int(FieldLineStage.CONTROL)] * 2
    assert stats.attempted_step_count == stats.accepted_step_count == 0
    assert stats.max_steps_count == stats.seed_outside_count == 1
    assert reader_state.calls == []
    assert_accounting(stats, result, session, reader_state)


@pytest.mark.parametrize("sign", [-1, 1])
def test_constant_field_has_bitwise_exact_path_oriented_integral_and_stats(sign: int) -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(
        artifact,
        counting_reader(reader_state),
        len(artifact.leaf_node_ids),
    )
    reader_state.reset()
    seed = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    step = 0.03125
    result, stats = run_lines(
        session, seed, np.asarray((sign,), dtype=np.int8), step, 2
    )
    positions, integrals, counts, codes, stages = result
    expected_positions = np.repeat(seed[:, None, :], 3, axis=1)
    expected_positions[0, :, 0] += sign * step * np.arange(3)
    expected_integrals = (
        sign * 2.0 * step * np.arange(3, dtype=np.float64)
    ).reshape(1, 3)
    expected_integrals[0, 0] = np.float64(0.0)
    assert np.array_equal(positions.view(np.uint64), expected_positions.view(np.uint64))
    assert np.array_equal(integrals.view(np.uint64), expected_integrals.view(np.uint64))
    assert counts.tolist() == [3]
    assert codes.tolist() == [int(FieldLineTermination.MAX_STEPS)]
    assert stages.tolist() == [int(FieldLineStage.CONTROL)]
    assert stats.attempted_step_count == 2
    assert stats.accepted_step_count == 2
    assert stats.stage_sample_count == 8
    assert stats.sampler_call_count == 8
    assert stats.hint_candidate_count == 7
    assert stats.hint_hit_count == 7
    assert stats.hierarchy_fallback_count == 1
    assert stats.cache_lookup_count == 8
    assert stats.cache_hit_count == 7
    assert stats.cache_miss_count == 1
    assert stats.cache_eviction_count == 0
    assert stats.halo_fill_count == 1
    assert stats.reader_call_count == 1
    assert stats.sampler_preflight_rejection_count == 0
    assert stats.selected_load_count == 8
    assert stats.support_load_count == 7
    assert stats.maximum_selected_slots == 8
    assert stats.logical_reader_bytes == 12288
    assert stats.sampler_call_managed_peak_bytes == 201
    assert stats.executor_managed_array_bytes == 314
    assert stats.session_managed_array_bytes == 125042
    assert_accounting(stats, result, session, reader_state)


def test_stage_major_compaction_rejects_domain_exit_before_k2_sampling() -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 0)
    reader_state.reset()
    seeds = np.asarray(((1.95, 0.5, 0.5), (0.5, 0.5, 0.5)), dtype=np.float64)
    result, stats = run_lines(
        session, seeds, np.ones(2, dtype=np.int8), 0.2, 1
    )
    positions, integrals, counts, codes, stages = result
    assert counts.tolist() == [1, 2]
    assert codes.tolist() == [
        int(FieldLineTermination.DOMAIN_EXIT),
        int(FieldLineTermination.MAX_STEPS),
    ]
    assert stages.tolist() == [int(FieldLineStage.K2), int(FieldLineStage.CONTROL)]
    assert stats.attempted_step_count == 2
    assert stats.accepted_step_count == 1
    assert stats.stage_sample_count == 5  # two at k1, one at each later stage
    assert stats.sampler_call_count == 4
    assert np.array_equal(positions[0, 0].view(np.uint64), seeds[0].view(np.uint64))
    assert np.array_equal(
        positions[1, 1].view(np.uint64),
        np.asarray((0.7, 0.5, 0.5), dtype=np.float64).view(np.uint64),
    )
    assert integrals[1, 1] == 0.4
    assert_tail_preserved(result)
    assert_accounting(stats, result, session, reader_state)


def test_rotational_field_has_fourth_order_trajectory_convergence() -> None:
    artifact = make_artifact("rotation")
    seed = np.asarray(((1.75, 0.5, 0.5),), dtype=np.float64)
    signs = np.ones(1, dtype=np.int8)
    final_errors: list[float] = []
    final_integral_errors: list[float] = []
    distance = 0.4
    radius = 0.25
    angle = distance / radius
    expected = np.asarray(
        (1.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle), 0.5),
        dtype=np.float64,
    )
    for step_count in (20, 40):
        session = make_session(
            artifact,
            array_block_reader(artifact.backing),
            len(artifact.leaf_node_ids),
        )
        result, stats = run_lines(
            session, seed, signs, distance / step_count, step_count
        )
        positions, integrals, counts, codes, stages = result
        assert counts.tolist() == [step_count + 1]
        assert codes.tolist() == [int(FieldLineTermination.MAX_STEPS)]
        assert stages.tolist() == [int(FieldLineStage.CONTROL)]
        final_errors.append(float(np.linalg.norm(positions[0, -1] - expected)))
        final_integral_errors.append(abs(float(integrals[0, -1]) - radius * distance))
        assert_accounting(stats, result, session)
    assert final_errors[0] / final_errors[1] > 12.0
    assert final_integral_errors[0] / final_integral_errors[1] > 12.0


def test_cache_capacity_reader_backend_and_mixed_refined_pbc_are_bitwise_invariant() -> None:
    artifact = make_artifact("oblique")
    relation_kinds, relation_masks, _, _ = balanced_refined_relations(
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        np.arange(artifact.leaf_node_ids.size, dtype=np.int64),
        CANONICAL_DIRECTIONS,
    )
    assert set(int(value) for value in np.unique(relation_kinds)) == {1, 2, 3, 4}
    assert np.any(relation_masks != 0)
    modes, _ = boundary_configuration(mixed=True)
    assert set(int(value) for value in np.unique(modes)) == {0, 1, 2, 3}

    seeds = np.asarray(
        ((0.02, 0.2, 0.5), (0.85, 0.8, 0.5), (1.25, 0.45, 0.5)),
        dtype=np.float64,
    )
    signs = np.asarray((1, 1, -1), dtype=np.int8)
    baselines: tuple[np.ndarray, ...] | None = None
    for backend in ("array", "callback"):
        for capacity in (0, 1, len(artifact.leaf_node_ids)):
            if backend == "array":
                reader = array_block_reader(artifact.backing)
            else:
                reader = counting_reader(ReaderState(artifact.backing))
            session = make_session(
                artifact, reader, capacity, mixed_boundaries=True
            )
            result, stats = run_lines(session, seeds, signs, 0.015625, 5)
            assert_accounting(stats, result, session)
            comparable = tuple(bits(value) for value in result)
            if baselines is None:
                baselines = comparable
            else:
                assert all(
                    np.array_equal(actual, expected)
                    for actual, expected in zip(comparable, baselines, strict=True)
                )
    assert baselines is not None


def write_native_artifact(path: Path, artifact: Artifact) -> None:
    leaf_nodes = artifact.leaf_node_ids
    block_levels = artifact.node_levels[leaf_nodes].astype(np.int32)
    block_coordinates = (artifact.node_coords[leaf_nodes] + 1).astype(np.int32)
    root_count = int(np.prod(artifact.root_shape, dtype=np.int64))
    flags = np.asarray(
        [False, *([True] * 8), *([True] * (root_count - 1))],
        dtype=np.int32,
    )
    header = header_template.copy()
    header.update(
        datfile_version=5,
        nw=3,
        ndir=3,
        ndim=3,
        levmax=artifact.max_level,
        nleafs=len(artifact.leaf_node_ids),
        nparents=len(artifact.node_levels) - len(artifact.leaf_node_ids),
        xmin=artifact.lower,
        xmax=artifact.upper,
        domain_nx=artifact.domain_counts.astype(np.int32),
        block_nx=artifact.block_counts.astype(np.int32),
        periodic=np.zeros(3, dtype=np.bool_),
        geometry="Cartesian_3D",
        staggered=False,
        w_names=["b0", "b1", "b2"],
    )
    tree = (
        block_levels,
        block_coordinates,
        np.zeros(len(leaf_nodes), dtype=np.int64),
    )
    write_datfile_from_sfc(
        str(path), artifact.backing, header, flags, tree, overwrite=True
    )


def test_native_v5_reader_matches_array_backend_trajectory_bits(tmp_path: Path) -> None:
    artifact = make_artifact("oblique")
    seeds = np.asarray(
        ((0.2, 0.2, 0.5), (0.9, 0.7, 0.5), (1.4, 0.4, 0.5)),
        dtype=np.float64,
    )
    signs = np.asarray((1, -1, 1), dtype=np.int8)
    array_session = make_session(
        artifact,
        array_block_reader(artifact.backing),
        1,
        mixed_boundaries=True,
    )
    expected, _ = run_lines(array_session, seeds, signs, 0.015625, 4)

    path = tmp_path / "sle-refined-v5.dat"
    write_native_artifact(path, artifact)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(descriptor)
        binding = bind_amrvac_v5_forest(index)
        forest = binding.forest
        native_reader = make_amrvac_v5_block_reader(
            descriptor, index, binding
        )
        modes, normals = boundary_configuration(mixed=True)
        native_session = make_completed_halo_sampling_session(
            native_reader,
            i3(0, 1, 2),
            index.domain_lower,
            index.domain_upper,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            modes,
            normals,
            len(forest.leaf_node_ids),
            cache_entry_bytes(artifact),
        )
        actual, stats = run_lines(
            native_session, seeds, signs, 0.015625, 4
        )
        assert stats.reader_call_count > 0
        assert all(
            np.array_equal(bits(left), bits(right))
            for left, right in zip(actual, expected, strict=True)
        )
        assert_accounting(stats, actual, native_session)
    finally:
        os.close(descriptor)


def synthetic_sampling_stats(
    state,
    point_count: int,
    hint_count: int,
    call_bytes: int,
) -> CachedVectorSamplingStats:
    owner_count = int(point_count > 0)
    return CachedVectorSamplingStats(
        point_count,
        point_count,
        owner_count,
        hint_count,
        hint_count,
        point_count - hint_count,
        owner_count,
        owner_count,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        call_bytes,
        state.session_managed_array_bytes,
    )


@pytest.mark.parametrize(
    ("target_call", "vector", "termination"),
    [
        (1, (0.0, -0.0, 0.0), FieldLineTermination.ZERO_FIELD),
        (2, (np.nan, 1.0, 0.0), FieldLineTermination.NONFINITE_FIELD),
        (
            3,
            (np.finfo(np.float64).max, np.finfo(np.float64).max, 0.0),
            FieldLineTermination.UNREPRESENTABLE_NORM,
        ),
        (4, (0.0, 0.0, 0.0), FieldLineTermination.ZERO_FIELD),
    ],
    ids=("zero-k1", "nonfinite-k2", "unrepresentable-norm-k3", "zero-k4"),
)
def test_rhs_numerical_terminations_are_classified_at_exact_stage(
    monkeypatch,
    target_call: int,
    vector: tuple[float, float, float],
    termination: FieldLineTermination,
) -> None:
    artifact = make_artifact()
    session = make_session(artifact, array_block_reader(artifact.backing), 0)
    calls = 0

    def fake_sample(state, points, hints, values, owners):
        nonlocal calls
        calls += 1
        values[:] = np.asarray(
            vector if calls == target_call else (1.0, 0.0, 0.0),
            dtype=np.float64,
        )
        owners[:] = 0
        hint_count = int(np.count_nonzero(hints >= 0))
        return synthetic_sampling_stats(state, len(points), hint_count, 80 + calls)

    monkeypatch.setattr(
        sle_module, "_sample_refined_trilinear_vectors_cached", fake_sample
    )
    result, stats = run_lines(
        session,
        np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64),
        np.ones(1, dtype=np.int8),
        0.03125,
        2,
    )
    _, _, counts, codes, stages = result
    assert calls == target_call
    assert counts.tolist() == [1]
    assert codes.tolist() == [int(termination)]
    assert stages.tolist() == [target_call]
    assert stats.sampler_call_count == target_call
    assert stats.stage_sample_count == target_call
    assert stats.sampler_call_managed_peak_bytes == 80 + target_call
    field_name = {
        FieldLineTermination.ZERO_FIELD: "zero_field_count",
        FieldLineTermination.NONFINITE_FIELD: "nonfinite_field_count",
        FieldLineTermination.UNREPRESENTABLE_NORM: "unrepresentable_norm_count",
    }[termination]
    assert getattr(stats, field_name) == 1
    assert_tail_preserved(result)
    assert_accounting(stats, result, session)


def test_nonfinite_generated_stage_checks_integral_before_domain_at_k2(monkeypatch) -> None:
    artifact = make_artifact()
    session = make_session(artifact, array_block_reader(artifact.backing), 0)

    def maximum_sample(state, points, hints, values, owners):
        values[:] = (np.finfo(np.float64).max, 0.0, 0.0)
        owners[:] = 0
        return synthetic_sampling_stats(state, len(points), 0, 111)

    monkeypatch.setattr(
        sle_module, "_sample_refined_trilinear_vectors_cached", maximum_sample
    )
    result, stats = run_lines(
        session,
        np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64),
        np.ones(1, dtype=np.int8),
        4.0,
        1,
    )
    _, _, counts, codes, stages = result
    assert counts.tolist() == [1]
    assert codes.tolist() == [int(FieldLineTermination.NONFINITE_STATE)]
    assert stages.tolist() == [int(FieldLineStage.K2)]
    assert stats.sampler_call_count == stats.stage_sample_count == 1
    assert stats.nonfinite_state_count == 1
    assert_accounting(stats, result, session)


def test_nonfinite_rk_weighted_candidate_is_not_committed(monkeypatch) -> None:
    artifact = make_artifact()
    session = make_session(artifact, array_block_reader(artifact.backing), 0)

    def maximum_sample(state, points, hints, values, owners):
        values[:] = (np.finfo(np.float64).max, 0.0, 0.0)
        owners[:] = 0
        return synthetic_sampling_stats(state, len(points), 0, 121)

    monkeypatch.setattr(
        sle_module, "_sample_refined_trilinear_vectors_cached", maximum_sample
    )
    result, stats = run_lines(
        session,
        np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64),
        np.ones(1, dtype=np.int8),
        0.5,
        1,
    )
    _, _, counts, codes, stages = result
    assert counts.tolist() == [1]
    assert codes.tolist() == [int(FieldLineTermination.NONFINITE_STATE)]
    assert stages.tolist() == [int(FieldLineStage.CANDIDATE)]
    assert stats.stage_sample_count == stats.sampler_call_count == 4
    assert stats.nonfinite_state_count == 1
    assert_tail_preserved(result)
    assert_accounting(stats, result, session)


def test_minimum_normal_step_detects_bitwise_coordinate_no_progress() -> None:
    artifact = make_artifact()
    session = make_session(artifact, array_block_reader(artifact.backing), 1)
    result, stats = run_lines(
        session,
        np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64),
        np.ones(1, dtype=np.int8),
        float(np.finfo(np.float64).tiny),
        1,
    )
    _, _, counts, codes, stages = result
    assert counts.tolist() == [1]
    assert codes.tolist() == [int(FieldLineTermination.NO_PROGRESS)]
    assert stages.tolist() == [int(FieldLineStage.CANDIDATE)]
    assert stats.stage_sample_count == stats.sampler_call_count == 4
    assert stats.no_progress_count == 1
    assert_tail_preserved(result)
    assert_accounting(stats, result, session)


def test_unrepresentable_samples_are_removed_retried_and_account_exception_bytes(
    monkeypatch,
) -> None:
    artifact = make_artifact()
    session = make_session(artifact, array_block_reader(artifact.backing), 0)
    batch_sizes: list[int] = []

    def rejecting_sample(state, points, hints, values, owners):
        batch_sizes.append(len(points))
        if len(batch_sizes) == 1:
            raise UnrepresentableRefinedSampleError(0, 2, 333)
        if len(batch_sizes) == 2:
            raise UnrepresentableRefinedSampleError(1, 7, 444)
        values[:] = (1.0, 0.0, 0.0)
        owners[:] = 0
        hint_count = int(np.count_nonzero(hints >= 0))
        return synthetic_sampling_stats(state, len(points), hint_count, 100)

    monkeypatch.setattr(
        sle_module, "_sample_refined_trilinear_vectors_cached", rejecting_sample
    )
    seeds = np.asarray(
        ((0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4)),
        dtype=np.float64,
    )
    result, stats = run_lines(
        session, seeds, np.ones(3, dtype=np.int8), 0.03125, 1
    )
    _, _, counts, codes, stages = result
    assert batch_sizes == [3, 2, 1, 1, 1, 1]
    assert counts.tolist() == [1, 2, 1]
    assert codes.tolist() == [
        int(FieldLineTermination.UNREPRESENTABLE_SAMPLE),
        int(FieldLineTermination.MAX_STEPS),
        int(FieldLineTermination.UNREPRESENTABLE_SAMPLE),
    ]
    assert stages.tolist() == [
        int(FieldLineStage.K1),
        int(FieldLineStage.CONTROL),
        int(FieldLineStage.K1),
    ]
    assert stats.attempted_step_count == 3
    assert stats.accepted_step_count == 1
    assert stats.sampler_call_count == 6
    assert stats.sampler_preflight_rejection_count == 2
    assert stats.stage_sample_count == 4
    assert stats.unrepresentable_sample_count == 2
    assert stats.sampler_call_managed_peak_bytes == 444
    assert_tail_preserved(result)
    assert_accounting(stats, result, session)


@pytest.mark.parametrize(
    ("failed_stage", "stage_code"),
    [
        (1, FieldLineStage.K1),
        (2, FieldLineStage.K2),
        (3, FieldLineStage.K3),
        (4, FieldLineStage.K4),
    ],
)
def test_reader_failure_at_each_stage_preserves_prior_accepted_prefix_and_resets_busy(
    failed_stage: int,
    stage_code: FieldLineStage,
) -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 0)
    reader_state.reset()
    reader_state.fail_nonempty_call = 4 + failed_stage
    seed = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    signs = np.ones(1, dtype=np.int8)
    result = outputs(1, 3)
    with pytest.raises(OSError, match="injected SLE reader failure"):
        execute_refined_field_lines(
            session, seed, signs, 0.03125, 3, *result
        )
    positions, integrals, counts, codes, stages = result
    assert reader_state.nonempty_calls == 4 + failed_stage
    assert counts.tolist() == [2]
    expected_prefix = np.asarray(
        ((0.25, 0.25, 0.25), (0.28125, 0.25, 0.25)), dtype=np.float64
    )
    assert np.array_equal(
        positions[0, :2].view(np.uint64), expected_prefix.view(np.uint64)
    )
    assert np.array_equal(
        integrals[0, :2].view(np.uint64),
        np.asarray((0.0, 0.0625), dtype=np.float64).view(np.uint64),
    )
    assert codes.tolist() == [int(FieldLineTermination.ACTIVE)]
    assert stages.tolist() == [int(FieldLineStage.CONTROL)]
    assert_tail_preserved(result)
    assert session._state.cache_leaf_ids.size == 0
    assert session._state.cache_recency.size == 0
    assert session._state.clock == 0
    assert session._state.active is False

    reader_state.reset()
    recovered, recovered_stats = run_lines(session, seed, signs, 0.03125, 1)
    assert recovered[2].tolist() == [2]
    assert recovered[3].tolist() == [int(FieldLineTermination.MAX_STEPS)]
    assert recovered_stats.accepted_step_count == 1


def point_in_leaf(artifact: Artifact, leaf_id: int) -> np.ndarray:
    selected = np.asarray((leaf_id,), dtype=np.int64)
    bounds, _ = refined_leaf_geometry(
        artifact.lower,
        artifact.upper,
        artifact.root_shape,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        selected,
    )
    return np.ascontiguousarray(np.mean(bounds[0], axis=0).reshape(1, 3))


def test_failed_new_owner_miss_preserves_prior_cache_admission() -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 1)
    first = point_in_leaf(artifact, 0)
    sample_refined_trilinear_vectors_cached(
        session,
        first,
        np.asarray((-1,), dtype=np.int64),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    before = snapshot_session(session)
    occupied = int(session._state.cache_leaf_ids[0])
    other = next(
        leaf for leaf in range(len(artifact.leaf_node_ids)) if leaf != occupied
    )
    reader_state.reset()
    reader_state.fail_nonempty_call = 1
    result = outputs(1, 1)
    with pytest.raises(OSError, match="injected SLE reader failure"):
        execute_refined_field_lines(
            session,
            point_in_leaf(artifact, other),
            np.ones(1, dtype=np.int8),
            0.03125,
            1,
            *result,
        )
    assert result[2].tolist() == [1]
    assert result[3].tolist() == [int(FieldLineTermination.ACTIVE)]
    assert_tail_preserved(result)
    assert_session_snapshot(session, before)
    assert session._state.active is False
    clear_completed_halo_sampling_session(session)


@dataclass
class ReentrantState:
    backing: np.ndarray
    session: object | None = None
    errors: list[Exception] = field(default_factory=list)
    attempted: bool = False


def reentrant_read(
    state: ReentrantState,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    if block_ids.size and not state.attempted and state.session is not None:
        state.attempted = True
        try:
            execute_refined_field_lines(
                state.session,
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int8),
                0.125,
                0,
                *outputs(0, 0),
            )
        except Exception as error:
            state.errors.append(error)
        try:
            clear_completed_halo_sampling_session(state.session)
        except Exception as error:
            state.errors.append(error)
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def test_reader_cannot_reenter_executor_or_clear_and_success_resets_busy() -> None:
    artifact = make_artifact()
    state = ReentrantState(artifact.backing)
    reader = make_block_reader(
        state,
        artifact.backing.shape,
        reentrant_read,
        memory_arrays=(artifact.backing,),
    )
    session = make_session(artifact, reader, 1)
    state.session = session
    result, stats = run_lines(
        session,
        np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64),
        np.ones(1, dtype=np.int8),
        0.03125,
        1,
    )
    assert stats.accepted_step_count == 1
    assert state.attempted
    assert len(state.errors) == 2
    assert all(isinstance(error, ValueError) for error in state.errors)
    assert all("already active" in str(error) for error in state.errors)
    assert session._state.active is False
    clear_completed_halo_sampling_session(session)
    assert result[3].tolist() == [int(FieldLineTermination.MAX_STEPS)]


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("seed_dtype", TypeError, "float64"),
        ("seed_nonfinite", ValueError, "finite"),
        ("bad_sign", ValueError, "-1 or \\+1"),
        ("integer_step", TypeError, "floating scalar"),
        ("max_kind", TypeError, "exact Python int"),
        ("max_overflow", OverflowError, "max_steps"),
        ("positions_layout", ValueError, "C-contiguous"),
        ("codes_readonly", ValueError, "writable"),
    ],
)
def test_ordinary_preflight_failure_preserves_every_output_cache_bit_and_has_no_io(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 1)
    warm_point = point_in_leaf(artifact, 0)
    sample_refined_trilinear_vectors_cached(
        session,
        warm_point,
        np.asarray((-1,), dtype=np.int64),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    reader_state.reset()
    seeds = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    signs = np.ones(1, dtype=np.int8)
    step: object = 0.03125
    max_steps: object = 1
    result = list(outputs(1, 1))
    if failure == "seed_dtype":
        seeds = seeds.astype(np.float32)
    elif failure == "seed_nonfinite":
        seeds = seeds.copy()
        seeds[0, 2] = np.nan
    elif failure == "bad_sign":
        signs[0] = 0
    elif failure == "integer_step":
        step = 1
    elif failure == "max_kind":
        max_steps = np.int64(1)
    elif failure == "max_overflow":
        max_steps = int(np.iinfo(np.int64).max)
    elif failure == "positions_layout":
        result[0] = np.full((1, 2, 6), SENTINEL)[:, :, ::2]
    else:
        result[3].setflags(write=False)
    output_before = tuple(bits(value) for value in result)
    cache_before = snapshot_session(session)
    with pytest.raises(error, match=match):
        execute_refined_field_lines(
            session,
            seeds,
            signs,
            step,
            max_steps,
            *result,
        )
    assert all(
        np.array_equal(bits(value), before)
        for value, before in zip(result, output_before, strict=True)
    )
    assert_session_snapshot(session, cache_before)
    assert reader_state.calls == []
    assert session._state.active is False


@pytest.mark.parametrize(
    ("alias_kind", "match"),
    [
        ("output_pair", "pairwise nonoverlapping"),
        ("seed_output", "outputs must not overlap"),
        ("sign_output", "outputs must not overlap"),
        ("session_input", "inputs must not overlap mutable session memory"),
        ("session_output", "outputs must not overlap"),
        ("reader_output", "outputs must not overlap"),
    ],
)
def test_input_output_and_session_aliases_are_preflight_atomic(
    alias_kind: str,
    match: str,
) -> None:
    artifact = make_artifact()
    reader_state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(reader_state), 1)
    reader_state.reset()
    seeds = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    signs = np.ones(1, dtype=np.int8)
    result = list(outputs(1, 1))
    if alias_kind == "output_pair":
        shared = np.full(6, SENTINEL)
        result[0] = shared.reshape(1, 2, 3)
        result[1] = shared[:2].reshape(1, 2)
    elif alias_kind == "seed_output":
        shared = np.full(6, SENTINEL)
        seeds = shared[:3].reshape(1, 3)
        seeds[:] = (0.25, 0.25, 0.25)
        result[0] = shared.reshape(1, 2, 3)
    elif alias_kind == "sign_output":
        shared = np.zeros(1, dtype=np.int64)
        signs = shared.view(np.int8)[:1]
        signs[0] = 1
        result[2] = shared
    elif alias_kind == "session_input":
        seeds = session._state.workspace.payload.reshape(-1)[:3].reshape(1, 3)
        seeds[:] = (0.25, 0.25, 0.25)
    elif alias_kind == "session_output":
        result[0] = session._state.cache_payload.reshape(-1)[:6].reshape(1, 2, 3)
    else:
        result[0] = artifact.backing.reshape(-1)[:6].reshape(1, 2, 3)
    output_before = tuple(bits(value) for value in result)
    cache_before = snapshot_session(session)
    backing_before = bits(artifact.backing)
    with pytest.raises(ValueError, match=match):
        execute_refined_field_lines(
            session, seeds, signs, 0.03125, 1, *result
        )
    assert all(
        np.array_equal(bits(value), before)
        for value, before in zip(result, output_before, strict=True)
    )
    assert_session_snapshot(session, cache_before)
    assert np.array_equal(bits(artifact.backing), backing_before)
    assert reader_state.calls == []
    assert session._state.active is False
