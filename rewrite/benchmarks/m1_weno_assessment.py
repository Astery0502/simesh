"""Focused real WENO field assessment of completed M1, without core changes."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
import tracemalloc

import numpy as np
import lfe_001 as b
from dat_003 import repack_weno_regular_fields
from m1_optimization import original_module, BASELINE
from m1_cache_scaling import array_inventory_bytes
from m1_halo_attribution import value_stages
import simesh_rewrite.refined_halo as halo
from simesh_rewrite.blockio import read_blocks_into
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.curl_reference import cartesian_curl_reference


REGIONS = (
    ("small", (.48, .48, .48), (.52, .52, .52)),
    ("mixed", (.42, .42, .42), (.58, .58, .58)),
    ("thin", (.495, .3, .3), (.505, .7, .7)),
    ("physical", (0, .3, .3), (.02, .7, .7)),
)


@contextmanager
def collect_reads(selected):
    original = halo.read_blocks_into
    def read(*args, **kwargs):
        selected.update(map(int, args[3]))
        return original(*args, **kwargs)
    halo.read_blocks_into = read
    try:
        yield
    finally:
        halo.read_blocks_into = original


def independent_interior(fixture, selection, output):
    _, spacing = b.refined_leaf_geometry(fixture.domain_lower, fixture.domain_upper,
        fixture.root_shape, fixture.domain_counts, fixture.block_counts,
        fixture.node_levels, fixture.node_coords, fixture.leaf_node_ids, selection.leaf_ids)
    checked = 0
    worst = 0.0
    for row, leaf in enumerate(selection.leaf_ids):
        lo = np.maximum(selection.cell_lower[row], 1)
        hi = np.minimum(selection.cell_upper[row], fixture.block_counts - 1)
        if np.any(lo >= hi):
            continue
        expected = np.full((1, 3, 8, 8, 8), b.SENTINEL)
        cartesian_curl_reference(fixture.backing[leaf:leaf+1], lo, hi, b.i3(0, 1, 2),
            spacing[row:row+1], expected, b.i3(0, 1, 2), lo)
        box = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
        actual = output[(row, slice(None), *box)]
        reference = expected[(0, slice(None), *box)]
        assert np.array_equal(actual.view(np.uint64), reference.view(np.uint64))
        worst = max(worst, float(np.max(np.abs(actual-reference))))
        checked += actual.size
    return {"value_count": checked, "maximum_absolute_difference": worst,
            "bitwise_equal": True, "scope": "Interior stencil only; not an AMR-interface accuracy oracle."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/weno509_sub_0000.dat"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    original_preflight = original_module("refined_halo")._preflight_chunk_actions
    retained = halo._preflight_chunk_actions
    source_fd = os.open(args.source, os.O_RDONLY)
    try:
        started = time.perf_counter()
        source_index = b.read_amrvac_v5_index(source_fd)
        source_binding = b.bind_amrvac_v5_forest(source_index)
        metadata_seconds = time.perf_counter() - started
        try:
            b.make_amrvac_v5_block_reader(source_fd, source_index, source_binding)
        except ValueError as error:
            assert "staggered" in str(error)
        else:
            raise AssertionError("direct staggered reader must reject this source")
    finally:
        os.close(source_fd)
    with tempfile.TemporaryDirectory(prefix="simesh-m1-weno-") as directory:
        path = Path(directory) / "regular-fields.dat"
        bridge = repack_weno_regular_fields(args.source, path, (4, 5, 6))
        fd = os.open(path, os.O_RDONLY)
        try:
            index = b.read_amrvac_v5_index(fd)
            binding = b.bind_amrvac_v5_forest(index)
            f = binding.forest
            for name in ("root_node_ids", "node_levels", "node_coords", "child_node_ids",
                         "node_leaf_ids", "leaf_node_ids"):
                np.testing.assert_array_equal(getattr(f, name), getattr(source_binding.forest, name))
            b.validate_refined_all_touch_2to1(binding.root_shape, binding.coord_to_rank,
                f.root_node_ids, f.node_levels, f.node_coords, f.child_node_ids,
                f.node_leaf_ids, f.leaf_node_ids)
            started = time.perf_counter()
            backing = b.read_blocks_sequential(str(args.source), [4, 5, 6])
            eager_seconds = time.perf_counter() - started
            fixture = b.Fixture(binding.root_shape, binding.coord_to_rank, f.root_node_ids,
                f.node_levels, f.node_coords, f.child_node_ids, f.node_leaf_ids,
                f.leaf_node_ids, f.max_level, index.domain_lower, index.domain_upper,
                index.domain_cell_counts, index.block_cell_counts, backing, index.forest_flags)
            reader = b.make_amrvac_v5_block_reader(fd, index, binding)
            rows = []
            for label, low, high in REGIONS:
                lower = index.domain_lower + np.array(low)*(index.domain_upper-index.domain_lower)
                upper = index.domain_lower + np.array(high)*(index.domain_upper-index.domain_lower)
                selection = b.refined_region_windows(*b.selection_args(fixture, lower, upper))
                kinds, masks, counts, sources = balanced_refined_relations(binding.root_shape,
                    binding.coord_to_rank, f.root_node_ids, f.node_levels, f.node_coords,
                    f.child_node_ids, f.node_leaf_ids, f.leaf_node_ids,
                    selection.leaf_ids, CANONICAL_DIRECTIONS)
                levels, level_counts = np.unique(f.node_levels[f.leaf_node_ids[selection.leaf_ids]], return_counts=True)
                output = np.full((len(selection.leaf_ids), 3, 8, 8, 8), b.SENTINEL)
                accumulator = np.asarray([1.25])
                expected = None
                capacities = []
                reference_stats = None
                for capacity in (57, 256):
                    arguments = b.execution_args(fixture, reader, b.i3(0, 1, 2), selection,
                        capacity, output, accumulator)
                    timings, cpus = [[], []], [[], []]
                    for repeat in range(args.repeats + 1):
                        stats = [None, None]
                        for variant in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            halo._preflight_chunk_actions = (original_preflight, retained)[variant]
                            output.view(np.uint64).fill(b.SENTINEL_BITS)
                            accumulator[0] = 1.25
                            start, cpu = time.perf_counter(), time.process_time()
                            stats[variant] = b.execute_selected_refined_curl_from_blocks(*arguments)
                            elapsed_cpu, elapsed = time.process_time()-cpu, time.perf_counter()-start
                            if repeat:
                                timings[variant].append(elapsed)
                                cpus[variant].append(elapsed_cpu)
                            if expected is None:
                                expected = (output.copy(), accumulator.copy())
                            assert np.array_equal(output.view(np.uint64), expected[0].view(np.uint64))
                            assert np.array_equal(accumulator.view(np.uint64), expected[1].view(np.uint64))
                        assert stats[0] == stats[1]
                    if capacity == 57:
                        reference_stats = stats[1]
                    halo._preflight_chunk_actions = retained
                    output.view(np.uint64).fill(b.SENTINEL_BITS)
                    accumulator[0] = 1.25
                    recorder, preads, selected, kernels = b.StageRecorder(), b.PreadCounter(), set(), {}
                    recorder.started_at = time.perf_counter()
                    with b.instrument_stages(recorder), b.count_native_preads(preads), collect_reads(selected), value_stages(kernels):
                        b.execute_selected_refined_curl_from_blocks(*arguments)
                    instrumented_wall = time.perf_counter()-recorder.started_at
                    assert preads.payload_bytes == stats[1].logical_reader_bytes
                    raw_ids = np.asarray(sorted(selected), dtype=np.int64)
                    for first in range(0, len(raw_ids), 256):
                        ids = raw_ids[first:first+256]
                        raw = np.empty((len(ids), 3, 8, 8, 8))
                        read_blocks_into(reader, b.i3(0, 0, 0), index.block_cell_counts,
                            ids, b.i3(0, 1, 2), raw, b.i3(0, 0, 0))
                        assert np.array_equal(raw.view(np.uint64), backing[ids].view(np.uint64))
                    del raw
                    tracemalloc.start()
                    accumulator[0] = 1.25
                    b.execute_selected_refined_curl_from_blocks(*arguments)
                    _, traced_peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    capacities.append({"capacity": capacity,
                        "before_wall": b.scalar_summary(timings[0]), "after_wall": b.scalar_summary(timings[1]),
                        "before_cpu": b.scalar_summary(cpus[0]), "after_cpu": b.scalar_summary(cpus[1]),
                        "raw_wall": timings, "raw_cpu": cpus, "stats": b.stats_record(stats[1]),
                        "first_result_instrumented_seconds": recorder.first_result_seconds,
                        "instrumented_wall": instrumented_wall, "stages_nested": recorder.seconds,
                        "value_and_planning_kernel_boundaries": kernels, "pread": preads.as_dict(),
                        "unique_read_blocks": len(selected), "traced_call_peak_bytes": traced_peak,
                        "caller_output_bytes": output.nbytes+accumulator.nbytes,
                        "all_live_array_bytes_outside_call": array_inventory_bytes(source_index,source_binding,
                            index,binding,reader,fixture,selection,output,accumulator,expected,kinds,masks,counts,sources),
                        "peak_rss_bytes": b.peak_rss_bytes(), "original_reader_support_bits_equal": True})
                array_stats = b.execute_selected_refined_curl_from_blocks(*b.execution_args(fixture,
                    b.array_block_reader(backing), b.i3(0, 1, 2), selection, 57, output,
                    np.asarray([1.25])))
                assert array_stats == reference_stats
                assert np.array_equal(output.view(np.uint64), expected[0].view(np.uint64))
                interior = independent_interior(fixture, selection, output)
                rows.append({"region": label, "fraction_lower": low, "fraction_upper": high,
                    "leaf_count": len(selection.leaf_ids), "level_histogram": dict(zip(map(str,levels), map(int,level_counts))),
                    "relation_histogram": {str(k): int(np.count_nonzero(kinds==k)) for k in (1,2,3,4)},
                    "physical_direction_rows": int(np.count_nonzero(masks)), "capacities": capacities,
                    "before_after_capacity_array_bits_equal": True, "independent_interior": interior,
                    "digest": b.logical_digest(output,selection.cell_lower,selection.cell_upper)})
                print(label, len(selection.leaf_ids), "validated", flush=True)
            levels, counts = np.unique(f.node_levels[f.leaf_node_ids],return_counts=True)
            report = {"environment": b.environment_record(), "original_m1_revision": BASELINE,
                "source": str(args.source), "source_file_bytes": args.source.stat().st_size,
                "source_staggered": True, "direct_staggered_reader_rejected": True,
                "leaf_count": index.leaf_count, "level_histogram": dict(zip(map(str,levels),map(int,counts))),
                "block_shape": index.block_cell_counts.tolist(), "metadata_seconds": metadata_seconds,
                "bridge": bridge, "canonical_eager_read_seconds": eager_seconds,
                "canonical_eager_array_bytes": backing.nbytes, "cases": rows,
                "scope": "Real regular fields on the original refined forest via bit-preserving bridge; no staggered computation or global physical-accuracy claim.",
                "policy": "One warmup then alternating five paired samples; OS cache uncontrolled. No implementation change or new performance gate."}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        finally:
            halo._preflight_chunk_actions = retained
            os.close(fd)


if __name__ == "__main__":
    main()
