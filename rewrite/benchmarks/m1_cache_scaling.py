"""Bounded cache/query attribution using native owners from the CHS fixture."""

import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import tempfile
import time
import tracemalloc
import sys

import numpy as np
import chs_001 as b
import simesh_rewrite.completed_halo_sampling as cache


def array_inventory_bytes(*roots):
    """Count simultaneous benchmark/caller arrays once, including owning bases."""
    seen, arrays = set(), {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            while isinstance(value.base, np.ndarray):
                value = value.base
            arrays[id(value)] = value.nbytes
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    for root in roots:
        visit(root)
    return sum(arrays.values())


@contextmanager
def stages(totals):
    names = ("_cache_access_plan", "_make_point_plan", "_sample_owner_group",
             "_prepare_refined_halo_chunk", "_complete_owner")
    originals = {name: getattr(cache, name) for name in names}
    try:
        for name, original in originals.items():
            def timed(*args, _name=name, _original=original, **kwargs):
                start = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    totals[_name] = totals.get(_name, 0.0) + time.perf_counter() - start
            setattr(cache, name, timed)
        yield
    finally:
        for name, original in originals.items():
            setattr(cache, name, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    if args.baseline:
        from m1_optimization import original_module
        cache._cache_access_plan = original_module("completed_halo_sampling")._cache_access_plan
    retained = cache._cache_access_plan
    from m1_optimization import original_module
    scan = original_module("completed_halo_sampling")._cache_access_plan
    case = b.synthetic_case()
    records = []
    with tempfile.TemporaryDirectory(prefix="simesh-m1-cache-") as directory:
        path = Path(directory) / "native.dat"
        b.write_synthetic_v5(path, case)
        fd = os.open(path, os.O_RDONLY)
        try:
            index = b.read_amrvac_v5_index(fd)
            binding = b.bind_amrvac_v5_forest(index)
            reader = b.make_amrvac_v5_block_reader(fd, index, binding)
            for capacity in (16, 36, 71):
                for batch in (4, capacity):
                    ids = np.linspace(0, case.leaf_count - 1, capacity, dtype=np.int64)
                    points = np.ascontiguousarray(case.leaf_bounds[ids].mean(axis=1))
                    query = b.Query(f"spread-{capacity}", points, ids.copy(), ids,
                                    b.batch_slices(capacity, batch))
                    entry = 3 * int(np.prod(case.block_counts + 2)) * 8 + 16
                    session = b.make_completed_halo_sampling_session(
                        *b.session_arguments(case, reader, 57, capacity * entry))
                    expected, owners, cold_stats = b.run_chs_sequence(session, query)
                    samples = []
                    scan_samples = []
                    for repeat in range(10 if args.paired else 6):
                        variants = (0, 1) if repeat % 2 == 0 else (1, 0)
                        for variant in variants if args.paired else (1,):
                            cache._cache_access_plan = (scan, retained)[variant]
                            start = time.perf_counter()
                            values, actual_owners, stats = b.run_chs_sequence(session, query)
                            (scan_samples, samples)[variant].append(time.perf_counter() - start)
                            assert np.array_equal(expected.view(np.uint64), values.view(np.uint64))
                            assert np.array_equal(owners, actual_owners)
                            assert stats["cache_miss_count"] == 0
                    cache._cache_access_plan = retained
                    attribution = {}
                    with stages(attribution):
                        b.run_chs_sequence(session, query)
                    plan_samples = []
                    for _ in range(6):
                        start = time.perf_counter()
                        cache._cache_access_plan(session._state, ids)
                        plan_samples.append(time.perf_counter() - start)
                    tracemalloc.start()
                    b.run_chs_sequence(session, query)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    array_peak = array_inventory_bytes(case, index, binding, reader,
                        session, query, ids, expected, owners, values, actual_owners)
                    # Warm call arrays include owner scratch in addition to CHS stats.
                    array_peak += stats["call_managed_array_peak_bytes"] + 8 * capacity
                    dictionary_bound = sys.getsizeof(dict.fromkeys(range(capacity))) + 2 * capacity * sys.getsizeof(0)
                    records.append({"capacity": capacity, "batch": batch,
                                    "raw_seconds": samples[1:],
                                    "wall": b.scalar_summary(samples[1:]),
                                    "paired_scan_raw_seconds": scan_samples[1:],
                                    "paired_scan_wall": b.scalar_summary(scan_samples[1:]) if args.paired else None,
                                    "access_plan_raw_seconds": plan_samples[1:],
                                    "access_plan": b.scalar_summary(plan_samples[1:]),
                                    "stage_seconds_nested": attribution,
                                    "memory": b.memory_breakdown(session),
                                    "warm_stats": stats, "cold_stats": cold_stats,
                                    "traced_warm_peak": peak,
                                    "complete_controlled_array_peak_bytes": array_peak,
                                    "temporary_index_python_upper_bytes": dictionary_bound,
                                    "native_transfer_scratch_upper_bytes": 2 * 3 * int(np.prod(case.block_counts)) * 8,
                                    "query_and_output_bytes": int(points.nbytes + 3*ids.nbytes
                                        + expected.nbytes + owners.nbytes),
                                    "fixture_payload_bytes": int(case.backing.nbytes),
                                    "bitwise_equal": True})
        finally:
            os.close(fd)
    report = {"environment": b.environment_record(), "records": records,
              "baseline_access_plan": args.baseline,
              "cache_policy": "native warm application cache; uncontrolled OS cache"}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
