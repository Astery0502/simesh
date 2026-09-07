"""Interleaved cache controls for runner drift seen in separate standard runs."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import chs_001 as b
import simesh_rewrite.completed_halo_sampling as cache
import simesh_rewrite.refined_halo as halo
from m1_optimization import original_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drift-only", action="store_true")
    args = parser.parse_args()
    old_halo = original_module("refined_halo")._preflight_chunk_actions
    old_cache = original_module("completed_halo_sampling")._cache_access_plan
    final_halo, final_cache = halo._preflight_chunk_actions, cache._cache_access_plan
    variants = ((old_halo, old_cache), (final_halo, old_cache), (final_halo, final_cache))
    case = b.synthetic_case()
    records = []
    with tempfile.TemporaryDirectory(prefix="simesh-query-controls-") as directory:
        path = Path(directory) / "native.dat"
        b.write_synthetic_v5(path, case)
        for fixture, source in (("synthetic", path), ("tdm", Path("data/tdm.dat"))):
            fd = os.open(source, os.O_RDONLY)
            try:
                index = b.read_amrvac_v5_index(fd)
                binding = b.bind_amrvac_v5_forest(index)
                reader = b.make_amrvac_v5_block_reader(fd, index, binding)
                current = case if fixture == "synthetic" else b.case_from_v5(index, binding, b.i3(4, 5, 6), "tdm")
                queries = b.synthetic_queries(case, 192, 4) if fixture == "synthetic" else (b.tdm_query(current, 96, 4),)
                for qi, query in enumerate(queries):
                    capacities = (1, 5) if fixture == "synthetic" and qi == 0 else ((0, 1, 71) if fixture == "synthetic" else (1,))
                    for capacity in capacities:
                        entry = 3 * int(np.prod(current.block_counts + 2)) * 8 + 16
                        sessions = [b.make_completed_halo_sampling_session(*b.session_arguments(
                            current, reader, min(57, current.leaf_count), capacity * entry)) for _ in variants]
                        for warm in (False, True):
                            if args.drift_only and not (fixture == "synthetic" and (
                                    (qi == 0 and capacity == 5) or
                                    (qi == 1 and capacity == 0 and not warm))):
                                continue
                            samples = [[], [], []]
                            cpu_samples = [[], [], []]
                            for repeat in range(8):
                                results = [None, None, None]
                                for vi in tuple((repeat + k) % 3 for k in range(3)):
                                    halo._preflight_chunk_actions, cache._cache_access_plan = variants[vi]
                                    b.clear_completed_halo_sampling_session(sessions[vi])
                                    if warm:
                                        b.run_chs_sequence(sessions[vi], query)
                                    start = time.perf_counter()
                                    cpu_start = time.process_time()
                                    results[vi] = b.run_chs_sequence(sessions[vi], query)
                                    cpu_elapsed = time.process_time() - cpu_start
                                    elapsed = time.perf_counter() - start
                                    if repeat:
                                        samples[vi].append(elapsed)
                                        cpu_samples[vi].append(cpu_elapsed)
                                for result in results[1:]:
                                    assert result[2] == results[0][2]
                                    for a, expected in zip(result[:2], results[0][:2]):
                                        assert np.array_equal(a.view(np.uint8), expected.view(np.uint8))
                            records.append({"fixture": fixture, "query": qi, "capacity": capacity,
                                "warm": warm, "raw_seconds": samples,
                                "process_cpu_seconds": [b.scalar_summary(row) for row in cpu_samples],
                                "before": b.scalar_summary(samples[0]), "hpr": b.scalar_summary(samples[1]),
                                "final": b.scalar_summary(samples[2]), "bits_stats_equal": True})
            finally:
                os.close(fd)
    halo._preflight_chunk_actions, cache._cache_access_plan = final_halo, final_cache
    args.output.write_text(json.dumps({"environment": b.environment_record(), "records": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
