"""Run the frozen M1 consumer benchmarks with retained or original preflight."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types
import time

import simesh_rewrite.refined_halo as halo


BASELINE = "397aaffc61be68d1141c8f30c0427dd37a4c8bc8"


def original_module(name):
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:rewrite/src/simesh_rewrite/{name}.py"],
        cwd=root, text=True,
    )
    module = types.ModuleType(f"simesh_rewrite._m1_original_{name}")
    module.__package__ = "simesh_rewrite"
    sys.modules[module.__name__] = module
    exec(compile(source, f"M1:{name}.py", "exec"), module.__dict__)
    return module


def sle_controls(output):
    """Interleave only controls with >10% drift in separate standard runs."""
    import sle_001 as b
    case = b.synthetic_case("constant")
    specs = b.trajectory_specs(case, 2, 32)
    baseline = original_module("refined_halo")._preflight_chunk_actions
    retained = halo._preflight_chunk_actions
    records = []
    try:
        for index, capacity, warm in ((0, 1, True), (1, 1, True),
                                      (1, case.leaf_count, False), (2, 1, False)):
            spec = specs[index]
            sessions = [b.make_completed_halo_sampling_session(*b.session_arguments(
                case, b.array_block_reader(case.backing), capacity)) for _ in range(2)]
            outputs = [b.allocate_outputs(spec) for _ in range(2)]
            samples = [[], []]
            for repeat in range(8):
                stats = [None, None]
                for variant in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    halo._preflight_chunk_actions = (baseline, retained)[variant]
                    b.clear_completed_halo_sampling_session(sessions[variant])
                    if warm:
                        b.run_trajectory(sessions[variant], spec, outputs[variant])
                    b.reset_outputs(outputs[variant])
                    start = time.perf_counter()
                    stats[variant] = b.run_trajectory(sessions[variant], spec, outputs[variant])
                    elapsed = time.perf_counter() - start
                    if repeat:
                        samples[variant].append(elapsed)
                assert stats[0] == stats[1]
                assert b.outputs_equal(*outputs)
            records.append({"case": spec.name, "capacity": capacity, "warm": warm,
                            "before": b.summary(samples[0]), "retained": b.summary(samples[1]),
                            "bits_and_stats_equal": True})
    finally:
        halo._preflight_chunk_actions = retained
    output.write_text(json.dumps({"environment": b.environment_record(),
                                  "records": records}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("lfe", "chs", "sle", "sle-controls"), required=True)
    parser.add_argument("--baseline", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.family == "sle-controls":
        controls = argparse.ArgumentParser()
        controls.add_argument("--output", type=Path, required=True)
        sle_controls(controls.parse_args(remaining).output)
        return
    if args.baseline:
        original = original_module("refined_halo")
        halo._preflight_chunk_actions = original._preflight_chunk_actions
    path = Path(__file__).with_name(f"{args.family}_001.py")
    sys.argv = [str(path), *remaining]
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
