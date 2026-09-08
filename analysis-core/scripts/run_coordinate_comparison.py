"""Run matched coordinate-phase comparisons sequentially with bounded artifacts."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import sysconfig

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--geometry", action="store_true")
    p.add_argument("--small-arrays", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    records = []
    for repeat in range(args.repeats):
        for flavor in (("donor", "new") if repeat%2 == 0 else ("new", "donor")):
            output = args.output_dir/f"{flavor}-{repeat}.json"
            command = [sys.executable, "-I", "-S", str(root/"scripts/compare_n2.py"),
                       "--source-root", str(args.donor/"src" if flavor == "donor" else root/"src"),
                       "--dependencies", sysconfig.get_path("purelib"), "--flavor", flavor,
                       "--file", str(args.file.resolve()), "--output", str(output), "--workers", str(args.workers)]
            if args.geometry:
                command += ["--geometry"]
            if repeat == 0 and (args.geometry or args.small_arrays):
                command += ["--arrays", str(args.output_dir/f"{flavor}.npz")]
            subprocess.run(command, env=env, check=True)
            record = json.loads(output.read_text())
            record["repeat"] = repeat
            records.append(record)
            if record["hashes"] != records[0]["hashes"]:
                raise AssertionError(f"output mismatch: {flavor}, repeat {repeat}")
    if args.geometry or args.small_arrays:
        with np.load(args.output_dir/"donor.npz") as old, np.load(args.output_dir/"new.npz") as new:
            for name in old.files:
                np.testing.assert_array_equal(old[name], new[name], err_msg=name)
    metrics = ("import_seconds", "file_to_result_seconds", "startup_to_result_seconds")
    medians = {}
    for flavor in ("donor", "new"):
        timed = [r for r in records if r["flavor"] == flavor and (args.repeats == 1 or r["repeat"] > 0)]
        medians[flavor] = {key: statistics.median(r[key] for r in timed) for key in metrics}
    (args.output_dir/"summary.json").write_text(json.dumps({"runs": records, "medians": medians,
        "timing": "NumPy ready, package loading included; OS cache uncontrolled; one process at a time"}, indent=2)+"\n")
    print(json.dumps(medians, indent=2), flush=True)


if __name__ == "__main__":
    main()
