"""Compare pinned donor and independent N1 workflows, one process at a time."""

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
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    if args.repeats < 1:
        raise ValueError("at least one comparison pair required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    dependencies = Path(sysconfig.get_path("purelib"))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    records = []
    for repeat in range(args.repeats):
        for flavor in (("donor", "new") if repeat % 2 == 0 else ("new", "donor")):
            output = args.output_dir/f"{flavor}-{repeat}.json"
            command = [sys.executable, "-I", "-S", str(root/"scripts/compare_n1.py"),
                       "--source-root", str((args.donor/"src") if flavor == "donor" else root/"src"),
                       "--dependencies", str(dependencies), "--flavor", flavor,
                       "--file", str(args.file.resolve()), "--output", str(output)]
            if repeat == 0:
                command += ["--arrays", str(args.output_dir/f"{flavor}.npz")]
            subprocess.run(command, env=env, check=True)
            record = json.loads(output.read_text())
            record["repeat"] = repeat
            records.append(record)
            if record["hashes"] != records[0]["hashes"]:
                raise AssertionError(f"full output mismatch: {flavor}, repeat {repeat}")
    # Full array comparison complements the repeat digests; no tolerance change.
    with np.load(args.output_dir/"donor.npz") as old, np.load(args.output_dir/"new.npz") as new:
        for name in old.files:
            np.testing.assert_array_equal(old[name], new[name], err_msg=name)
    summary = {"scope": "N1 exact-phase region, raw selector, curl/slice, samples, 64x64-step traces",
               "cache": "first run separate; operating-system cache uncontrolled",
               "arrays_compared": list(records[0]["hashes"]), "runs": records, "medians": {}}
    for flavor in ("donor", "new"):
        timed = [r for r in records if r["flavor"] == flavor and (r["repeat"] > 0 or args.repeats == 1)]
        summary["medians"][flavor] = {key: statistics.median(r[key] for r in timed)
            for key in ("import_seconds", "open_seconds", "prepare_and_raw_read_seconds", "consumer_seconds",
                        "file_to_result_seconds", "startup_to_result_seconds")}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["medians"], indent=2), flush=True)


if __name__ == "__main__":
    main()
