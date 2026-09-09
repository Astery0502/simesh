"""Run one frozen N1 comparison case in an explicitly isolated interpreter."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--flavor", choices=("donor", "new"), required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.arrays and args.arrays.exists()):
        raise FileExistsError("comparison outputs must be new")
    if not sys.flags.no_site or not sys.flags.isolated:
        raise RuntimeError("run with -I -S to prevent mixed package imports")
    sys.path[:0] = [str(args.source_root.resolve()), str(args.dependencies.resolve())]
    import numpy as np
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    import simesh
    if args.flavor == "donor":
        from simesh.analysis import open_source, prepare_region, curl, sample, trace, Plane, sample_plane
    else:
        from simesh import open_amrvac, prepare, curl, sample, trace, Plane, sample_plane
    loaded_from = Path(simesh.__file__).resolve()
    if not loaded_from.is_relative_to(args.source_root.resolve()):
        raise AssertionError(f"wrong package loaded: {loaded_from}")
    limit = 2*1024**3
    imported = time.perf_counter()
    context = (open_source(args.file, field_names=("b1", "b2", "b3"), support_capacity=128,
                           budget_bytes=limit) if args.flavor == "donor"
               else open_amrvac(args.file, memory_limit=limit))
    with context as source:
        opened = time.perf_counter()
        mesh = source.mesh
        width = mesh.upper-mesh.lower
        region = np.array([mesh.lower+.40*width, mesh.lower+.60*width])
        raw_ids = np.unique([mesh.leaf_count-1, 0, mesh.leaf_count//2]).astype(np.int64)[::-1].copy()
        if args.flavor == "donor":
            raw = np.empty((len(raw_ids), 2, *mesh.block_shape))
            source.read_interiors(raw_ids, np.array([2, 0], dtype=np.int64), raw)
            raw = np.ascontiguousarray(np.moveaxis(raw, 1, -1))
            ready = prepare_region(source, region, [0, 1, 2], budget_bytes=limit)
        else:
            from simesh import read_fields
            raw = read_fields(source, ("b3", "b1"), leaf_ids=raw_ids, memory_limit=limit).interior()
            ready = prepare(source, ("b1", "b2", "b3"), region=region, scheme="exact-phase",
                            memory_limit=limit, support_capacity=128)
        prepared = time.perf_counter()
    coordinates = np.indices((4, 4, 4)).reshape(3, -1).T
    seeds = np.ascontiguousarray(mesh.lower+(.47+.06*(coordinates+.5)/4)*width)
    step = float(.125*np.min(mesh.spacing))
    plane = Plane(mesh.lower+width*[.44, .44, .5], width*[.12, 0, 0], width*[0, .12, 0], (48, 40))
    options = {"workers": 4, ("budget_bytes" if args.flavor == "donor" else "memory_limit"): limit}
    derived = curl(ready, **options)
    image = sample_plane(derived, plane, **options)
    sampled, owners, valid = sample(ready, seeds, workers=4)
    lines = trace(ready, seeds, step=step, max_steps=64, seed_batch=32, trajectories=True, **options)
    finish = time.perf_counter()
    after = resource.getrusage(resource.RUSAGE_SELF)
    if not image.valid.all() or not valid.all() or np.any(lines.termination == 7):
        raise AssertionError("frozen comparison coverage must be complete")
    arrays = dict(raw=raw, ready=ready.values, leaf_ids=ready.leaf_ids, derived=derived.values,
                  image=image.values, image_valid=image.valid, image_owners=image.owners,
                  samples=sampled, sample_owners=owners, sample_valid=valid,
                  positions=lines.positions, length=lines.length, steps=lines.steps,
                  termination=lines.termination, trace_samples=lines.samples,
                  misses=lines.misses, trajectories=lines.trajectories)
    def digest(array):
        array = np.ascontiguousarray(array)
        value = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
        value.update(memoryview(array).cast("B"))
        return value.hexdigest()
    record = {"flavor": args.flavor, "source_root": str(args.source_root),
              "file": str(args.file), "file_size": args.file.stat().st_size,
              "numpy": np.__version__, "import_seconds": imported-start,
              "open_seconds": opened-imported,
              "prepare_and_raw_read_seconds": prepared-opened,
              "consumer_seconds": finish-prepared, "file_to_result_seconds": finish-imported,
              "startup_to_result_seconds": finish-start,
              "timing_start": "NumPy ready; import simesh through completed consumer results",
              "leaf_count": mesh.leaf_count, "target_count": len(ready.leaf_ids),
              "accepted_steps": int(lines.steps.sum()), "preparation": ready.preparation_stats,
              "minor_faults": after.ru_minflt-before.ru_minflt,
              "major_faults": after.ru_majflt-before.ru_majflt,
              "peak_rss_bytes": after.ru_maxrss,
              "hashes": {name: digest(array) for name, array in arrays.items()}}
    if args.arrays:
        np.savez(args.arrays, **arrays)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ("flavor", "target_count", "accepted_steps",
                                            "file_to_result_seconds", "startup_to_result_seconds")}))


if __name__ == "__main__":
    main()
