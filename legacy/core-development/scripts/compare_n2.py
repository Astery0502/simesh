"""Isolated geometry or full coordinate-phase workflow against a pinned donor."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dependencies", type=Path, required=True)
    p.add_argument("--flavor", choices=("donor", "new"), required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry", action="store_true")
    p.add_argument("--arrays", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--backend", choices=("threadpool", "openmp"), default="threadpool")
    args = p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError("run with -I -S")
    if args.output.exists() or (args.arrays is not None and args.arrays.exists()):
        raise FileExistsError("use new comparison outputs")
    sys.path[:0] = [str(args.source_root.resolve()), str(args.dependencies.resolve())]
    import numpy as np
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    import simesh
    assert Path(simesh.__file__).resolve().is_relative_to(args.source_root.resolve())
    if args.flavor == "donor":
        from simesh.analysis import open_prepared, curl, sample, sample_plane, trace, Plane
    else:
        from simesh import open_amrvac, prepare, curl, sample, sample_plane, trace, Plane
    imported = time.perf_counter()
    limit = 2*1024**3
    if args.geometry:
        if args.flavor == "donor":
            import os
            from simesh.amrvac._v5.index import read_amrvac_v5_index
            from simesh.utils.lib.amr.forest import AMRForest
            from simesh.utils.lib.amr.mesh import AMRMesh
            fd = os.open(args.file, os.O_RDONLY)
            try:
                index = read_amrvac_v5_index(fd)
            finally:
                os.close(fd)
            roots = index.domain_cell_counts//index.block_cell_counts
            forest = AMRForest(3, *map(np.uint32, roots), index.forest_flags.astype(np.int32))
            owner = AMRMesh(3, index.block_cell_counts.astype(np.uint32),
                            index.domain_cell_counts.astype(np.uint32),
                            index.domain_lower.copy(), index.domain_upper.copy(), 0, 1, forest)
            arrays = {"type": np.asarray(forest.neighbor_type), "index": np.asarray(forest.neighbor_index),
                      "children": np.asarray(forest.neighbor_children), "rnode": np.asarray(owner.rnode)}
        else:
            from simesh.preparation.coordinate import build_geometry
            with open_amrvac(args.file, memory_limit=limit) as source:
                geometry = build_geometry(source.mesh)
            arrays = {"type": geometry.neighbor_type, "index": geometry.neighbor_index,
                      "children": geometry.neighbor_children, "rnode": geometry.rnode}
        finish = time.perf_counter()
        record = {"geometry_seconds": finish-imported}
    else:
        if args.flavor == "donor":
            ready = open_prepared(args.file, field_names=("b1", "b2", "b3"),
                                  budget_bytes=limit, workers=args.workers, backend=args.backend)
        else:
            with open_amrvac(args.file, memory_limit=limit) as source:
                ready = prepare(source, ("b1", "b2", "b3"), scheme="coordinate-phase",
                                workers=args.workers, backend=args.backend, memory_limit=limit)
        prepared = time.perf_counter()
        mesh = ready.mesh
        width = mesh.upper-mesh.lower
        plane = Plane(mesh.lower+width*[.1, .1, .5], width*[.8, 0, 0], width*[0, .8, 0], (128, 128))
        oblique = Plane(mesh.lower+width*[.2, .15, .35], width*[.55, .1, .15], width*[.05, .6, -.1], (96, 80))
        options = {"workers": args.workers,
                   ("budget_bytes" if args.flavor == "donor" else "memory_limit"): limit}
        derived = curl(ready, **options)
        differentiated = time.perf_counter()
        first = sample_plane(derived, plane, **options)
        second = sample_plane(derived, oblique, **options)
        seeds = mesh.lower+(.3+.4*(np.indices((4,4,4)).reshape(3,-1).T+.5)/4)*width
        seeds = np.ascontiguousarray(seeds)
        sampled, owners, valid = sample(ready, seeds, workers=args.workers)
        lines = trace(ready, seeds, step=float(.125*np.min(mesh.spacing)), max_steps=32,
                      seed_batch=32, trajectories=True, **options)
        finish = time.perf_counter()
        assert first.valid.all() and second.valid.all() and valid.all()
        assert not np.any(lines.termination == 7)
        arrays = {"ready": ready.values, "derived": derived.values,
                  "axis": first.values, "oblique": second.values,
                  "axis_valid": first.valid, "oblique_valid": second.valid,
                  "samples": sampled, "owners": owners, "valid": valid,
                  "positions": lines.positions, "length": lines.length,
                  "steps": lines.steps, "termination": lines.termination,
                  "trace_samples": lines.samples, "trajectories": lines.trajectories}
        record = {"preparation_seconds": prepared-imported, "derivative_seconds": differentiated-prepared,
                  "other_consumers_seconds": finish-differentiated,
                  "preparation": ready.preparation_stats, "leaf_count": mesh.leaf_count,
                  "accepted_steps": int(lines.steps.sum()), "prepared_bytes": ready.nbytes,
                  "derived_bytes": derived.nbytes}
    after = resource.getrusage(resource.RUSAGE_SELF)
    hashes = {}
    for name, array in arrays.items():
        assert array.flags.c_contiguous, (name, array.shape)
        digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
        hashes[name] = digest.hexdigest()
    if args.arrays:
        # Use only for geometry/small cases; full WENO comparisons stream hashes.
        np.savez(args.arrays, **arrays)
    record.update(flavor=args.flavor, file=str(args.file), workers=args.workers, backend=args.backend,
                  import_seconds=imported-start, file_to_result_seconds=finish-imported,
                  startup_to_result_seconds=finish-start, hashes=hashes,
                  minor_faults=after.ru_minflt-before.ru_minflt,
                  major_faults=after.ru_majflt-before.ru_majflt, peak_rss_bytes=after.ru_maxrss)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({key: record[key] for key in ("flavor", "workers", "startup_to_result_seconds", "peak_rss_bytes")}))


if __name__ == "__main__":
    main()
