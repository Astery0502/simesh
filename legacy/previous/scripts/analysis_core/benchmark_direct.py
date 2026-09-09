"""Compare direct prepared workflows and same-runner tracing before/after.

Save the previous native extension, providers.py and global_fields.py, then run
with --baseline-dir and a fresh --output path. OS page cache is uncontrolled.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

from simesh.analysis import open_source, open_prepared, prepare, prepare_region, trace, with_curl, global_curl, curl, sample_plane, Plane, PreparedPool
from simesh.utils.lib.analysis import native
from simesh.analysis import providers
from simesh_rewrite.blockio import array_block_reader
from simesh_rewrite.forest import RefinedForest
from test_prepared import source_fixture


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_result(actual, expected):
    for name, value in vars(expected).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(actual, name), value)


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def paired(before, after, check, repeats=5):
    reference = before()
    check(after(), reference)
    rows = {"before": [], "after": []}
    for repeat in range(repeats):
        order = (("before", before), ("after", after))
        for name, call in (order if repeat % 2 == 0 else reversed(order)):
            value, seconds = timed(call)
            check(value, reference)
            rows[name].append(seconds)
            del value
    return {**rows, "medians": {name: float(np.median(values)) for name, values in rows.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weno", type=Path, default=Path("data/weno509_sub_0000.dat"))
    parser.add_argument("--full-curl", action="store_true", help="also compare all WENO curl values and two slices")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old_native = load_module("direct_baseline.native", next(args.baseline_dir.glob("native*.so")))
    old_global = load_module("simesh.analysis._baseline_global", args.baseline_dir / "global_fields.py")
    old_provider = load_module("simesh.analysis._baseline_provider", args.baseline_dir / "providers.py")
    result = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "dirty": True, "platform": platform.platform(), "python": sys.version,
              "numpy": np.__version__, "openmp": native.openmp_build_info(),
              "cache": "same-process interleaved repeats; OS page cache uncontrolled"}

    def old_trace(fields, seeds, **kwargs):
        name = "simesh.utils.lib.analysis.native"
        sys.modules[name] = old_native
        try:
            return trace(fields, seeds, **kwargs)
        finally:
            sys.modules[name] = native

    @contextmanager
    def old_file_source(path):
        factory = providers.make_source
        providers.make_source = old_provider.make_source
        try:
            with open_source(path,field_names=['b1','b2','b3'],support_capacity=256) as original:
                providers.make_source = factory
                yield original
        finally:
            providers.make_source = factory

    def helix(x, y, z):
        return np.array([-(y-5), x-3, np.full_like(x, .4)])

    source, fixture = source_fixture(helix)
    f = fixture
    forest = RefinedForest(f.node_levels,f.node_coords,np.full(len(f.node_levels),-1,dtype=np.int64),
        f.child_node_ids,f.node_leaf_ids,f.leaf_node_ids,f.root_node_ids,f.max_level)
    original_source = old_provider.make_source(f.root_shape,f.coord_to_rank,forest,
        f.domain_lower,f.domain_upper,f.block_counts,array_block_reader(f.backing),source.fields)
    fields, setup = timed(lambda: prepare(source, np.arange(source.mesh.leaf_count), [0, 1, 2]))
    phases = np.linspace(0, 2*np.pi, 2048, endpoint=False)
    seeds = np.ascontiguousarray(np.column_stack((3+np.cos(phases), 5+np.sin(phases), np.zeros(2048))))
    result["helix_prepare_seconds"] = setup
    result["helix"] = {}
    for workers in (1, 4):
        kwargs = dict(step=.005, max_steps=600, seed_batch=2048, workers=workers)
        result["helix"][str(workers)] = paired(
            lambda: old_trace(fields, seeds, **kwargs), lambda: trace(fields, seeds, **kwargs), same_result)
    coupled = with_curl(fields)
    result["helix_twist"] = paired(
        lambda: old_trace(coupled, seeds[:128], step=.005, max_steps=600, twist=True, trajectories=True),
        lambda: trace(coupled, seeds[:128], step=.005, max_steps=600, twist=True, trajectories=True), same_result)
    result["synthetic_global_curl"] = paired(
        lambda: old_global.global_curl(original_source, batch_size=13),
        lambda: global_curl(source, batch_size=13),
        lambda actual, expected: np.testing.assert_array_equal(actual.values, expected.values), repeats=3)
    del coupled, fields, source, original_source, fixture, f, forest
    print("manufactured comparisons complete", flush=True)

    start = time.perf_counter()
    with open_source(args.weno, field_names=['b1', 'b2', 'b3'], support_capacity=256) as source:
        opened = time.perf_counter()
        mesh = source.mesh
        width = mesh.upper-mesh.lower
        middle, _, _ = mesh.select_box(mesh.lower+.46*width, mesh.lower+.54*width)
        ids = middle[np.linspace(0, len(middle)-1, 256, dtype=int)]
        seeds = np.ascontiguousarray(mesh.bounds[ids].mean(axis=1))
        step = float(.25*np.min(mesh.spacing[ids]))
        kwargs = dict(step=step, max_steps=128, workers=1, seed_batch=256)
        # The declared maximum arclength bounds every normalized-RK stage.
        # Select this region before reading values, without following misses.
        reach = kwargs['max_steps']*step
        bounds = np.array([seeds.min(axis=0)-reach, seeds.max(axis=0)+reach])
        ready, preparation = timed(lambda: prepare_region(source, bounds, [0, 1, 2]))
        output, first_query = timed(lambda: trace(ready, seeds, **kwargs))
        if np.any(output.termination == 7):
            raise AssertionError("chosen regional tracing fixture has insufficient coverage")
        result["weno_region"] = {"file": str(args.weno), "file_bytes": args.weno.stat().st_size,
            "bounds": bounds.tolist(), "leaves": len(ready.leaf_ids), "all_leaves": mesh.leaf_count,
            "source_open_seconds": opened-start, "preparation_seconds": preparation,
            "first_query_seconds": first_query, "file_to_first_seconds": time.perf_counter()-start,
            "prepared_bytes": ready.nbytes, "source_resident_bytes": source.resident_bytes,
            "mesh_bytes": mesh.nbytes, "preparation": ready.preparation_stats,
            "steps": int(output.steps.sum()), "termination": np.unique(output.termination, return_counts=True)[1].tolist(),
            "trace": paired(lambda: old_trace(ready, seeds, **kwargs), lambda: trace(ready, seeds, **kwargs), same_result)}
        with old_file_source(args.weno) as original:
            result['weno_region']['preparation_comparison'] = paired(
                lambda: prepare_region(original,bounds,[0,1,2]),
                lambda: prepare_region(source,bounds,[0,1,2]),
                lambda actual, expected: np.testing.assert_array_equal(actual.values,expected.values),repeats=3)
        pool = PreparedPool(source,[0,1,2],256)
        pool_rows = []
        try:
            for repeat in range(4):
                before = pool.prepared_count
                pooled, seconds = timed(lambda: trace(pool,seeds,**kwargs))
                for name in ('seed_ids','seeds','positions','length','steps','termination'):
                    np.testing.assert_array_equal(getattr(pooled,name),getattr(output,name))
                pool_rows.append({'seconds':seconds,'prepared_leaves':pool.prepared_count-before})
            result['weno_region']['dynamic_pool_comparator'] = {
                'scope':'same results and current kernel, different retained field coverage',
                'controlled_bytes':pool.controlled_bytes,'runs':pool_rows}
        finally:
            pool.close()
        print("WENO region comparisons complete", flush=True)
    # Prepared values survive file closure and are reused for independent results.
    derived, derive_seconds = timed(lambda: curl(ready))
    plane = Plane(mesh.lower+width*[.46,.46,.5], width*[.08,0,0], width*[0,.08,0], (64,64))
    sliced, slice_seconds = timed(lambda: sample_plane(derived, plane))
    if not sliced.valid.all():
        raise AssertionError("regional curl slice is incomplete")
    result["weno_region"].update(curl_seconds=derive_seconds, slice_seconds=slice_seconds,
                                derived_bytes=derived.nbytes, slice_valid=True)
    # Check the public file entrypoint without retaining another full-domain field.
    reopened, reopen_seconds = timed(lambda: open_prepared(args.weno, field_names=['b1','b2','b3'], bounds=bounds))
    np.testing.assert_array_equal(reopened.values, ready.values)
    result["weno_region"]["public_region_seconds"] = reopen_seconds
    del reopened, ready, derived, sliced
    if args.full_curl:
        axis = Plane(mesh.lower+width*[0,0,.5],width*[1,0,0],width*[0,1,0],(128,128))
        oblique = Plane(mesh.lower+width*[0,0,.2],width*[1,0,.3],width*[0,1,.3],(96,80))
        def consume(call):
            field = call()
            return field, sample_plane(field,axis), sample_plane(field,oblique)
        def check(actual, expected):
            for first in range(0,mesh.leaf_count,128):
                np.testing.assert_array_equal(actual[0].values[first:first+128],expected[0].values[first:first+128])
            for a,b in zip(actual[1:],expected[1:]):
                np.testing.assert_array_equal(a.values,b.values)
                np.testing.assert_array_equal(a.valid,b.valid)
        with open_source(args.weno,field_names=['b1','b2','b3'],support_capacity=256) as source:
            with old_file_source(args.weno) as original:
                result['weno_full_curl'] = paired(
                    lambda: consume(lambda: old_global.global_curl(original,batch_size=256)),
                    lambda: consume(lambda: global_curl(source,batch_size=256)),check,repeats=3)
        print("full WENO curl and slices complete", flush=True)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
