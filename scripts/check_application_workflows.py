"""Exercise advertised native workflows and report local thread scaling.

Run from a source checkout. Timings describe this warmed synthetic fixture,
including Python orchestration, and are not large-file performance guarantees.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import simesh as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from fixtures import write_dat


def measure(operation, repeats):
    operation()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        times.append(time.perf_counter()-start)
    return result, {"median_seconds":float(np.median(times)), "runs_seconds":times}


def error_from(operation):
    try:
        operation()
    except Exception as exc:
        return {"type":type(exc).__name__, "message":str(exc)}
    return None


def main(args):
    k, alpha = 1.2, .6
    decay = np.sqrt(k*k-alpha*alpha)
    mesh = sm.mesh_from_forest((2,1,1), np.array([False]+[True]*9),
                              lower=(-1,-1,0), upper=(1,1,1), block_shape=(16,16,16))
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count,3,*mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        x,y,z = mesh.bounds[leaf,0,:,None,None,None]+local*mesh.spacing[leaf,:,None,None,None]
        values[leaf] = np.array([decay*np.cos(k*x), alpha*np.cos(k*x), -k*np.sin(k*x)])*np.exp(-decay*z)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty = bool(subprocess.check_output(["git","-C",str(root),"status","--porcelain","--","src","tests","examples","scripts","setup.py","pyproject.toml"],text=True).strip())
    report = {"revision":revision, "working_tree_dirty":dirty,"revision_scope":"root-package",
              "platform":platform.platform(),
              "fixture":{"leaves":mesh.leaf_count,"block_shape":mesh.block_shape,
                         "field":"linear force-free arcade", "alpha":alpha,
                         "map_shape":[32,24],"uniform_shape":[96,96,96]},
              "timings":{}, "checks":{}}
    with tempfile.TemporaryDirectory(prefix="simesh-app-audit-") as directory:
        path = Path(directory)/"arcade.dat"
        write_dat(path, mesh, values)
        report["fixture"]["file_bytes"] = path.stat().st_size
        with sm.open_amrvac(path) as source:
            for workers in (1,2,4):
                ready, timing = measure(lambda: sm.prepare(source, scheme="coordinate-phase", workers=workers), args.repeats)
                report["timings"].setdefault("prepare_coordinate",{})[workers] = timing
                if workers == 1:
                    baseline = ready.values.copy()
                else:
                    assert np.array_equal(baseline,ready.values)
            report["checks"]["exact_phase_workers_4"] = error_from(
                lambda: sm.prepare(source, scheme="exact-phase", workers=4))
            partial = sm.prepare(source, region=([-.9,-.8,0],[-.4,.8,.3]), scheme="exact-phase")
            iterator = sm.iter_prepared(source, scheme="exact-phase", batch_size=1)
            borrowed = next(iterator)
            next(iterator)
            report["checks"]["expired_borrow"] = error_from(lambda: borrowed.values)
            iterator.close()
    del baseline

    def uniform(workers):
        total, invalid = 0., 0
        for _, slab in sm.iter_uniform(ready,(96,96,96),workers=workers):
            total += float(slab.values.sum())
            invalid += int((~slab.valid).sum())
        return total, invalid

    plane = sm.Plane([-.8,-.2,0],[1.6,0,0],[0,.4,0],(32,24))
    u,v = (np.arange(n)+.5 for n in plane.shape)
    seeds = np.ascontiguousarray((plane.origin + u[:,None,None]/plane.shape[0]*plane.u +
                                 v[None,:,None]/plane.shape[1]*plane.v).reshape(-1,3))
    for workers in (1,2,4):
        summary,timing = measure(lambda: uniform(workers), args.repeats)
        report["timings"].setdefault("uniform_96_cube",{})[workers] = timing
        assert summary[1] == 0
        curl_b,timing = measure(lambda: sm.curl(ready,workers=workers), args.repeats)
        report["timings"].setdefault("curl",{})[workers] = timing
        if workers == 1:
            uniform_reference, curl_reference = summary, curl_b.values.copy()
        else:
            assert summary == uniform_reference and np.array_equal(curl_reference,curl_b.values)
    report["checks"]["uniform_valid_samples"] = 96**3
    b_squared,timing = measure(lambda: sm.derive(ready,"b_squared",
        lambda ctx: sum(ctx.field(name)**2 for name in ("b1","b2","b3"))), args.repeats)
    report["timings"]["derive_b_squared"] = {1:timing}
    gradient = sm.derivative(b_squared, [[("b_squared","x",1.)]], [sm.FieldDefinition("dx")],workers=4)
    section = sm.sample_plane(gradient,plane,workers=4)
    assert section.valid.all()
    second = sm.derivative(gradient, [[("dx","x",1.)]], [sm.FieldDefinition("dxx")])
    report["checks"]["halo_chain"] = [ready.valid_halo,b_squared.valid_halo,gradient.valid_halo,second.valid_halo]
    report["checks"]["sample_after_two_derivatives"] = error_from(lambda: sm.sample_plane(second,plane))

    for workers in (1,2,4):
        connectivity,timing = measure(lambda: sm.qsl(ready,seeds,curl_field=curl_b,
            workers=workers,step_fraction=.125,max_steps=4000), args.repeats)
        report["timings"].setdefault("qsl_bottom_map",{})[workers] = timing
        if workers == 1:
            map_reference = connectivity
        else:
            for name in ("q","q_perp","twist","footpoints","termination","valid"):
                assert np.array_equal(getattr(map_reference,name),getattr(connectivity,name),equal_nan=True)
    report["checks"]["parallel_outputs_equal"] = True
    report["checks"]["qsl_complete"] = int(connectivity.complete.sum())
    report["checks"]["qsl_valid"] = int(connectivity.valid.sum())
    report["checks"]["qsl_termination"] = dict(Counter(map(str,connectivity.termination.ravel())))
    closed = connectivity.valid & np.all(connectivity.boundary == sm.Boundary.ZMIN,axis=1)
    expected_q = 2+4*alpha*alpha/(decay*decay)
    tw_expected = alpha*connectivity.length/(4*np.pi)
    report["checks"]["closed_bottom_lines"] = int(closed.sum())
    report["checks"]["closed_q_expected"] = expected_q
    report["checks"]["closed_q_max_relative_error"] = float(np.max(np.abs(connectivity.q[closed]/expected_q-1)))
    report["checks"]["twist_vs_alpha_length_max_absolute_error"] = float(np.nanmax(np.abs(connectivity.twist-tw_expected)))

    # The closed arcade has Q=10/3: thresholds must exceed that background,
    # rather than selecting roundoff differences in a nearly constant map.
    q_cut, tw_cut = float(np.log10(4.)), .09
    selected = ((connectivity.valid & (connectivity.log10_q>=q_cut)) |
                (connectivity.complete & np.isfinite(connectivity.twist) & (np.abs(connectivity.twist)>=tw_cut)))
    ids = np.flatnonzero(selected)
    chosen = np.ascontiguousarray(connectivity.seeds[ids])
    bseed = sm.sample(ready,chosen)[0]
    trace_step = float(mesh.spacing.min())*.125
    for workers in (1,2,4):
        lines,timing = measure(lambda: sm.trace(ready,chosen,seed_ids=ids,step=trace_step,
            max_steps=4000,trajectories=True,workers=workers), args.repeats)
        report["timings"].setdefault("trace_selected_default",{})[workers] = timing
        if workers == 1:
            paths_reference = lines
        else:
            for name in ("termination","positions","trajectories","length"):
                assert np.array_equal(getattr(paths_reference,name),getattr(lines,name),equal_nan=True)
    report["checks"]["selection"] = {"count":len(ids),"log10_q_threshold":q_cut,"abs_twist_threshold":tw_cut,
        "high_q_count":int((connectivity.valid & (connectivity.log10_q>=q_cut)).sum()),
        "high_twist_count":int((connectivity.complete & (np.abs(connectivity.twist)>=tw_cut)).sum()),
        "positive_bz":int((bseed[:,2]>0).sum()),"negative_bz":int((bseed[:,2]<0).sum()),
        "default_zero_step_lines":int((lines.steps==0).sum()),"allocated_path_bytes":lines.trajectories.nbytes,
        "accepted_path_bytes":int(lines.point_counts.sum())*24}
    report["checks"]["qsl_to_retrace"] = error_from(lambda: sm.retrace(ready,connectivity,ids[:2],step=trace_step))
    def trace_inward(workers):
        return {direction:sm.trace(ready,np.ascontiguousarray(chosen[rows]),seed_ids=ids[rows],direction=direction,
                                  step=trace_step,max_steps=4000,trajectories=True,workers=workers)
                for direction in (1,-1)
                if len(rows := np.flatnonzero(direction*bseed[:,2]>0))}

    for workers in (1,2,4):
        inward_results,timing = measure(lambda: trace_inward(workers),args.repeats)
        report["timings"].setdefault("trace_selected_inward",{})[workers] = timing
        if workers == 1:
            inward_reference = inward_results
        else:
            for direction, inward in inward_results.items():
                assert np.array_equal(inward.trajectories,inward_reference[direction].trajectories,equal_nan=True)
    grouped = {}
    for direction,inward in inward_results.items():
        rows = np.flatnonzero(direction*bseed[:,2]>0)
        target = connectivity.footpoints[ids[rows], 1 if direction==1 else 0]
        grouped[direction] = {"seeds":len(rows),"zero_step_lines":int((inward.steps==0).sum()),
            "domain_exits":int((inward.termination==sm.Termination.DOMAIN_EXIT).sum()),
            "max_distance_to_qsl_endpoint":float(np.linalg.norm(inward.positions-target,axis=1).max())}
    report["checks"]["inward_grouped_trace"] = grouped
    report["checks"]["inward_paths_allocated_bytes"] = sum(r.trajectories.nbytes for r in inward_results.values())
    report["checks"]["inward_paths_accepted_bytes"] = sum(int(r.point_counts.sum())*24 for r in inward_results.values())
    upper_seed = np.array([[0.,0.,mesh.upper[2]]])
    report["checks"]["upper_face_trace_status"] = int(sm.trace(ready,upper_seed,step=trace_step).termination[0])
    report["checks"]["missing_qsl_coverage"] = sm.qsl(partial,seeds[:1],twist=False).termination.tolist()
    report["checks"]["memory_limit_refusal"] = error_from(lambda: sm.qsl(ready,seeds[:1],memory_limit=1))
    from simesh._kernels.native import openmp_build_info
    report["checks"]["openmp_build"] = openmp_build_info()
    for values_by_worker in report["timings"].values():
        base = values_by_worker[1]["median_seconds"]
        for timing in values_by_worker.values():
            timing["speedup_vs_one"] = base/timing["median_seconds"]
    text = json.dumps(report,indent=2)+"\n"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    main(args)
