"""Bounded WENO E1/E5 comparisons; run one --stage at a time."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
from simesh.analysis import (open_source, prepare, PreparedPool, trace, sample,
    curl, integrate_los, orthographic_plane, Plane)
from simesh.analysis.geometry_plans import build_fill_plan
from simesh.analysis.thermal import ray_segments, ray_nodes
from analysis_core.rebricking import build_bricks, pack_bricks, sample_bricks, curl_bricks, trace_samples
from analysis_core.benchmark_prepared import measure, compare


def requests(source):
    mesh = source.mesh
    ids = np.linspace(0,mesh.leaf_count-1,8,dtype=np.int64)
    seeds = mesh.bounds[ids].mean(axis=1)
    step = float(mesh.spacing[ids].min()*.25)
    touched = []
    def fill(ids,fields,halo,out):
        touched.extend(ids.tolist())
        return source.fill(ids,fields,halo,out)
    probe = replace(source,fill=fill)
    pool = PreparedPool(probe,[0,1,2],256)
    t = time.perf_counter()
    reference = trace(pool,seeds,step=step,max_steps=128)
    reference_seconds = time.perf_counter()-t
    pool.close()
    sparse = np.unique(touched)
    centers = mesh.bounds.mean(axis=1)
    distance = np.linalg.norm((centers-(mesh.lower+mesh.upper)/2)/(mesh.upper-mesh.lower),axis=1)
    dense = np.sort(np.argsort(distance)[:768])
    full = orthographic_plane(mesh.lower,mesh.upper,[0.,0.,1.],(8,8))
    plane = Plane(full.origin+.4375*(full.u+full.v),full.u*.125,full.v*.125,full.shape)
    direction = np.array([0.,0.,1.])
    points,weights,pixels,ray_leaves = [],[],[],[]
    t = time.perf_counter()
    for pixel in np.ndindex(plane.shape):
        origin = plane.origin+(pixel[0]+.5)/8*plane.u+(pixel[1]+.5)/8*plane.v
        leaves,first,last = ray_segments(mesh,origin,direction,0.,np.inf)
        for leaf,lo,hi in zip(leaves,first,last):
            nodes,w = ray_nodes(mesh,leaf,origin,direction,lo,hi,1)
            p = origin+nodes[:,None]*direction
            p = np.maximum(mesh.bounds[leaf,0],np.minimum(p,np.nextafter(mesh.bounds[leaf,1],mesh.bounds[leaf,0])))
            points.extend(p); weights.extend(w); pixels.extend([pixel[0]*8+pixel[1]]*len(w)); ray_leaves.append(leaf)
    return {"sparse":sparse,"dense":np.union1d(dense,ray_leaves),"derivative_ids":dense,
        "seeds":seeds,"step":step,"plane":plane,"points":np.array(points),"weights":np.array(weights),
        "pixels":np.array(pixels),"reference_positions":reference.positions,
        "reference_steps":reference.steps,"trace_file_to_result_seconds":reference_seconds,
        "ray_plan_seconds":time.perf_counter()-t}


def consumers(sampler, derivative, source, request, kind):
    if kind=="sparse":
        pos,steps = trace_samples(sampler,source.mesh,request["seeds"],request["step"],128)
        return pos,steps
    d = derivative()
    vals,_,valid = sampler(request["points"])
    if not valid.all(): raise AssertionError("missing dense LOS samples")
    image = np.bincount(request["pixels"],weights=vals[:,0]*request["weights"],minlength=64)
    return d,image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['bricks','plans'],required=True)
    args = parser.parse_args()
    result = {"stage":args.stage,"fixture":"data/weno509_sub_0000.dat",
        "baseline":"bdfda30 source/prepare and current unmodified numerical kernels",
        "request":"8 spread seeds/128 RK4 steps; 768 center leaves curl plus 8x8 central 1/8-width full-depth LOS",
        "cache":"fresh application preparation, OS cache uncontrolled",
        "comparison":"matched vectorized RK4 and direct sampled Gauss2; native compiled products separately labelled",
        "variants":[],"geometry":[]}
    t = time.perf_counter()
    with open_source(result["fixture"],field_names=['b1','b2','b3'],support_capacity=256) as source:
        result["source_open_seconds"] = time.perf_counter()-t
        req = requests(source)
        result["requests"] = {k:len(req[k]) for k in ('sparse','dense','derivative_ids','points')}
        result["ray_plan_seconds"] = req["ray_plan_seconds"]
        result["request_array_bytes"] = sum(v.nbytes for v in req.values() if isinstance(v,np.ndarray))
        result["initial_native_sparse_file_to_result_seconds"] = req['trace_file_to_result_seconds']
        if args.stage=='bricks':
            plans = [build_bricks(source.mesh,shape) for shape in ((1,1,1),(2,1,1),(2,2,2))]
            result['geometry'] = [{"merge_shape":p.merge_shape,**p.storage_report()} for p in plans]
        else: plans = []
        for kind in ('sparse','dense'):
            ids = req[kind]
            t = time.perf_counter()
            baseline = prepare(source,ids,[0,1,2])
            baseline_seconds = time.perf_counter()-t
            native_bricks = pack_bricks(build_bricks(source.mesh,(1,1,1)),baseline)
            reference = consumers(lambda p:sample(baseline,p),
                lambda:curl_bricks(native_bricks,req['derivative_ids']),source,req,kind)
            if kind=='sparse':
                check = compare(reference[0],req['reference_positions'])
                if not check['within_tolerance']: raise AssertionError(check)
                np.testing.assert_array_equal(reference[1],req['reference_steps'])
                compiled = measure(lambda:trace(baseline,req['seeds'],step=req['step'],max_steps=128),3)
            else:
                d = curl(baseline)
                slots = d.slot_of_leaf[req['derivative_ids']]
                interior = d.values[slots,1:-1,1:-1,1:-1]
                if not compare(reference[0],interior)['within_tolerance']: raise AssertionError('curl mismatch')
                native_los = integrate_los(baseline,req['plane'],[0.,0.,1.])
                if not native_los.complete or not compare(reference[1],native_los.values.ravel())['within_tolerance']:
                    raise AssertionError('LOS mismatch')
                compiled = {"curl_with_extra_valid_halo":measure(lambda:curl(baseline),3),
                            "scalar_los":measure(lambda:integrate_los(baseline,req['plane'],[0.,0.,1.]),3)}
                del d,interior
            base_consumption = measure(lambda:consumers(lambda p:sample(baseline,p),
                lambda:curl_bricks(native_bricks,req['derivative_ids']),source,req,kind),3)
            if args.stage=='bricks':
                for plan in plans:
                    expanded = plan.expand(ids)
                    t = time.perf_counter()
                    prepared = prepare(source,expanded,[0,1,2])
                    prep_seconds = time.perf_counter()-t
                    t = time.perf_counter()
                    packed = pack_bricks(plan,prepared)
                    pack_seconds = time.perf_counter()-t
                    # Check all requested two-halo windows, not only raw values.
                    maximum = 0.
                    for leaf in ids:
                        a = packed.window(leaf,halo=2)
                        b = baseline.values[baseline.slot_of_leaf[leaf]]
                        maximum = max(maximum,float(np.max(np.abs(a-b))))
                        if not np.allclose(a,b,rtol=1e-10,atol=1e-10): raise AssertionError('mapped halo mismatch')
                    def consume():
                        return consumers(lambda p:sample_bricks(packed,p),
                            lambda:curl_bricks(packed,req['derivative_ids']),source,req,kind)
                    actual = consume()
                    checks = [compare(a,b) for a,b in zip(actual,reference)]
                    if not all(x['within_tolerance'] for x in checks): raise AssertionError(checks)
                    timing = measure(consume,3)
                    result['variants'].append({"request":kind,"merge_shape":plan.merge_shape,
                        "requested_leaves":len(ids),"expanded_leaves":len(expanded),"primary_amplification":len(expanded)/len(ids),
                        "prepare_seconds":prep_seconds,"pack_seconds":pack_seconds,"consumer":timing,
                        "first_complete_seconds":plan.build_seconds+prep_seconds+pack_seconds+timing['median'],
                        "native_first_prepare_seconds":baseline_seconds,"native_consumer":base_consumption,
                        "compiled_native_consumer":compiled,"prepared_stats":prepared.preparation_stats,
                        "native_prepared_stats":baseline.preparation_stats,"packed_bytes":packed.nbytes,
                        "native_expanded_bytes":prepared.nbytes,"native_requested_bytes":baseline.nbytes,
                        "conversion_live_bytes":prepared.nbytes+packed.nbytes+plan.nbytes+source.resident_bytes+source.mesh.nbytes,
                        "max_halo_difference":maximum,"consumer_comparisons":checks})
                    del packed,prepared,actual
                    print('E1 complete',kind,plan.merge_shape,flush=True)
            else:
                plan = build_fill_plan(source,ids,capacity=256)
                t = time.perf_counter()
                prepared = plan.prepare(source,[0,1,2])
                first = time.perf_counter()-t
                np.testing.assert_array_equal(prepared.values,baseline.values)
                new = consumers(lambda p:sample(prepared,p),
                    lambda:curl_bricks(pack_bricks(build_bricks(source.mesh,(1,1,1)),prepared),req['derivative_ids']),source,req,kind)
                checks = [compare(a,b) for a,b in zip(new,reference)]
                if not all(x['within_tolerance'] for x in checks): raise AssertionError(checks)
                repeated = measure(lambda:plan.prepare(source,[0,1,2]),3)
                unretained = measure(lambda:prepare(source,ids,[0,1,2]),3)
                result['variants'].append({"request":kind,"primary_leaves":len(ids),"plan_bytes":plan.nbytes,
                    "plan_build_seconds":plan.build_seconds,"first_execute_seconds":first,
                    "planned_repeat":repeated,"unretained_repeat":unretained,"consumer":base_consumption,
                    "planned_first_complete_seconds":plan.build_seconds+first+base_consumption['median'],
                    "unretained_first_complete_seconds":baseline_seconds+base_consumption['median'],
                    "planned_repeat_complete_seconds":repeated['median']+base_consumption['median'],
                    "unretained_repeat_complete_seconds":unretained['median']+base_consumption['median'],
                    "prepared_stats":prepared.preparation_stats,"native_prepared_stats":baseline.preparation_stats,
                    "consumer_comparisons":checks})
                del prepared,plan,new
                print('E5 complete',kind,flush=True)
            del baseline,native_bricks,reference
            gc.collect()
            Path(f'benchmark-results/analysis-core/{args.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(f'benchmark-results/analysis-core/{args.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
