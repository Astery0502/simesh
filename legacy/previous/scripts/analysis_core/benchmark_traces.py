"""P2 serial/thread and cache comparisons, explicitly requested real-data run."""

from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from simesh.analysis import FieldDefinition, PreparedPool, prepare, trace
from simesh.amrvac.datio import read_blocks_sequential
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source
from analysis_core.benchmark_prepared import measure, compare
from test_prepared import source_fixture
import sle_001 as sle


def assert_same(a,b):
    for name in ("positions","length","steps","termination","seed_ids"):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def main():
    result = {"revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "dirty":True,"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,
        "strategy":"rk4-arclength-accepted-prefix-v1","workers":"Python threads, nogil kernel",
        "cache":"application cleared/warm; OS page cache uncontrolled"}
    def helix(x,y,z):
        return np.array([-(y-5),x-3,np.full_like(x,.4)])
    start = time.perf_counter()
    synthetic,_ = source_fixture(helix)
    ready = prepare(synthetic,np.arange(synthetic.mesh.leaf_count),[0,1,2])
    phases = np.linspace(0,2*np.pi,2048,endpoint=False)
    seeds = np.ascontiguousarray(np.column_stack((3+np.cos(phases),5+np.sin(phases),np.zeros(2048))))
    result["synthetic_setup_seconds"] = time.perf_counter()-start
    one = trace(ready,seeds,step=.005,max_steps=600,workers=1,seed_batch=2048)
    four = trace(ready,seeds,step=.005,max_steps=600,workers=4,seed_batch=2048)
    assert_same(one,four)
    result["synthetic"] = {"seeds":2048,"steps":600,
        "one":measure(lambda:trace(ready,seeds,step=.005,max_steps=600,workers=1,seed_batch=2048),3),
        "four":measure(lambda:trace(ready,seeds,step=.005,max_steps=600,workers=4,seed_batch=2048),3),
        "prepared_bytes":ready.nbytes,"mesh_bytes":ready.mesh.nbytes,
        "result_bytes":sum(a.nbytes for a in vars(one).values() if isinstance(a,np.ndarray)),
        "private_stage_upper_bytes":2048*512}
    del ready, synthetic
    path = "data/weno509_sub_0000.dat"
    start = time.perf_counter()
    fd = os.open(path,os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(fd)
        binding = bind_amrvac_v5_forest(index)
    finally:
        os.close(fd)
    result["weno_index_seconds"] = time.perf_counter()-start
    start = time.perf_counter()
    backing = read_blocks_sequential(path,[4,5,6])
    result["weno_read_seconds"] = time.perf_counter()-start
    reader = array_block_reader(backing)
    start = time.perf_counter()
    source = make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
        index.domain_lower,index.domain_upper,index.block_cell_counts,reader,
        tuple(FieldDefinition(n,"code") for n in ("b1","b2","b3")),support_capacity=128)
    result["weno_adapter_seconds"] = time.perf_counter()-start
    mesh = source.mesh
    lo = mesh.lower+.42*(mesh.upper-mesh.lower)
    hi = mesh.lower+.58*(mesh.upper-mesh.lower)
    selected,_,_ = mesh.select_box(lo,hi)
    result["weno"] = []
    for count,steps,capacity in ((8,8,16),(32,128,64)):
        chosen = selected[np.linspace(0,len(selected)-1,count,dtype=np.int64)]
        seeds = np.ascontiguousarray(mesh.bounds[chosen].mean(axis=1))
        step = float(.25*np.min(mesh.spacing[chosen]))
        reference = None
        for slots in ((capacity,4) if count==8 else (capacity,)):
            for workers in (1,4):
                pool = PreparedPool(source,[0,1,2],slots)
                kwargs = dict(step=step,max_steps=steps,workers=workers,seed_batch=min(count,slots))
                for warm in (False,True):
                    observations = []
                    def run():
                        if not warm:
                            pool.clear()
                        before = pool.prepared_count
                        output = trace(pool,seeds,**kwargs)
                        if reference is not None:
                            assert_same(output,reference)
                        observations.append(pool.prepared_count-before)
                        return output
                    output = run()
                    if reference is None:
                        reference = output
                    timing = measure(run,3)
                    result["weno"].append({"seeds":count,"max_steps":steps,"step":step,
                        "slots":slots,"workers":workers,"warm":warm,"timing":timing,
                        "prepared_counts":observations,"steps_sum":int(output.steps.sum()),
                        "termination":np.unique(output.termination,return_counts=True)[1].tolist(),
                        "termination_codes":np.unique(output.termination).tolist(),
                        "pool_controlled_bytes":pool.controlled_bytes,
                        "misses":int(output.misses.sum()),"samples":int(output.samples.sum())})
                pool.close()
        if count == 8:
            case = sle.case_from_index(index,binding,"WENO",np.arange(3,dtype=np.int64))
            old = sle.make_completed_halo_sampling_session(*sle.session_arguments(case,reader,16))
            spec = sle.Trajectory("P2-short",seeds,np.ones(count,dtype=np.int8),step,steps,False,count)
            arrays = sle.allocate_outputs(spec)
            start = time.perf_counter()
            sle.run_trajectory(old,spec,arrays)
            first = time.perf_counter()-start
            end = arrays[0][np.arange(count),arrays[2]-1]
            check = compare(reference.positions,end)
            if not check["within_tolerance"]:
                raise AssertionError(check)
            np.testing.assert_array_equal(reference.steps,arrays[2]-1)
            result["rewrite_short"] = {"position_comparison":check,"first_seconds":first,
                "warm":measure(lambda:sle.run_trajectory(old,spec,arrays),3),
                "scope":"different work: one primary halo, trajectories and B-dot-dx vs two halos and arclength summaries"}
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    target = Path("benchmark-results/analysis-core/traces.json")
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"synthetic_speedup":result["synthetic"]["one"]["median"]/result["synthetic"]["four"]["median"],
        "weno":[[r["seeds"],r["slots"],r["workers"],r["warm"],r["timing"]["median"]] for r in result["weno"]],
        "rewrite":result["rewrite_short"]},indent=2))


if __name__ == "__main__":
    main()
