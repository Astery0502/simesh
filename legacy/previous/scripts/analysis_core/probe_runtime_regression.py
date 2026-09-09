"""Paired whole-query regression against bdfda30 Python and compiled consumers.

Build the independent native_before module from that revision as described in
runtime-execution.md before running. Both variants use identical ready inputs.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
from simesh.analysis import open_prepared,trace,integrate_los,orthographic_plane
import simesh.utils.lib.analysis.native as native
from analysis_core.probe_runtime import digest

REFERENCE=Path('benchmark-results/analysis-core/runtime-native-reference').resolve()
sys.path.insert(0,str(REFERENCE))
import native_before


def old_module(stem):
    name='simesh.analysis._'+stem
    spec=importlib.util.spec_from_file_location(name,REFERENCE/(stem+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    before_f=old_module('field_lines_before')
    before_l=old_module('los_before')
    rows=[]
    for workload in ('f','l'):
        ready=open_prepared('data/weno509_sub_0000.dat',field_names=['b1','b2','b3'] if workload=='f' else ['rho'])
        if workload=='f':
            selected,_,_=ready.mesh.select_box(ready.mesh.lower+.3*(ready.mesh.upper-ready.mesh.lower),
                                               ready.mesh.lower+.7*(ready.mesh.upper-ready.mesh.lower))
            ids=selected[np.linspace(0,len(selected)-1,2048,dtype=np.int64)]
            seeds=np.ascontiguousarray(ready.mesh.bounds[ids].mean(axis=1))
            step=float(.25*np.min(ready.mesh.spacing[ids]))
        else:
            direction=[.3,.2,1.]
            plane=orthographic_plane(ready.mesh.lower,ready.mesh.upper,direction,(256,256))
        ref=None
        for repeat in range(6):
            for old in ((True,False) if repeat%2==0 else (False,True)):
                stamp,cpu=time.perf_counter(),time.process_time()
                if workload=='f':
                    if old:
                        with patch.object(native,'advance_lines',native_before.advance_lines):
                            out=before_f.trace(ready,seeds,step=step,max_steps=256,seed_batch=256,workers=1)
                    else:
                        out=trace(ready,seeds,step=step,max_steps=256,seed_batch=256,workers=1)
                    arrays=(out.positions,out.length,out.steps,out.termination,out.samples)
                else:
                    if old:
                        with patch.object(native,'advance_rays',native_before.advance_rays):
                            out=before_l.integrate_los(ready,plane,direction,tile_shape=(64,64),workers=1)
                    else:
                        out=integrate_los(ready,plane,direction,tile_shape=(64,64),workers=1)
                    arrays=(out.values,out.entry,out.exit,out.status,out.samples)
                wall,used=time.perf_counter()-stamp,time.process_time()-cpu
                signature=digest(*arrays)
                if ref is not None and signature!=ref:
                    raise AssertionError('current query differs from pre-round result')
                ref=signature
                rows.append(dict(workload=workload,repeat=repeat,variant='before' if old else 'current',
                                 wall_seconds=wall,cpu_seconds=used,signature=signature))
                del arrays,out
        del ready
        print(workload,'before/current matched',flush=True)
    Path('benchmark-results/analysis-core/runtime-regression.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
