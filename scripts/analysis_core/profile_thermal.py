"""Profile historical thermal LOS; keep attribution separate from timings."""
import cProfile
import gc
import io
import json
from pathlib import Path
import pstats
import time
import numpy as np
from simesh.analysis import open_prepared,FieldDefinition,thermal_fields,orthographic_plane,integrate_thermal_los
from simesh.analysis.thermal import PROTON_MASS_G
from simesh.amrvac.analysis import prepare_resident


def weno_state():
    t=time.perf_counter()
    density=open_prepared('data/weno509_sub_0000.dat',field_names='rho')
    mesh=density.mesh
    loaded=time.perf_counter()
    raw=np.empty((mesh.leaf_count,1,*mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        z=mesh.bounds[leaf,0,2]+(np.arange(mesh.block_shape[2])+.5)*mesh.spacing[leaf,2]
        z=(z-mesh.lower[2])/(mesh.upper[2]-mesh.lower[2])
        raw[leaf,0]=1.05e6+6e5*np.sin(2*np.pi*z)
    temperature=prepare_resident(mesh,mesh.roots.shape,mesh.node_leaves>=0,raw,
        (FieldDefinition('external_T','K'),),budget_bytes=2*1024**3-density.nbytes)
    del raw
    state=thermal_fields(density,temperature,density_unit_g_cm3=1.4*PROTON_MASS_G*1e9,
        temperature_label='manufactured 0.45--1.65 MK sinusoid along normalized z')
    stats={'density_seconds':loaded-t,'temperature_and_state_seconds':time.perf_counter()-loaded,
        'file_to_thermal_seconds':time.perf_counter()-t,'state_bytes':state.nbytes,
        'admission':state.preparation_stats}
    del density,temperature
    gc.collect()
    return state,stats


def main():
    state,setup=weno_state()
    p=orthographic_plane(state.mesh.lower,state.mesh.upper,[.3,.2,1.],(32,32))
    profiler=cProfile.Profile()
    result=profiler.runcall(integrate_thermal_los,state,p,[.3,.2,1.],length_unit_cm=1e8,implementation='reference')
    out=Path('benchmark-results/analysis-core')
    profiler.dump_stats(out/'thermal-reference.prof')
    stream=io.StringIO()
    pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
    (out/'thermal-profile.txt').write_text(stream.getvalue())
    (out/'thermal-profile-setup.json').write_text(json.dumps(setup,indent=2)+'\n')
    print('complete',result.complete,'samples',result.samples.sum())
    print(stream.getvalue())

if __name__=='__main__': main()
