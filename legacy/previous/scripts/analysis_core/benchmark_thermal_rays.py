"""Matched nonlinear LOS implementation/worker comparison, then actual 500^2."""
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
from simesh.analysis import integrate_thermal_los,orthographic_plane,emissivity_fields,integrate_los
from analysis_core.profile_thermal import weno_state
from analysis_core.benchmark_prepared import measure as _measure


def measure(fn,repeats=3):
    before=resource.getrusage(resource.RUSAGE_SELF)
    result=_measure(fn,repeats)
    after=resource.getrusage(resource.RUSAGE_SELF)
    result['major_faults_with_warmup']=after.ru_majflt-before.ru_majflt
    result['minor_faults_with_warmup']=after.ru_minflt-before.ru_minflt
    return result


def parity(actual,reference):
    np.testing.assert_array_equal(actual.status,reference.status)
    np.testing.assert_allclose(actual.values,reference.values,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(actual.depth,reference.depth,atol=2e-12,rtol=2e-13)
    delta=actual.values-reference.values
    return {'max_absolute':float(np.max(np.abs(delta))),
        'relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(reference.values)),
        'sample_count_difference':int(actual.samples.sum()-reference.samples.sum())}


def main():
    out=Path('benchmark-results/analysis-core')
    state,setup=weno_state()
    result={'platform':platform.platform(),'machine':platform.machine(),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source':'data/weno509_sub_0000.dat','physical_validation':False,
        'temperature':state.source[2],'setup':setup,'repeats':3,
        'profile':'thermodynamics-first, subdivisions=4; same native prepared nodes; no OpenMP',
        'matching':[],'large':[]}
    for direction in ([0.,0.,1.],[.3,.2,1.],[1.,-.8,.6]):
        plane=orthographic_plane(state.mesh.lower,state.mesh.upper,direction,(64,64))
        def call(implementation='native',workers=1):
            return integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,
                implementation=implementation,workers=workers)
        start=time.perf_counter()
        reference=call('reference')
        reference_first=time.perf_counter()-start
        if not reference.complete: raise AssertionError('incomplete reference')
        reference_timing=measure(lambda:call('reference'),3)
        row={'direction':direction,'shape':plane.shape,'reference_first_seconds':reference_first,
             'reference_repeat':reference_timing,'variants':[]}
        serial=None
        for workers in (1,2,4):
            start=time.perf_counter()
            image=call(workers=workers)
            first=time.perf_counter()-start
            check=parity(image,reference)
            if serial is None: serial=image
            else:
                np.testing.assert_array_equal(image.values,serial.values)
                np.testing.assert_array_equal(image.samples,serial.samples)
            timing=measure(lambda:call(workers=workers),3)
            row['variants'].append({'workers':workers,'first_seconds':first,'repeat':timing,
                'samples':int(image.samples.sum()),'parity':check,
                'speedup_vs_reference':reference_timing['median']/timing['median']})
        result['matching'].append(row)
        (out/'thermal-rays.json').write_text(json.dumps(result,indent=2)+'\n')
        print('matched 64x64 complete',direction,flush=True)
    # A complete true 500x500 image on each view, only after matching acceptance.
    images=[]
    for direction in ([0.,0.,1.],[.3,.2,1.],[1.,-.8,.6]):
        plane=orthographic_plane(state.mesh.lower,state.mesh.upper,direction,(500,500))
        row={'direction':direction,'shape':plane.shape,'variants':[]}
        serial=None
        for workers in (1,2,4):
            start=time.perf_counter()
            image=integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,workers=workers)
            first=time.perf_counter()-start
            if not image.complete: raise AssertionError(np.unique(image.status,return_counts=True))
            if serial is None: serial=image
            else:
                np.testing.assert_array_equal(image.values,serial.values)
                np.testing.assert_array_equal(image.status,serial.status)
                np.testing.assert_array_equal(image.samples,serial.samples)
            timing=measure(lambda:integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,workers=workers),3)
            row['variants'].append({'workers':workers,'first_seconds':first,'repeat':timing,
                'samples':int(image.samples.sum()),'nonempty_pixels':int(np.count_nonzero(image.depth)),
                'image_array_bytes':sum(v.nbytes for v in vars(image).values() if isinstance(v,np.ndarray))})
            print('500x500',direction,'workers',workers,'median',timing['median'],flush=True)
        images.append(serial.values)
        result['large'].append(row)
        (out/'thermal-rays.json').write_text(json.dumps(result,indent=2)+'\n')
    # Retained node-emissivity comparator preserves its separately named meaning.
    start=time.perf_counter()
    epsilon=emissivity_fields(state)
    result['emissivity_build_seconds']=time.perf_counter()-start
    result['emissivity_bytes']=epsilon.nbytes
    result['node_emissivity_views']=[]
    for direction,thermo_image in zip(([0.,0.,1.],[.3,.2,1.],[1.,-.8,.6]),images):
        plane=orthographic_plane(state.mesh.lower,state.mesh.upper,direction,(500,500))
        image=integrate_los(epsilon,plane,direction,workers=4,budget_bytes=2*1024**3-state.nbytes)
        if not image.complete: raise AssertionError('incomplete scalar comparator')
        result['node_emissivity_views'].append({'direction':direction,
            'four_workers':measure(lambda:integrate_los(epsilon,plane,direction,workers=4,budget_bytes=2*1024**3-state.nbytes),3),
            'relative_l2_to_thermo4':float(np.linalg.norm(image.values*1e8-thermo_image)/np.linalg.norm(thermo_image))})
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    np.savez_compressed(out/'thermal-500-images.npz',axis=images[0],oblique=images[1],diagonal=images[2],
        temperature=state.source[2],model=state.source[1],density_unit_g_cm3=state.preparation_stats['density_unit_g_cm3'],
        length_unit_cm=1e8,units='DN s^-1 pixel^-1',physical_validation=False)
    (out/'thermal-rays.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
