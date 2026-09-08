"""Build a fresh source copy and exercise its wheel without editable path hooks."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import textwrap
import zipfile


def main():
    root=Path(__file__).resolve().parents[2]
    results=root/'benchmark-results/analysis-core'
    if shutil.disk_usage(root).free<2*1024**3:
        raise OSError('insufficient disk headroom for clean source build')
    stage=Path(tempfile.mkdtemp(prefix='source-build-',dir=results))
    source=stage/'source'
    source.mkdir()
    ignore=shutil.ignore_patterns('*.so','*.pyd','*.c','*.cpp','__pycache__','*.egg-info')
    shutil.copytree(root/'src',source/'src',ignore=ignore)
    shutil.copytree(root/'rewrite/src',source/'rewrite/src',ignore=ignore)
    for name in ('setup.py','build.py','pyproject.toml','MANIFEST.in','README.md','LICENSE'):
        shutil.copy2(root/name,source/name)
    wheels=stage/'wheels'
    wheels.mkdir()
    start=time.perf_counter()
    with (stage/'build.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pip','wheel','.', '--no-deps','--no-build-isolation',
                        '--wheel-dir',str(wheels)],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
    elapsed=time.perf_counter()-start
    wheel=next(wheels.glob('*.whl'))
    installed=stage/'installed'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    # -S suppresses site/.pth processing. Append dependency files without
    # activating the checkout's editable finder; source modules come from wheel.
    code='''
import sys, pathlib, json, tempfile
wheel, dependencies = sys.argv[1:3]
sys.path.insert(0,wheel)
sys.path.append(dependencies)
import numpy as np
import simesh, simesh_rewrite
from simesh.analysis import (open_source, PreparedPool, trace, open_prepared, sample,
                            global_curl_file, integrate_los_views, Plane,
                            build_fill_plan, thermal_fields, integrate_thermal_los)
from simesh.amrvac import write_datfile_from_uniform, read_blocks
assert str(simesh.__file__).startswith(wheel)
assert str(simesh_rewrite.__file__).startswith(wheel)
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory)/"snapshot.dat"
    data = np.zeros((8,8,8,3)); data[...,0]=1.
    write_datfile_from_uniform(path,data,["b1","b2","b3"],[0.,0.,0.],[1.,1.,1.],[4,4,4])
    with open_source(path,field_names=["b1","b2","b3"],value_cache_capacity=8) as source:
        pool=PreparedPool(source,[0,1,2],2)
        result=trace(pool,np.array([[.2,.4,.5]]),step=.05,max_steps=3)
        np.testing.assert_allclose(result.positions,[[.35,.4,.5]],atol=1e-14)
        pool.close()
        plan=build_fill_plan(source,np.arange(source.mesh.leaf_count))
        planned=plan.prepare(source,[0])
    fields=open_prepared(path,field_names=["b1","b2","b3"])
    np.testing.assert_array_equal(sample(fields,np.array([[.2,.4,.5]]))[0],[[1.,0.,0.]])
    assert read_blocks(path).shape == (8,3,4,4,4)
    derived=global_curl_file(path,workers=2,task_size=4,batch_size=2)
    np.testing.assert_array_equal(derived.values,0.)
    plane=Plane([0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],(4,4))
    images=integrate_los_views(fields,[plane,plane],[[0.,0.,1.],[0.,0.,1.]])
    for image in images:
        np.testing.assert_array_equal(image.values,1.)
    state=thermal_fields(planned,1.e6,density_unit_g_cm3=2.341670693166e-15,
                         temperature_label="isolated wheel manufactured isothermal input")
    thermal=integrate_thermal_los(state,plane,[0.,0.,1.],length_unit_cm=1.e8,workers=2)
    reference=integrate_thermal_los(state,plane,[0.,0.,1.],length_unit_cm=1.e8,
                                   implementation="reference")
    assert thermal.complete and reference.complete
    np.testing.assert_allclose(thermal.values,reference.values,rtol=1.e-13)
print(json.dumps({"simesh":simesh.__file__,"provider":simesh_rewrite.__file__,"result":result.positions.tolist()}))
'''
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None)
    smoke=stage/'smoke.py'
    smoke.write_text('def main():\n'+textwrap.indent(code,'    ')+"\nif __name__=='__main__':\n    main()\n")
    checked=subprocess.run([sys.executable,'-S',str(smoke),str(installed),sysconfig.get_paths()['purelib']],
        cwd=installed,env=environment,text=True,capture_output=True,check=True)
    record={'stage':str(stage),'wheel':str(wheel),'wheel_bytes':wheel.stat().st_size,
            'build_seconds':elapsed,'isolated_result':json.loads(checked.stdout),
            'source_extensions_preexisting':False}
    (results/'install.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
