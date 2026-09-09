"""Run real-snapshot magnetic/line/density workflows without inferred SI units.

Usage: .venv/bin/python scripts/run_representative_workflow.py DATA --output DIRECTORY
The output directory must be new. Products use code values and coordinates.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import simesh as sm
from simesh import applications as app


def json_value(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True)
    report={'input':{},'controls':vars(args).copy(),'stages':{},'checks':{},'products':[],
            'python':platform.python_version(),'numpy':np.__version__,'simesh':sm.__version__}
    report['controls'].update(path=str(args.path.resolve()),output=str(output))
    report['revision']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report['working_tree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    def checkpoint():
        (output/'summary.json').write_text(json.dumps(report,indent=2,default=json_value,allow_nan=False)+'\n')
    def stage(name,operation):
        print(f'Start: {name}',flush=True)
        started=time.perf_counter()
        result=operation()
        elapsed=time.perf_counter()-started
        report['stages'][name]={'seconds':elapsed}
        print(f'Done: {name}: {elapsed:.3f} s',flush=True)
        checkpoint()
        return result
    def persist(name,result):
        path=output/f'{name}.npz'
        sm.save_result(path,result,metadata={'workflow':name,'units':'stored code values and coordinates'},
                       source=report['input'])
        report['products'].append(path.name)
        return sm.load_result(path).result
    def histogram(status,enum):
        return dict(Counter(enum(int(i)).name if int(i)>=0 else 'NOT_REQUESTED' for i in status.ravel()))
    start_stat=args.path.stat()
    report['input']={'path':str(args.path.resolve()),'bytes':start_stat.st_size,
                     'sha256':stage('input_sha256',lambda:digest(args.path))}
    limit=int(args.memory_gib*1024**3)
    with sm.open_amrvac(args.path,memory_limit=limit) as source:
        mesh=source.mesh
        lo,hi=mesh.lower,mesh.upper
        width=hi-lo
        report['input'].update(header=dict(source.metadata.header),leaf_count=mesh.leaf_count,
            block_shape=mesh.block_shape,finest_spacing=mesh.spacing.min(axis=0))
        report['input']=json.loads(json.dumps(report['input'],default=json_value,allow_nan=False))
        names=tuple(f.name for f in source.fields)
        for field in ('rho','b1','b2','b3'):
            if field not in names:
                raise ValueError(f'required stored field missing: {field}')
        gamma=dict(zip(source.metadata.header.get('param_names',()),source.metadata.header.get('params',()))).get('gamma')
        mhd_reason='This magnetic/density workflow does not infer an energy model or physical normalization.'
        if 'e' not in names:
            mhd_reason+=' No stored energy field e is available.'
        if gamma is not None and gamma<=1:
            mhd_reason+=f' Stored gamma={gamma} is outside the current IdealMHD model.'
        report['scope']={'included':['magnetic maps','Q and twist','selected inward traces',
            'line profiles','uniform extraction','density volume integral','density column LOS',
            'NPZ roundtrip','v2 shard roundtrip'],
            'excluded':{'MHD_recovery':mhd_reason,
                        'thermal_LOS':'No measured/recovered temperature or supplied physical normalization.',
                        'SI_current_and_energy':'No supplied SI field or length normalization.'}}
        magnetic=stage('prepare_magnetic_full_domain',lambda:sm.prepare(source,('b1','b2','b3'),
            scheme='coordinate-phase',workers=args.workers,memory_limit=limit))
        report['magnetic_preparation']=magnetic.preparation_stats
        report['magnetic_payload_bytes']=magnetic.values.nbytes
        plane=sm.Plane(lo+[0,0,.1]*width,[width[0],0,0],[0,width[1],0],(128,128))
        section=stage('magnetic_section',lambda:app.field_map(magnetic,plane,workers=args.workers,memory_limit=limit))
        assert section.valid.all() and section.usable.all()
        restored=persist('magnetic-section',section)
        np.testing.assert_array_equal(restored.values,section.values)
        del restored
        curl=stage('curl_full_domain',lambda:sm.curl(magnetic,workers=args.workers,memory_limit=limit))
        curl_map=stage('curl_section',lambda:app.field_map(curl,plane,workers=args.workers,memory_limit=limit))
        assert curl_map.usable.all()
        persist('curl-section',curl_map)
        divergence=stage('divergence_full_domain',lambda:sm.divergence(magnetic,workers=args.workers,
            memory_limit=limit-curl.nbytes))
        div_map=app.field_map(divergence,plane,workers=args.workers,memory_limit=limit-magnetic.nbytes-curl.nbytes)
        assert div_map.usable.all()
        persist('divergence-section',div_map)
        del divergence
        gc.collect()
        points=sm.PointSet.boundary(mesh,'zmin',(args.map_size,args.map_size))
        diagnostic=stage('bottom_Q_twist',lambda:app.connectivity(magnetic,points,quantities=('q','twist'),
            method='finite-difference',normalization='mapping',step_fraction=.25,max_steps=args.max_steps,
            curl_field=curl,workers=args.workers,seed_batch=64,memory_limit=limit))
        restored=persist('bottom-diagnostics',diagnostic)
        for name in ('q','twist','footpoints','termination','valid'):
            np.testing.assert_array_equal(getattr(restored.data,name),getattr(diagnostic.data,name))
        del restored,curl
        gc.collect()
        data=diagnostic.data
        report['diagnostics']={'seeds':len(points),'complete':int(data.complete.sum()),
            'valid_q':int(data.valid.sum()),'valid_twist':int(diagnostic.twist_valid.sum()),
            'termination':histogram(data.termination,sm.ConnectivityTermination)}
        # Rank finite Q and complete twist separately; this is output sampling,
        # not a physical threshold or a convergence study.
        good_q=np.flatnonzero(data.valid & np.isfinite(data.q))
        good_tw=np.flatnonzero(diagnostic.twist_valid)
        picked=set(good_q[np.argsort(data.q[good_q])[-args.lines//2:]].tolist())
        picked.update(good_tw[np.argsort(np.abs(data.twist[good_tw]))[-args.lines//2:]].tolist())
        if not picked:
            raise RuntimeError('no complete finite diagnostic seeds available for representative lines')
        mask=np.zeros(len(points),dtype=bool)
        mask[sorted(picked)]=True
        selected=points.select(mask)
        report['selection']={'policy':'union of highest finite Q and highest absolute complete twist',
                             'seed_ids':selected.ids,'count':len(selected)}
        trace_options=dict(direction='inward',step=float(mesh.spacing.min())*.125,
                           max_steps=args.max_steps,workers=args.workers)
        report['trace_controls']=trace_options
        lines=stage('trace_selected_collected',lambda:app.trace(magnetic,selected,seed_batch=8,
            memory_limit=limit,**trace_options))
        report['lines']={'points':len(lines.positions),'bytes':lines.nbytes,
                         'termination':histogram(lines.termination,sm.Termination)}
        restored=persist('selected-lines',lines)
        for name in ('positions','offsets','termination'):
            np.testing.assert_array_equal(getattr(restored,name),getattr(lines,name))
        del restored
        line_shards=stage('trace_selected_sharded',lambda:sm.save_result_shards(output/'line-shards',
            app.iter_lines(magnetic,selected,seed_batch=8,memory_limit=limit,**trace_options),
            seed_ids=selected.ids,source=report['input']))
        line_shards=sm.open_result_shards(line_shards.path)
        assert line_shards.complete
        for index in range(len(line_shards)):
            loaded=line_shards.load(index).result
            for seed in loaded.seeds.ids:
                for direction in (-1,1):
                    np.testing.assert_array_equal(loaded.branch(seed,direction),lines.branch(seed,direction))
        report['checks']['collected_and_sharded_lines_bitwise_equal']=True
        def line_batches():
            for i in range(len(line_shards)):
                yield line_shards.load(i).result
        magnetic_profiles=stage('magnetic_profile_shards',lambda:sm.save_result_shards(output/'magnetic-profiles',
            sm.iter_line_profiles(magnetic,line_batches(),('b1','b2','b3'),point_batch=4096,
                workers=args.workers,memory_limit=limit),seed_ids=selected.ids,source=report['input']))
        profile_points=0
        for i in range(len(magnetic_profiles)):
            profile=magnetic_profiles.load(i).result
            assert profile.usable.all()
            expected=sm.sample(magnetic,profile.lines.positions)[0]
            np.testing.assert_array_equal(profile.values,expected)
            profile_points+=len(profile.values)
        assert profile_points==len(lines.positions)
        report['checks']['magnetic_profiles_match_direct_sampling']=True
        del profile,expected
        uniform=stage('uniform_64_64_32',lambda:app.uniform_grid(magnetic,(64,64,32),
            workers=args.workers,memory_limit=limit))
        assert uniform.valid.all()
        restored=persist('magnetic-uniform',uniform)
        np.testing.assert_array_equal(restored.values,uniform.values)
        del uniform,restored,magnetic
        gc.collect()
        density=stage('prepare_density_full_domain',lambda:sm.prepare(source,('rho',),
            scheme='coordinate-phase',workers=args.workers,memory_limit=limit))
        mass=stage('density_volume_integral',lambda:sm.volume_integral(density,'rho'))
        extrema=stage('density_extrema',lambda:sm.extrema(density,'rho'))
        report['density']={'volume_integral':asdict(mass),'extrema':asdict(extrema)}
        rays=sm.RaySet.from_plane(sm.Plane(lo-[0,0,.05]*width,[width[0],0,0],
            [0,width[1],0],(64,64)),[0,0,1])
        column=stage('density_column_LOS',lambda:app.los(density,rays,component=0,
            workers=args.workers,memory_limit=limit))
        assert column.complete
        restored=persist('density-column',column)
        np.testing.assert_array_equal(restored.values,column.values)
        del restored
        density_profiles=stage('density_profile_shards',lambda:sm.save_result_shards(output/'density-profiles',
            sm.iter_line_profiles(density,line_batches(),('rho',),point_batch=4096,
                workers=args.workers,memory_limit=limit),seed_ids=selected.ids,source=report['input']))
        for i in range(len(density_profiles)):
            profile=density_profiles.load(i).result
            assert profile.usable.all()
            np.testing.assert_array_equal(profile.values,sm.sample(density,profile.lines.positions)[0])
        report['checks']['density_profiles_match_direct_sampling']=True
        report['checks']['NPZ_roundtrips_bitwise_equal']=True
        report['checks']['all_profile_points_usable']=True
        report['shards']={'lines':len(line_shards),'magnetic_profiles':len(magnetic_profiles),
                          'density_profiles':len(density_profiles),'schema_version':line_shards.manifest['schema_version']}
        report['density_column']={'rays':len(rays.origins),'valid':int(column.valid.sum()),
            'minimum':float(column.values.min()),'maximum':float(column.values.max())}
        del density
        gc.collect()
        stage('render_figures',lambda:render(output,diagnostic,section,curl_map,div_map,column,lines,density_profiles,lo,hi))
        source.validate()
    end_stat=args.path.stat()
    assert (start_stat.st_size,start_stat.st_mtime_ns)==(end_stat.st_size,end_stat.st_mtime_ns)
    report['checks']['input_file_unchanged']=True
    report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    report['process_peak_rss_scope']='whole process high-water RSS, including plotting; not per-stage allocation'
    report['output_bytes']=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    report['complete']=True
    checkpoint()
    print(json.dumps({'complete':True,'output':str(output),'lines':report['lines'],
        'diagnostics':report['diagnostics'],'peak_rss_bytes':report['process_peak_rss_bytes']},indent=2),flush=True)
    return report


def render(output,diagnostic,section,curl_map,div_map,column,lines,density_profiles,lo,hi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    extent=[lo[0],hi[0],lo[1],hi[1]]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    maps=[(section.image[...,2],'Bz at z = 0.1 domain height','code B','RdBu_r'),
        (diagnostic.image('log10_q'),'Bottom log10 Q','log10 Q','magma'),
        (np.where(diagnostic.image('twist_valid'),diagnostic.image('twist'),np.nan),'Bottom twist (complete lines)','turns','RdBu_r'),
        (curl_map.image[...,2],'curl(B)z on section','code B / coordinate','RdBu_r'),
        (div_map.image[...,0],'div(B) on section','code B / coordinate','RdBu_r'),
        (column.image,'Density column along +z','code rho * coordinate','viridis')]
    maps[1]=(np.where(diagnostic.image('q_valid'),maps[1][0],np.nan),*maps[1][1:])
    for ax,(values,title,label,cmap) in zip(axes.ravel(),maps):
        options={}
        if cmap=='RdBu_r':
            bound=float(np.nanmax(np.abs(values)))
            if bound>0:
                options.update(vmin=-bound,vmax=bound)
        im=ax.imshow(values.T,origin='lower',extent=extent,cmap=cmap,aspect='equal',**options)
        ax.set(title=title,xlabel='x [coordinate]',ylabel='y [coordinate]')
        fig.colorbar(im,ax=ax,label=label,shrink=.85)
    fig.suptitle('WENO snapshot: stored fields and representative diagnostics')
    fig.savefig(output/'overview.png',dpi=160)
    plt.close(fig)
    fig=plt.figure(figsize=(12,5),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    for seed in lines.seeds.ids:
        path=lines.line(seed)
        ax.plot(*path[::max(1,len(path)//1500)].T,lw=.8)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
        xlabel='x [coordinate]',ylabel='y [coordinate]',zlabel='z [coordinate]',title='Selected inward trajectories')
    ax.set_box_aspect(hi-lo)
    ax=fig.add_subplot(122)
    for i in range(len(density_profiles)):
        profile=density_profiles.load(i).result
        for seed in profile.seed_ids[:4]:
            for direction in (-1,1):
                branch=profile.branch(int(seed),direction)
                if len(branch.arclength):
                    ax.plot(branch.arclength,branch.values[:,0],lw=.8,label=str(seed))
    ax.set(xlabel='Arclength [coordinate]',ylabel='Density [code]',title='Density along representative branches')
    fig.savefig(output/'lines-and-profiles.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--map-size',type=int,default=32)
    parser.add_argument('--lines',type=int,default=32)
    parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--memory-gib',type=float,default=2.)
    args=parser.parse_args()
    if (args.workers<1 or args.map_size<1 or args.lines<2 or args.lines%2 or args.max_steps<1
            or not np.isfinite(args.memory_gib) or args.memory_gib<=0):
        parser.error('positive controls and an even line count >=2 are required')
    run(args)
