"""Build a checkpointed current proxy with native or explicitly weighted seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import simesh as sm


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    with sm.open_amrvac(args.snapshot) as source:
        mesh = source.mesh
    if args.seeds:
        with np.load(args.seeds) as archive:
            points = sm.PointSet(archive['positions'],ids=archive['ids'] if 'ids' in archive else None)
            areas = archive['areas'].copy()
    else:
        points,areas = sm.native_bottom_seeds(mesh)
    if args.pilot:
        indices = np.random.default_rng(20260916).choice(len(points),min(args.pilot,len(points)),replace=False)
        points,areas = sm.PointSet(points.positions[indices],ids=points.ids[indices]),areas[indices]
    lower = mesh.lower.copy() if args.bounds is None else np.asarray(args.bounds[:3])
    upper = mesh.upper.copy() if args.bounds is None else np.asarray(args.bounds[3:])
    shape = tuple(args.resolution)
    if not np.isfinite(args.step_fraction) or not 0 < args.step_fraction <= 1:
        raise ValueError('step_fraction must be in (0, 1]')
    step = float(min(np.min((upper-lower)/shape),
                     np.max(mesh.spacing.min(axis=1))*args.step_fraction))
    snapshot = args.snapshot.resolve()
    seed_hash = hashlib.sha256(points.positions.tobytes()+points.ids.tobytes()+areas.tobytes()).hexdigest()
    config = dict(format_version=3,snapshot=str(snapshot),size_bytes=snapshot.stat().st_size,mtime_ns=snapshot.stat().st_mtime_ns,
                  seed_count=len(points),seed_hash=seed_hash,native_default=args.seeds is None,
                  pilot=args.pilot,lower=lower.tolist(),upper=upper.tolist(),shape=list(shape),step=step,step_fraction=args.step_fraction,
                  max_steps=args.max_steps,seed_batch=args.seed_batch,scheme='exact-phase',
                  weighting='Native bottom-face area' if args.seeds is None else 'Explicit supplied areas',
                  proxy='Mean squared interpolated raw curl times seed area, once per visited display voxel')
    config_path = args.output/'configuration.json'
    try:
        existing = json.loads(config_path.read_text())
    except FileNotFoundError:
        write_json(config_path,config)
    else:
        if existing!=config:
            raise ValueError('Output belongs to different controls, input or checkpoint format; use a new directory')
    np.savez_compressed(args.output/'seeds.npz',positions=points.positions,ids=points.ids,areas=areas)
    volume = np.zeros(shape)
    visits = np.zeros(shape,np.int64)
    means = np.full(len(points),np.nan)
    closed = np.zeros(len(points),bool)
    accepted = np.zeros(len(points),bool)
    termination = np.full((len(points),2),sm.LineSet.NOT_REQUESTED,np.int64)
    faces = np.full((len(points),2),sm.Boundary.NONE,np.int64)
    lengths = np.zeros(len(points))
    folder = args.output/'batches'; folder.mkdir(exist_ok=True)
    print(f'Loaded geometry in {time.monotonic()-started:.1f}s; {len(points)} seeds, step cap={step:g}, local fraction={args.step_fraction:g}',flush=True)
    integration_started = time.monotonic()
    magnetic = curl = None
    closed_count = accepted_count = 0
    for first in range(0,len(points),args.seed_batch):
        last = min(first+args.seed_batch,len(points))
        path = folder/f'{first:06d}-{last:06d}.npz'
        try:
            with np.load(path) as archive:
                saved = {key:archive[key] for key in archive.files}
        except FileNotFoundError:
            if magnetic is None:
                with sm.open_amrvac(args.snapshot) as source:
                    magnetic = sm.prepare(source,('b1','b2','b3'),scheme='exact-phase',workers=1,memory_limit=3*1024**3)
                curl = sm.curl(magnetic,workers=args.workers,memory_limit=3*1024**3)
            group = sm.PointSet(points.positions[first:last],ids=points.ids[first:last])
            batch = next(sm.iter_current_proxy(magnetic,shape,points=group,seed_areas=areas[first:last],
                bounds=(lower,upper),step=step,step_fraction=args.step_fraction,max_steps=args.max_steps,curl_field=curl,
                seed_batch=args.seed_batch,workers=args.workers))
            temporary = path.with_suffix('.tmp.npz')
            saved = dict(ids=batch.points.ids,voxel_ids=batch.voxel_ids,increments=batch.increments,
                visits=batch.visits,mean=batch.mean_current_squared,closed=batch.closed,
                accepted=batch.accepted,termination=batch.termination,faces=batch.endpoint_faces,length=batch.length)
            np.savez_compressed(temporary,**saved)
            temporary.replace(path)
        if not np.array_equal(saved['ids'],points.ids[first:last]):
            raise ValueError(f'Checkpoint seed IDs do not match: {path}')
        volume.ravel()[saved['voxel_ids']] += saved['increments']
        visits.ravel()[saved['voxel_ids']] += saved['visits']
        means[first:last],closed[first:last],accepted[first:last] = saved['mean'],saved['closed'],saved['accepted']
        termination[first:last],faces[first:last] = saved['termination'],saved['faces']
        lengths[first:last] = saved['length']
        closed_count += int(saved['closed'].sum())
        accepted_count += int(saved['accepted'].sum())
        elapsed = time.monotonic()-integration_started
        write_json(args.output/'progress.json',dict(completed=last,total=len(points),closed=closed_count,accepted=accepted_count,
            elapsed_seconds=elapsed,status='complete' if last==len(points) else 'running'))
        if first==0 or last%2048==0 or last==len(points):
            print(f'{last}/{len(points)} seeds; accepted {accepted_count}; {elapsed:.1f}s',flush=True)
    dx = (upper-lower)/shape
    np.savez_compressed(args.output/'proxy.npz',emissivity=volume,visits=visits,lower=lower,upper=upper,
                        proxy_xy=volume.sum(axis=2)*dx[2],proxy_xz=volume.sum(axis=1)*dx[1],
                        seed_ids=points.ids,mean=means,closed=closed,accepted=accepted,
                        termination=termination,faces=faces,length=lengths)
    codes,counts = np.unique(termination,return_counts=True)
    summary = dict(seeds=len(points),closed=closed_count,accepted=accepted_count,rejected=len(points)-accepted_count,
        termination_counts={str(int(c)):int(n) for c,n in zip(codes,counts)},
        unresolved_exit_branches=int(((termination==int(sm.Termination.DOMAIN_EXIT))&(faces==sm.Boundary.NONE)).sum()),
        mean_current_squared_percentiles=np.percentile(means[accepted],[0,50,95,99,100]).tolist() if accepted.any() else [],
        seed_area_sum=float(areas.sum()),volume_sum=float(volume.sum()),
        integration_seconds=time.monotonic()-integration_started,total_seconds=time.monotonic()-started)
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=Path,help='NPZ with positions, areas and optional ids')
    parser.add_argument('--resolution',type=int,nargs=3,default=(128,128,128))
    parser.add_argument('--bounds',type=float,nargs=6,help='xmin ymin zmin xmax ymax zmax; deposition only')
    parser.add_argument('--step-fraction',type=float,default=.25,
                        help='RK step cap as a fraction of the local minimum cell edge')
    parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--seed-batch',type=int,default=128)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--pilot',type=int,default=0,help='Reproducible seed subset for a pilot')
    run(parser.parse_args())
