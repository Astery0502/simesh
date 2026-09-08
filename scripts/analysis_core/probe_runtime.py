"""Explicit same-layout WENO runtime attribution and scheduling probe.

Application-cold/warm refers to numerical caches; the OS page cache is uncontrolled.
Instrumentation is opt-in, uses nested wall timers, and is not a timing backend.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from unittest.mock import patch

import numpy as np
from simesh.analysis import (open_source, PreparedPool, trace, global_curl,
                             Plane, sample_plane, integrate_los, orthographic_plane)
from analysis_core.benchmark_native_source import count_reads


class Attribution:
    def __init__(self):
        self.stats = defaultdict(float)
        self.stack = ExitStack()

    def wrap(self, module, name, label):
        old = getattr(module, name)
        def run(*args, **kwargs):
            start = time.perf_counter()
            try:
                return old(*args, **kwargs)
            finally:
                self.stats[label+'_seconds'] += time.perf_counter()-start
                self.stats[label+'_calls'] += 1
        self.stack.enter_context(patch.object(module, name, run))

    def __enter__(self):
        import simesh_rewrite.refined_halo as rhe
        import simesh.utils.lib.analysis.native as native
        from concurrent.futures import Future, ThreadPoolExecutor
        self.wrap(rhe, '_prepare_refined_halo_chunk', 'planning')
        self.wrap(rhe, '_apply_chunk_actions_unchecked', 'ghost_apply')
        self.wrap(rhe, 'read_blocks_into', 'reader')
        for name in ('advance_lines', 'advance_rays', 'differentiate', 'sample_ready'):
            self.wrap(native, name, name)
        self.wrap(Future, 'result', 'future_wait')
        self.wrap(ThreadPoolExecutor, 'submit', 'submit')
        return self

    def source(self, source):
        old = source.fill
        def fill(ids, fields, halo, out):
            start = time.perf_counter()
            result = old(ids, fields, halo, out)
            self.stats['fill_seconds'] += time.perf_counter()-start
            self.stats['fill_calls'] += 1
            self.stats['prepared_owners'] += len(ids)
            for name in ('packing_seconds', 'selected_load_count', 'chunk_count', 'read_value_bytes','requested_value_bytes',
                         'value_cache_hits','value_cache_misses','value_cache_read_seconds',
                         'value_cache_copy_seconds'):
                self.stats[name] += result.get(name, 0)
            return result
        return replace(source, fill=fill)

    def __exit__(self, *args):
        self.stack.close()


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', choices=('f','d','l'), required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--backend', choices=('threadpool','openmp'), default='threadpool')
    p.add_argument('--schedule', choices=('static','dynamic'), default='static')
    p.add_argument('--pool', type=int, default=256)
    p.add_argument('--fill-batch', type=int, default=0)
    p.add_argument('--support', type=int, default=128)
    p.add_argument('--value-cache', type=int, default=0)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--tile', type=int, default=16)
    p.add_argument('--size', type=int, default=32)
    p.add_argument('--seeds', type=int, default=256)
    p.add_argument('--order', choices=('input','spatial'), default='input')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--instrument', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = {'parameters':vars(args), 'platform':platform.platform(),
              'cache':'numerical cold then warm; OS file cache uncontrolled', 'runs':[]}
    definitions = ['rho'] if args.workload=='l' else ['b1','b2','b3']
    clock = time.perf_counter()
    with count_reads() as io, ExitStack() as stack:
        source = stack.enter_context(open_source('data/weno509_sub_0000.dat',
            field_names=definitions, support_capacity=args.support,value_cache_capacity=args.value_cache))
        result['open_seconds'] = time.perf_counter()-clock
        result['open_reads'] = dict(io)
        meter = stack.enter_context(Attribution()) if args.instrument else None
        if meter:
            source = meter.source(source)
        mesh = source.mesh
        if args.workload=='f':
            selected,_,_ = mesh.select_box(mesh.lower+.42*(mesh.upper-mesh.lower),
                                          mesh.lower+.58*(mesh.upper-mesh.lower))
            chosen = selected[np.linspace(0,len(selected)-1,args.seeds,dtype=np.int64)]
            seeds = np.ascontiguousarray(mesh.bounds[chosen].mean(axis=1))
            # Same seeds and output order for every ordering candidate.
            perm = np.random.default_rng(177).permutation(len(seeds))
            seeds, chosen = seeds[perm],chosen[perm]
            order = np.argsort(chosen,kind='stable') if args.order=='spatial' else np.arange(len(seeds))
            inverse = np.argsort(order)
            step = float(.25*np.min(mesh.spacing[chosen]))
            result['step'] = step
        if args.workload in ('f','l'):
            pool = PreparedPool(source,range(len(definitions)),args.pool,fill_batch_size=args.fill_batch)
            stack.callback(pool.close)
            result['pool_controlled_bytes'] = pool.controlled_bytes
        if args.workload=='l':
            direction = [.3,.2,1.]
            plane = orthographic_plane(mesh.lower,mesh.upper,direction,(args.size,args.size))
        reference = None
        for repeat in range(args.repeats):
            before_io = dict(io)
            if meter:
                meter.stats.clear()
            fills = pool.prepared_count if args.workload!='d' else 0
            wall, cpu = time.perf_counter(), time.process_time()
            if args.workload=='f':
                out = trace(pool,np.ascontiguousarray(seeds[order]),seed_ids=order.astype(np.int64),
                    step=step,max_steps=128,seed_batch=args.batch,workers=args.workers,
                    backend=args.backend,schedule=args.schedule)
                signature = digest(*(getattr(out,name)[inverse] for name in
                    ('seed_ids','seeds','positions','length','steps','termination','samples')))
                details = {'steps':int(out.steps.sum()), 'misses':int(out.misses.sum())}
            elif args.workload=='l':
                out = integrate_los(pool,plane,direction,workers=args.workers,
                                    tile_shape=(args.tile,args.tile),backend=args.backend,schedule=args.schedule)
                if not out.complete:
                    raise AssertionError(np.unique(out.status,return_counts=True))
                signature = digest(out.values,out.entry,out.exit,out.status,out.samples)
                details = {'samples':int(out.samples.sum()),'misses':int(out.misses.sum())}
            else:
                out = global_curl(source,batch_size=args.pool)
                lower, upper = mesh.lower,mesh.upper
                extent=upper-lower
                slices = [sample_plane(out,Plane(lower+[0,0,.5*extent[2]],
                    [extent[0],0,0],[0,extent[1],0],(128,128))),
                    sample_plane(out,Plane(lower+[0,0,.25*extent[2]],
                    [extent[0],0,.25*extent[2]],[0,extent[1],.25*extent[2]],(128,128)))]
                signature = digest(out.values,*(s.values for s in slices))
                details = out.preparation_stats
            row = {'repeat':repeat,'wall_seconds':time.perf_counter()-wall,
                   'cpu_seconds':time.process_time()-cpu,'signature':signature,**details,
                   'io':{k:io[k]-before_io[k] for k in io}}
            if repeat==0:
                row['source_to_first_result_seconds']=time.perf_counter()-clock
            if reference is not None and signature!=reference:
                raise AssertionError('repeated result changed')
            reference=signature
            if args.workload!='d':
                row['prepared_owners'] = pool.prepared_count-fills
            if meter:
                row['attribution'] = dict(meter.stats)
            result['runs'].append(row)
            del out
            if args.workload=='d':
                del slices
            print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
