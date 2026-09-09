"""隔离进程测量根子树导出的阶段时间、读取量与受控数组预算。"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import simesh as sm
from simesh.io import amrvac, products
from simesh.io._v5 import reader, writer


CASES = {
    'small-one': (((3, 2, 1), (4, 3, 2)), [0]),
    'contiguous-three': (((0, 0, 0), (4, 4, 4)), [7, 0, 3]),
    'reordered-three': (((1, 1, 0), (7, 5, 4)), [7, 0, 3]),
    'whole-eight': (((0, 0, 0), (8, 6, 4)), list(range(8))),
}


def generate(path):
    mesh = sm.mesh_from_forest((8, 6, 4), np.array(([False]+[True]*9)*96),
                              lower=(-2., -3., 1.), upper=(6., 3., 5.), block_shape=(16, 16, 16))
    raw = np.arange(mesh.leaf_count*8*4096, dtype=np.float64).reshape(mesh.leaf_count, 8, 16, 16, 16)
    header = dict(datfile_version=5, ndim=3, ndir=3, geometry='Cartesian_3D', periodic=[False]*3,
                  staggered=False, xmin=mesh.lower, xmax=mesh.upper,
                  domain_nx=mesh.root_shape*16, block_nx=mesh.block_shape,
                  time=1., it=1, physics_type='mhd', n_par=0, params=[], param_names=[],
                  snapshotnext=1, slicenext=0, collapsenext=0)
    with sm.source_from_arrays(mesh, raw, [f'f{i}' for i in range(8)], copy=False) as source:
        sm.write_amrvac(path, sm.read_fields(source), metadata=header)


class TimedStream:
    def __init__(self, stream, stats):
        self.stream, self.stats = stream, stats

    def tell(self):
        return self.stream.tell()

    def write(self, data):
        start = perf_counter()
        try:
            return self.stream.write(data)
        finally:
            self.stats['write_s'] += perf_counter()-start
            self.stats['write_bytes'] += len(data)


def measure(path, target, case, batch, mode):
    box, chosen = CASES[case]
    stats = dict(case=case, batch_request=batch, mode=mode, open_s=0., plan_s=0., read_s=0.,
                 pack_s=0., serialize_s=0., write_s=0., payload_read_bytes=0,
                 payload_read_calls=0, record_header_bytes=0, record_header_calls=0, write_bytes=0)
    held = None
    metadata = None
    if mode == 'fields':
        start = perf_counter()
        with sm.open_amrvac(path) as source:
            selected = sm.select_roots(source.mesh, box)
            held = sm.read_fields(source, chosen, region=selected)
            metadata = source.metadata
        stats['preload_s'] = perf_counter()-start
    with ExitStack() as stack:
        if batch:
            stack.enter_context(patch.object(products, '_BATCH_LEAVES', batch))
            stack.enter_context(patch.object(products, '_BATCH_BYTES', batch*len(chosen)*4096*8))
        def timed(owner, name, key):
            original = getattr(owner, name)
            def call(*args, **kwargs):
                start = perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    stats[key] += perf_counter()-start
            stack.enter_context(patch.object(owner, name, call))
        timed(amrvac, 'open_amrvac', 'open_s')
        timed(products, '_export_plan', 'plan_s')
        timed(sm.Source, 'read_into', 'read_s')
        original_pread = reader._pread_exact
        def pread(fd, size, offset, *, section):
            key = 'payload_read' if section == 'payload' else 'record_header'
            stats[key+'_bytes'] += size
            stats[key+'_calls'] += 1
            return original_pread(fd, size, offset, section=section)
        stack.enter_context(patch.object(reader, '_pread_exact', pread))
        original_serial = writer.write_datfile_from_batches
        def serialize(stream, *args):
            return original_serial(TimedStream(stream, stats), *args)
        stack.enter_context(patch.object(writer, 'write_datfile_from_batches', serialize))
        original_blocks = writer._write_blocks
        def blocks(stream, data):
            before_write = stats['write_s']
            start = perf_counter()
            original_blocks(stream, data)
            stats['serialize_s'] += perf_counter()-start-(stats['write_s']-before_write)
        stack.enter_context(patch.object(writer, '_write_blocks', blocks))
        original_fields = products._field_batches
        def fields(*args):
            iterator = original_fields(*args)
            while True:
                start = perf_counter()
                try:
                    batch_values = next(iterator)
                except StopIteration:
                    return
                stats['pack_s'] += perf_counter()-start
                yield batch_values
        stack.enter_context(patch.object(products, '_field_batches', fields))
        original_count = products._batch_count
        def count(n, block_bytes, required, limit, read_scratch=0):
            size = original_count(n, block_bytes, required, limit, read_scratch)
            stats.update(selected_leaves=n, batch_actual=size, payload_bytes=n*block_bytes,
                         batch_bytes=size*block_bytes,
                         controlled_upper_bytes=required+(size+1)*block_bytes+read_scratch)
            return size
        stack.enter_context(patch.object(products, '_batch_count', count))
        start = perf_counter()
        if mode == 'file':
            sm.crop_amrvac(path, target, root_bounds=box, fields=chosen)
        else:
            sm.write_amrvac(target, held, metadata=metadata, root_bounds=box)
        stats['total_s'] = perf_counter()-start
    phases = ('open_s', 'plan_s', 'read_s', 'pack_s', 'serialize_s', 'write_s')
    stats['other_s'] = stats['total_s'] - sum(stats[key] for key in phases)
    # macOS reports bytes; Linux reports KiB. Each sample runs in a fresh process.
    stats['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    stats['input_bytes'] = path.stat().st_size
    with sm.open_amrvac(target) as source:
        stats['output_leaves'] = source.mesh.leaf_count
    target.unlink()
    return stats


def run(directory):
    directory.mkdir(parents=True)
    path = directory/'input.dat'
    generate(path)
    rows = []
    configurations = [(case, 0, 'file') for case in CASES]
    configurations += [('reordered-three', batch, 'file') for batch in (1, 16, 64, 256)]
    configurations += [('reordered-three', 0, 'fields'), ('whole-eight', 0, 'fields')]
    for case, batch, mode in configurations:
        samples = []
        for repeat in range(3):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                     '--directory', str(directory), '--case', case,
                                     '--batch', str(batch), '--mode', mode], check=True,
                                    capture_output=True, text=True)
            sample = json.loads(result.stdout)
            sample['repeat'] = repeat
            samples.append(sample)
        rows.extend(samples)
        print(case, batch, mode, f"{np.median([s['total_s'] for s in samples]):.4f}s", flush=True)
    result = {'python': sys.version, 'numpy': np.__version__, 'platform': sys.platform,
              'input_shape': [864, 8, 16, 16, 16],
              'notes': '本地临时文件；未清理系统页缓存；耗时包含测量包装开销；write_s 不包含 fsync。',
              'samples': rows}
    (directory/'measurements.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', choices=CASES, default='reordered-three')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--mode', choices=('file', 'fields'), default='file')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure(args.directory/'input.dat', args.directory/'output.dat',
                                 args.case, args.batch, args.mode)))
    else:
        run(args.directory.resolve())
