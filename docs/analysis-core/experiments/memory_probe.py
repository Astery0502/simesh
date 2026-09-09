"""开发用内存探针；从 analysis-core 目录以其虚拟环境运行。"""
import gc
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
import numpy as np
import simesh as sm
from simesh.bounded import iter_prepared

mode, side = sys.argv[1], int(sys.argv[2])
directory = Path(sys.argv[3])
directory.mkdir(parents=True, exist_ok=True)
path = directory / f'uniform-{side}.dat'
if mode == 'create':
    sys.path.insert(0, str(Path.cwd() / 'tests'))
    from fixtures import write_dat
    mesh = sm.mesh_from_forest((side,)*3, np.ones(side**3, dtype=bool), lower=(0,)*3,
                              upper=(1,)*3, block_shape=(8,)*3)
    values = np.ones((mesh.leaf_count,3,8,8,8))
    write_dat(path,mesh,values)
    print(json.dumps({'created':str(path), 'leaves':side**3}))
else:
    with sm.open_amrvac(path) as source:
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        if mode == 'raw':
            result = sm.read_fields(source)
            np.testing.assert_array_equal(result.interior(),1.)
            controlled = None
            retained = result.nbytes
            count = len(result.leaf_ids)
        elif mode in ('coordinate','exact'):
            result = sm.prepare(source,scheme='coordinate-phase' if mode == 'coordinate' else 'exact-phase',support_capacity=57)
            np.testing.assert_array_equal(result.interior(),1.)
            controlled = result.preparation_stats['controlled_upper_bytes']
            retained = result.nbytes
            count = len(result.leaf_ids)
        elif mode == 'bounded':
            batches = iter_prepared(source,scheme='exact-phase',batch_size=32,support_capacity=57)
            count = 0
            for _ in range(2):
                result = next(batches)
                np.testing.assert_array_equal(result.interior(),1.)
                controlled = result.preparation_stats['controlled_upper_bytes']
                retained = result.nbytes
                count += len(result.leaf_ids)
            batches.close()
        else:
            raise ValueError(mode)
        _, peak = tracemalloc.get_traced_memory()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            rss *= 1024
        print(json.dumps({'mode':mode,'leaves':source.mesh.leaf_count,'visited_leaves':count,
            'mesh_bytes':source.mesh.nbytes,'source_bytes':source.nbytes,
            'retained_field_bytes':retained,'controlled_upper_bytes':controlled,
            'traced_peak_bytes':peak,'process_peak_rss_bytes':rss,'seconds':time.perf_counter()-started}))
