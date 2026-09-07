"""Bounded P2 probe: actual long-trace working set versus a 64-slot cache."""
import json
import os
from pathlib import Path
import time
import numpy as np
from simesh.analysis import FieldDefinition, PreparedPool, trace
from simesh.amrvac.datio import read_blocks_sequential
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source
from analysis_core.benchmark_prepared import measure

path = 'data/weno509_sub_0000.dat'
start = time.perf_counter()
fd = os.open(path, os.O_RDONLY)
try:
    index = read_amrvac_v5_index(fd)
    binding = bind_amrvac_v5_forest(index)
finally:
    os.close(fd)
backing = read_blocks_sequential(path,[4,5,6])
source = make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
    index.domain_lower,index.domain_upper,index.block_cell_counts,array_block_reader(backing),
    tuple(FieldDefinition(n,'code') for n in ('b1','b2','b3')),support_capacity=128)
setup = time.perf_counter()-start
mesh = source.mesh
selected,_,_ = mesh.select_box(mesh.lower+.42*(mesh.upper-mesh.lower),
                              mesh.lower+.58*(mesh.upper-mesh.lower))
chosen = selected[np.linspace(0,len(selected)-1,32,dtype=np.int64)]
seeds = np.ascontiguousarray(mesh.bounds[chosen].mean(axis=1))
step = float(.25*np.min(mesh.spacing[chosen]))
rows, reference = [], None
for capacity in (64,256):
    pool = PreparedPool(source,[0,1,2],capacity)
    prepared_counts = []
    def run():
        before = pool.prepared_count
        output = trace(pool,seeds,step=step,max_steps=128,seed_batch=32)
        prepared_counts.append(pool.prepared_count-before)
        if reference is not None:
            np.testing.assert_array_equal(output.positions, reference.positions)
            np.testing.assert_array_equal(output.steps, reference.steps)
            np.testing.assert_array_equal(output.termination, reference.termination)
        return output
    start = time.perf_counter()
    output = run()
    cold = time.perf_counter()-start
    if reference is None:
        reference = output
    timing = measure(run,3)
    rows.append({'capacity':capacity,'first_seconds':cold,'warm':timing,
                 'prepared_counts':prepared_counts,'controlled_bytes':pool.controlled_bytes})
    pool.close()
result = {'source':path,'setup_seconds':setup,'seeds':32,'steps':128,'step':step,'rows':rows}
Path('benchmark-results/analysis-core/long-cache.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
