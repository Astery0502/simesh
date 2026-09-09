"""Direct original-record startup and equal-result resident-source comparison."""

import time
IMPORT_START=time.perf_counter()
from contextlib import contextmanager
import os
import json
from pathlib import Path
import resource
import numpy as np

from simesh.analysis import open_source,open_prepared,prepare,PreparedPool,trace
from simesh.analysis.providers import make_source
from simesh.amrvac.datio import read_blocks_sequential
import simesh_rewrite.amrvac_dat as metadata
import simesh_rewrite.amrvac_dat_reader as reader_module
from simesh_rewrite.blockio import array_block_reader
IMPORT_SECONDS=time.perf_counter()-IMPORT_START


@contextmanager
def count_reads():
    stats={'metadata_calls':0,'metadata_bytes':0,'reader_calls':0,'reader_bytes':0}
    old_meta,old_read=metadata._pread_exact,reader_module._pread_exact
    def meta(*args,**kwargs):
        value=old_meta(*args,**kwargs)
        stats['metadata_calls']+=1;stats['metadata_bytes']+=len(value)
        return value
    def read(*args,**kwargs):
        value=old_read(*args,**kwargs)
        stats['reader_calls']+=1;stats['reader_bytes']+=len(value)
        return value
    metadata._pread_exact,reader_module._pread_exact=meta,read
    try:
        yield stats
    finally:
        metadata._pread_exact,reader_module._pread_exact=old_meta,old_read


def main():
    path='data/weno509_sub_0000.dat'
    start=time.perf_counter()
    with count_reads() as reads:
        with open_source(path,field_names=['b1','b2','b3']) as source:
            opened=time.perf_counter()
            mesh=source.mesh
            selected,_,_=mesh.select_box(mesh.lower+.42*(mesh.upper-mesh.lower),mesh.lower+.58*(mesh.upper-mesh.lower))
            ids=selected[np.linspace(0,len(selected)-1,8,dtype=np.int64)]
            seeds=np.ascontiguousarray(mesh.bounds[ids].mean(axis=1))
            step=float(.25*np.min(mesh.spacing[ids]))
            pool=PreparedPool(source,[0,1,2],16)
            direct=trace(pool,seeds,step=step,max_steps=8)
            finished=time.perf_counter()
            first_reads=reads.copy()
            direct_bound=pool.controlled_bytes
            ordinary=prepare(source,ids,[2,0,1],halo=0)
            definitions=source.fields
            pool.close()
    eager_start=time.perf_counter()
    fd=os.open(path,os.O_RDONLY)
    try:
        index=metadata.read_amrvac_v5_index(fd)
        binding=metadata.bind_amrvac_v5_forest(index)
    finally:
        os.close(fd)
    backing=read_blocks_sequential(path,[4,5,6])
    np.testing.assert_array_equal(ordinary.values.view(np.uint64),np.moveaxis(backing[ids][:,[2,0,1]],1,-1).view(np.uint64))
    eager_source=make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
        index.domain_lower,index.domain_upper,index.block_cell_counts,array_block_reader(backing),definitions)
    eager_pool=PreparedPool(eager_source,[0,1,2],16)
    reference=trace(eager_pool,seeds,step=step,max_steps=8)
    eager_time=time.perf_counter()-eager_start
    for name in ('positions','steps','termination','length'):
        np.testing.assert_array_equal(getattr(direct,name),getattr(reference,name))
    eager_bound=eager_pool.controlled_bytes
    eager_pool.close()
    del eager_source,backing
    whole_start=time.perf_counter()
    with count_reads() as whole_reads:
        full=open_prepared(path,field_names=['b1','b2','b3'])
    whole_seconds=time.perf_counter()-whole_start
    result={'source':path,'source_bytes':Path(path).stat().st_size,'library_import_seconds':IMPORT_SECONDS,
        'direct_open_seconds':opened-start,'direct_file_to_short_trace_seconds':finished-start,
        'direct_reads':first_reads,'direct_pool_controlled_bytes':direct_bound,
        'eager_file_to_same_trace_seconds':eager_time,'eager_pool_controlled_bytes':eager_bound,
        'selected_interior_bits_equal':True,'trace_arrays_equal':True,
        'full_resident_file_seconds':whole_seconds,'full_resident_stats':full.preparation_stats,
        'full_resident_reads':whole_reads,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'same immutable original WENO file and short trace; no bridge or tail payload decoding'}
    Path('benchmark-results/analysis-core/native-source.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
