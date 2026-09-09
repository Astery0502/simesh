"""P3-D bounded comparison: cell-major versus term-major exact derivatives."""
import json
import os
import argparse
from pathlib import Path
import numpy as np
from simesh.analysis import FieldDefinition, curl, Plane, sample_plane
from simesh.amrvac.analysis import prepare_resident
from simesh.amrvac.datio import read_blocks_sequential
from simesh.utils.lib.analysis.native import differentiate, differentiate_cellwise
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source
from analysis_core.benchmark_prepared import measure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--composition-only',action='store_true')
options = parser.parse_args()

fd = os.open('data/weno509_sub_0000.dat',os.O_RDONLY)
try:
    index = read_amrvac_v5_index(fd)
    binding = bind_amrvac_v5_forest(index)
finally:
    os.close(fd)
backing = read_blocks_sequential('data/weno509_sub_0000.dat',[4,5,6])
source = make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
    index.domain_lower,index.domain_upper,index.block_cell_counts,array_block_reader(backing),
    tuple(FieldDefinition(n,'code') for n in ('b1','b2','b3')))
mesh = source.mesh
primary = prepare_resident(mesh,binding.root_shape,index.forest_flags,backing,source.fields)
del source,backing
selected,_,_ = mesh.select_box(mesh.lower+.42*(mesh.upper-mesh.lower),
                              mesh.lower+.58*(mesh.upper-mesh.lower))
terms = np.array([(0,2,1),(0,1,2),(1,0,2),(1,2,0),(2,1,0),(2,0,1)],dtype=np.int64)
weights = np.tile([1.,-1.],3)
records = []
cases = [] if options.composition_only else [('mixed',selected),('full',np.arange(mesh.leaf_count,dtype=np.int64))]
for name,ids in cases:
    output = np.empty((len(ids),10,10,10,3))
    args = (primary.values,primary.slot_of_leaf,ids,mesh.spacing,terms,weights,output)
    if name=='mixed':
        differentiate_cellwise(*args)
        reference = output.copy()
        differentiate(*args)
        np.testing.assert_array_equal(output.view(np.uint64),reference.view(np.uint64))
        del reference
    records.append({'name':name,'leaves':len(ids),'output_bytes':output.nbytes,
        'cell_major':measure(lambda:differentiate_cellwise(*args),3),
        'term_major':measure(lambda:differentiate(*args),3)})
    del output,args
if options.composition_only:
    from simesh.utils.lib.analysis import native
    spans = mesh.upper-mesh.lower
    planes = (Plane(mesh.lower+spans*[0,0,.5],spans*[1,0,0],spans*[0,1,0],(128,128)),
              Plane(mesh.lower+spans*[0,0,.2],spans*[1,0,.3],spans*[0,1,.3],(96,80)))
    def consume():
        derived = curl(primary)
        for plane in planes:
            sample_plane(derived,plane)
    try:
        for name,function in [('cell_major',differentiate_cellwise),('term_major',differentiate)]:
            native.differentiate = function
            records.append({'variant':name,'full_curl_and_two_slices':measure(consume,3)})
    finally:
        native.differentiate = differentiate
    target = 'benchmark-results/analysis-core/derivative-composition.json'
else:
    target = 'benchmark-results/analysis-core/derivative-order.json'
Path(target).write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
