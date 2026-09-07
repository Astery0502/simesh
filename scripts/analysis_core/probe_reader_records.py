"""Checked general versus zero-ghost record/field-copy paths on whole WENO fields."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from simesh.analysis import open_source
import simesh_rewrite.amrvac_dat_reader as module
from analysis_core.benchmark_prepared import measure


@contextmanager
def strategy(fast):
    original=module.make_amrvac_v5_ordinary_block_reader
    def create(*args,**kwargs):
        reader=original(*args,**kwargs)
        return replace(reader,state=replace(reader.state,zero_ghost_fast=fast))
    module.make_amrvac_v5_ordinary_block_reader=create
    try:
        yield
    finally:
        module.make_amrvac_v5_ordinary_block_reader=original


records=[]
for names in (['b1','b2','b3'],['rho']):
    reference=None
    for fast in (False,True):
        with strategy(fast),open_source('data/weno509_sub_0000.dat',field_names=names) as source:
            ids=np.arange(source.mesh.leaf_count,dtype=np.int64)
            fields=np.arange(len(names),dtype=np.int64)
            output=np.empty((len(ids),len(names),*source.mesh.block_shape))
            source.read_interiors(ids,fields,output)
            if reference is None:
                reference=output.copy()
            else:
                np.testing.assert_array_equal(reference.view(np.uint64),output.view(np.uint64))
            timing=measure(lambda:source.read_interiors(ids,fields,output),3)
            records.append({'fields':names,'zero_ghost_fast':fast,'timing':timing,
                            'output_bytes':output.nbytes,'all_bits_equal':True})
            del output
    del reference
Path('benchmark-results/analysis-core/reader-records.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
