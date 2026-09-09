"""Dense native I/O order, preserving each checked callback's complete preflight."""
import json
from pathlib import Path
import numpy as np
from simesh.analysis import open_source
from analysis_core.benchmark_prepared import measure
records=[]
for names in (['b1','b2','b3'],['rho']):
    with open_source('data/weno509_sub_0000.dat',field_names=names) as source:
        ids=np.arange(source.mesh.leaf_count,dtype=np.int64)
        fields=np.arange(len(names),dtype=np.int64)
        output=np.empty((len(ids),len(names),*source.mesh.block_shape))
        for batch in (128,1):
            timing=measure(lambda:source.read_interiors(ids,fields,output,batch_size=batch),3)
            records.append({'fields':names,'batch':batch,'timing':timing})
        del output
Path('benchmark-results/analysis-core/read-order.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
