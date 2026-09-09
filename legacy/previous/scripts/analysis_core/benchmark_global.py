"""P3-D full WENO derivative and retained-slice evidence within 2 GiB arrays."""

import gc
import json
import os
from pathlib import Path
import resource
import time
import numpy as np

from simesh.analysis import FieldDefinition, global_curl, curl, Plane, sample_plane
from simesh.amrvac.analysis import prepare_resident
from simesh.amrvac.datio import read_blocks_sequential
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source
from analysis_core.benchmark_prepared import measure, compare


def main():
    start = time.perf_counter()
    path = "data/weno509_sub_0000.dat"
    fd = os.open(path,os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(fd)
        binding = bind_amrvac_v5_forest(index)
    finally:
        os.close(fd)
    index_seconds = time.perf_counter()-start
    stamp = time.perf_counter()
    backing = read_blocks_sequential(path,[4,5,6])
    read_seconds = time.perf_counter()-stamp
    definitions = tuple(FieldDefinition(n,"code") for n in ("b1","b2","b3"))
    source = make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
        index.domain_lower,index.domain_upper,index.block_cell_counts,array_block_reader(backing),
        definitions,support_capacity=256)
    mesh = source.mesh
    extent = mesh.upper-mesh.lower
    axis = Plane(mesh.lower+extent*[0,0,.5],extent*[1,0,0],extent*[0,1,0],(128,128))
    oblique = Plane(mesh.lower+extent*[0,0,.2],extent*[1,0,.3],extent*[0,1,.3],(96,80))
    outpath = Path("benchmark-results/analysis-core/global-curl.npy")
    if outpath.exists():
        raise FileExistsError(f"preserve or explicitly remove previous task output: {outpath}")
    output = np.lib.format.open_memmap(outpath,mode="w+",dtype=float,
        shape=(mesh.leaf_count,10,10,10,3))
    result = {"source":path,"source_bytes":Path(path).stat().st_size,
        "index_seconds":index_seconds,"read_seconds":read_seconds,
        "output_path":str(outpath),"output_bytes":output.nbytes,
        "profile":"whole native domain curl(B), two primary / one derived halo, continuous",
        "cache":"application-warm repeats, OS cache uncontrolled", "bounded":[]}
    slice_references = []
    for repeat in range(2):
        begin = time.perf_counter()
        product = global_curl(source,batch_size=256,output=output)
        computed = time.perf_counter()
        slices = [sample_plane(product,plane) for plane in (axis,oblique)]
        end = time.perf_counter()
        if not all(s.valid.all() for s in slices):
            raise AssertionError("in-domain plane has missing coverage")
        if repeat==0:
            slice_references = [s.values.copy() for s in slices]
        else:
            for s,ref in zip(slices,slice_references):
                np.testing.assert_array_equal(s.values,ref)
        result["bounded"].append({**product.preparation_stats,
            "wall_seconds":computed-begin,"first_two_slices_seconds":end-computed,
            "global_plus_slices_seconds":end-begin})
        del product,slices
        print("bounded global pass",repeat,"completed",flush=True)
    output.flush()
    del output,source,backing
    gc.collect()
    # Complete reference lives on disk now, not a second resident derived array.
    backing = read_blocks_sequential(path,[4,5,6])
    resident = prepare_resident(mesh,binding.root_shape,index.forest_flags,backing,definitions)
    result["resident_preparation"] = resident.preparation_stats
    del backing
    gc.collect()
    begin = time.perf_counter()
    derived = curl(resident)
    computed = time.perf_counter()
    slices = [sample_plane(derived,plane) for plane in (axis,oblique)]
    end = time.perf_counter()
    result["resident"] = {"derivative_first_seconds":computed-begin,
        "first_two_slices_seconds":end-computed,
        "prepared_to_two_slices_seconds":end-begin,
        "input_retained_bound":resident.nbytes+mesh.nbytes,
        "derived_bytes":derived.nbytes}
    for sliced,ref in zip(slices,slice_references):
        check = compare(sliced.values,ref)
        if not check["within_tolerance"]:
            raise AssertionError(check)
    result["slice_checks"] = [compare(s.values,ref) for s,ref in zip(slices,slice_references)]
    result["warm_slices"] = [measure(lambda:sample_plane(derived,p),3) for p in (axis,oblique)]
    reference = np.load(outpath,mmap_mode="r")
    worst,compared = 0.,0
    for first in range(0,mesh.leaf_count,128):
        check = compare(derived.values[first:first+128],reference[first:first+128])
        if not check["within_tolerance"]:
            raise AssertionError((first,check))
        worst = max(worst,check["max_abs"])
        compared += check["values"]
    result["all_derived_check"] = {"values":compared,"max_abs":worst,"within_tolerance":True}
    del reference
    gc.collect()
    # Actual canonical operator, one component buffer at a time. Its larger
    # padded output and repeated zeroing are included; not Dataset materialization.
    operator_bytes = mesh.leaf_count*12**3*8
    metadata = sum(a.nbytes for a in (*index,*binding.forest) if isinstance(a,np.ndarray))
    admission = resident.nbytes+derived.nbytes+mesh.nbytes+operator_bytes+metadata+8*1024**2
    if admission > 2*1024**3:
        raise MemoryError(f"canonical operator comparison requires {admission} bytes")
    temporary = np.empty((mesh.leaf_count,12,12,12,1),dtype=float)
    result["canonical_operators"] = []
    for component,(fields,axes) in enumerate((((2,1),(1,2)),((0,2),(2,0)),((1,0),(0,1)))):
        fields,axes = np.asarray(fields,dtype=np.uint32),np.asarray(axes,dtype=np.uint32)
        def run():
            resident.owner.first_derivative_fields(temporary,np.zeros(2,np.uint32),fields,axes,
                                                    np.array([1.,-1.]))
        timing = measure(run,3)
        worst = 0.
        for first in range(0,mesh.leaf_count,128):
            check = compare(derived.values[first:first+128,...,component],
                            temporary[first:first+128,1:-1,1:-1,1:-1,0])
            if not check["within_tolerance"]:
                raise AssertionError((component,first,check))
            worst = max(worst,check["max_abs"])
        result["canonical_operators"].append({"component":component,"timing":timing,"max_abs":worst})
    del temporary
    # Do not allocate a second complete derived result while the first is live.
    del derived,slices
    gc.collect()
    result["resident_derivative_repeat"] = measure(lambda:curl(resident),3)
    result["controlled_operator_comparison_upper"] = admission
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path("benchmark-results/analysis-core/global.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"bounded":result["bounded"],"resident":result["resident"],
        "all_derived_check":result["all_derived_check"],"peak_rss_bytes":result["peak_rss_bytes"]},indent=2))


if __name__ == "__main__":
    main()
