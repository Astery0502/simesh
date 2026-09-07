"""Explicit P1 WENO width-two composition, never an ordinary test hook."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import Cython
import numpy as np

from simesh.analysis import FieldDefinition, prepare, sample, curl
from simesh.amrvac.datio import get_metadata, read_blocks_sequential
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh, openmp_build_info
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source


def measure(fn, repeats=5):
    fn()
    wall, cpu = [], []
    for _ in range(repeats):
        start, process = time.perf_counter(), time.process_time()
        fn()
        wall.append(time.perf_counter()-start)
        cpu.append(time.process_time()-process)
    return {"wall": wall, "cpu": cpu, "median": statistics.median(wall),
            "stdev": statistics.pstdev(wall)}


def compare(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a[finite]-b[finite])
    okay = (np.array_equal(np.isnan(a), np.isnan(b)) and
            np.array_equal(np.isposinf(a), np.isposinf(b)) and
            np.array_equal(np.isneginf(a), np.isneginf(b)) and
            np.allclose(a[finite], b[finite], atol=1e-10, rtol=1e-10))
    return {"values": a.size, "max_abs": float(error.max(initial=0)),
            "within_tolerance": bool(okay)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/weno509_sub_0000.dat")
    parser.add_argument("--output", default="benchmark-results/analysis-core/prepared-weno.json")
    args = parser.parse_args()
    record = {"source": args.source, "source_bytes": Path(args.source).stat().st_size,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
              "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
              "cython": Cython.__version__, "openmp": openmp_build_info(),
              "profile": "P1 mixed/physical/small/thin width-two; 512 mixed grid centers",
              "cache": "application-warm; OS cache uncontrolled",
              "tolerance": {"rtol": 1e-10, "atol": 1e-10}}
    start = time.perf_counter()
    fd = os.open(args.source, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(fd)
        binding = bind_amrvac_v5_forest(index)
    finally:
        os.close(fd)
    if (index.leaf_count != 22614 or tuple(index.block_cell_counts) != (8,8,8) or
            index.field_names[4:7] != ("b1", "b2", "b3") or np.any(index.periodic)):
        raise ValueError("fixture differs from the frozen WENO profile")
    record["index_seconds"] = time.perf_counter()-start
    start = time.perf_counter()
    backing = read_blocks_sequential(args.source, [4,5,6])
    record["ordinary_resident_read_seconds"] = time.perf_counter()-start
    start = time.perf_counter()
    source = make_source(binding.root_shape, binding.coord_to_rank, binding.forest,
        index.domain_lower, index.domain_upper, index.block_cell_counts,
        array_block_reader(backing), tuple(FieldDefinition(n, "code") for n in ("b1","b2","b3")),
        support_capacity=256,
        extra_resident_bytes=sum(a.nbytes for a in index if isinstance(a,np.ndarray)))
    record["adapter_seconds"] = time.perf_counter()-start
    mesh = source.mesh
    header,flags,_ = get_metadata(args.source)
    root = header["domain_nx"]//header["block_nx"]
    # Bound includes raw B, canonical padded/coarse, selected outputs, temporary
    # canonical sampling and provider metadata/scratch; C allocations count too.
    bound = 22614*8*(3*12**3+2*3*8**3)+256*1024**2
    if bound > 2*1024**3:
        raise MemoryError("comparison exceeds the frozen controlled budget")
    start = time.perf_counter()
    forest = AMRForest(3, *map(np.uint32,root), flags.astype(np.int32))
    canonical = AMRMesh(3, header["block_nx"].astype(np.uint32), header["domain_nx"].astype(np.uint32),
                        header["xmin"], header["xmax"], 2, 3, forest)
    canonical.load_interior_data(backing)
    canonical.apply_ghost_cells()
    record["canonical_create_copy_prepare_seconds"] = time.perf_counter()-start
    record["canonical_full_resident_refresh"] = measure(canonical.apply_ghost_cells)
    record["regions"] = []
    regions = [("mixed",(.42,.42,.42),(.58,.58,.58)),
               ("physical",(0,.3,.3),(.02,.7,.7)),
               ("small",(.48,.48,.48),(.52,.52,.52)),
               ("thin",(.495,.3,.3),(.505,.7,.7))]
    product = None
    for name, lo, hi in regions:
        lo = mesh.lower+np.array(lo)*(mesh.upper-mesh.lower)
        hi = mesh.lower+np.array(hi)*(mesh.upper-mesh.lower)
        ids, _, _ = mesh.select_box(lo, hi)
        start = time.perf_counter()
        product = prepare(source, ids, [0,1,2])
        first = time.perf_counter()-start
        reference = canonical.padded_view()[ids]
        check = compare(product.values, reference)
        if not check["within_tolerance"]:
            raise AssertionError((name, check))
        del reference
        timings = measure(lambda: prepare(source, ids, [0,1,2]))
        record["regions"].append({"name": name, "primaries": len(ids), "first_seconds": first,
            "timing": timings, "comparison": check, "stats": product.preparation_stats,
            "product_bytes": product.nbytes,
            "classification": "restricted-common-domain; canonical prepares full domain"})
        if name == "mixed":
            shape = np.asarray([16,8,4], dtype=np.uint32)
            axes = [lo[a]+(np.arange(shape[a])+.5)*(hi[a]-lo[a])/shape[a] for a in range(3)]
            points = np.ascontiguousarray(np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3))
            output = np.empty((3,*shape), dtype=float)
            def canonical_sample():
                canonical.uniform_grid_linear(output,shape,lo,hi,np.arange(3,dtype=np.uint32))
            canonical_sample()
            actual,_,valid = sample(product,points)
            check = compare(actual, np.moveaxis(output,0,-1).reshape(-1,3))
            if not valid.all() or not check["within_tolerance"]:
                raise AssertionError(check)
            derived = curl(product)
            record["sampling"] = {"points":len(points), "comparison":check,
                "native":measure(lambda: sample(product,points)),
                "canonical_regular_grid":measure(canonical_sample),
                "classification":"same points; general owner lookup vs regular-grid interface"}
            record["derivative"] = {"timing":measure(lambda:curl(product)),
                "output_bytes":derived.nbytes, "valid_halo":derived.halo,
                "note":"local composition only; global D acceptance is separate"}
            del derived
    del product
    gc.collect()
    record["controlled_upper_bytes"] = bound
    record["native_mesh_bytes"] = mesh.nbytes
    record["source_resident_bytes"] = source.resident_bytes
    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.output).write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"regions":[(r["name"],r["timing"]["median"],r["comparison"]["max_abs"])
                                  for r in record["regions"]],
                      "sampling":record["sampling"], "peak_rss_bytes":record["peak_rss_bytes"]},indent=2))


if __name__ == "__main__":
    main()
