"""Explicit feature-completion WENO components; never a default test hook."""

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import lfe_001 as lfe
import chs_001 as chs
import sle_001 as sle
from dat_003 import repack_weno_regular_fields
from m1_optimization import original_module, BASELINE
from m1_cache_scaling import stages, array_inventory_bytes
from m1_weno_assessment import REGIONS
import simesh_rewrite.refined_halo as halo
import simesh_rewrite.completed_halo_sampling as cache
from simesh_rewrite.completed_primary import (
    make_completed_primary_consumer, execute_selected_refined_halos_with_consumer,
)
from simesh_rewrite.repeated_sampling import execute_refined_zero_order_points_from_blocks
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS


def timed(operation, repeats):
    wall, cpu = [], []
    result = None
    for repetition in range(repeats + 1):
        start, process = time.perf_counter(), time.process_time()
        result = operation()
        elapsed_cpu, elapsed = time.process_time()-process, time.perf_counter()-start
        if repetition:
            wall.append(elapsed)
            cpu.append(elapsed_cpu)
    return result, {"wall": lfe.scalar_summary(wall), "cpu": lfe.scalar_summary(cpu),
                    "raw_wall": wall, "raw_cpu": cpu}


def differences(actual, expected, tolerance=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    finite = np.isfinite(actual) & np.isfinite(expected)
    error = np.abs(actual[finite]-expected[finite])
    classification = np.array_equal(np.isnan(actual),np.isnan(expected)) and np.array_equal(
        np.isposinf(actual),np.isposinf(expected)) and np.array_equal(np.isneginf(actual),np.isneginf(expected))
    return {"values": actual.size, "finite_pairs": int(finite.sum()),
        "bit_mismatches": int(np.count_nonzero(actual.view(np.uint64)!=expected.view(np.uint64))),
        "max_abs": float(error.max()) if error.size else None,
        "rms": float(np.sqrt(np.mean(error**2))) if error.size else None,
        "classification_equal": bool(classification),
        "within_tolerance": bool(classification and np.allclose(actual[finite],expected[finite],rtol=tolerance,atol=tolerance)),
        "rtol_atol": tolerance}


@contextmanager
def bridge_case(source):
    source_fd = os.open(source,os.O_RDONLY)
    try:
        original = lfe.read_amrvac_v5_index(source_fd)
        if (original.leaf_count != 22614 or not original.staggered
                or original.geometry != "Cartesian_3D" or np.any(original.periodic)
                or tuple(original.block_cell_counts) != (8,8,8)
                or tuple(original.domain_cell_counts) != (16,16,8)
                or tuple(original.field_names[4:7]) != ("b1","b2","b3")):
            raise ValueError("source does not match the frozen WENO reference profile")
        projected_bytes = original.leaf_count*(24+3*8**3*8)+int(original.offset_blocks)
        if projected_bytes > 512*1024**2:
            raise MemoryError("bridge exceeds the frozen disk budget")
    finally:
        os.close(source_fd)
    with tempfile.TemporaryDirectory(prefix="simesh-weno-reference-") as directory:
        path = Path(directory)/"regular.dat"
        setup = repack_weno_regular_fields(source,path,(4,5,6))
        fd = os.open(path,os.O_RDONLY)
        try:
            start = time.perf_counter()
            index = lfe.read_amrvac_v5_index(fd)
            binding = lfe.bind_amrvac_v5_forest(index)
            forest = binding.forest
            lfe.validate_refined_all_touch_2to1(binding.root_shape,binding.coord_to_rank,
                forest.root_node_ids,forest.node_levels,forest.node_coords,forest.child_node_ids,
                forest.node_leaf_ids,forest.leaf_node_ids)
            setup["rewrite_index_bind_balance_seconds"] = time.perf_counter()-start
            reader = lfe.make_amrvac_v5_block_reader(fd,index,binding)
            case = chs.case_from_v5(index,binding,lfe.i3(0,1,2),"WENO")
            setup["metadata_array_bytes"] = array_inventory_bytes(original,index,binding)
            yield case, reader, setup
        finally:
            os.close(fd)


def local_fixture(case):
    f = case.forest
    return lfe.Fixture(case.root_shape,case.coord_to_rank,f.root_node_ids,f.node_levels,
        f.node_coords,f.child_node_ids,f.node_leaf_ids,f.leaf_node_ids,f.max_level,
        case.domain_lower,case.domain_upper,case.domain_counts,case.block_counts,
        np.empty((0,3,8,8,8)),np.empty(0,dtype=np.int32))


def selection(case, label="mixed"):
    _, lo, hi = next(row for row in REGIONS if row[0]==label)
    lower = case.domain_lower + np.array(lo)*(case.domain_upper-case.domain_lower)
    upper = case.domain_lower + np.array(hi)*(case.domain_upper-case.domain_lower)
    return lfe.refined_region_windows(*lfe.selection_args(local_fixture(case),lower,upper)), lower, upper


def queries(case):
    line = np.tile([.5,.5,.5],(192,1))
    line[:,0] = np.linspace(.42,.58,192)
    points = np.ascontiguousarray(case.domain_lower+line*(case.domain_upper-case.domain_lower))
    owners = chs.locate(case,points)
    coherent = chs.Query("coherent",points,chs.last_owner_hints(owners),owners,chs.batch_slices(192,4))
    selected, _, _ = selection(case)
    ids = selected.leaf_ids[np.linspace(0,len(selected.leaf_ids)-1,36,dtype=np.int64)]
    owners = np.tile(ids,4)
    points = np.ascontiguousarray(case.leaf_bounds[owners,0]+.1*case.leaf_spacing[owners])
    assert np.array_equal(chs.locate(case,points),owners)
    divergent = chs.Query("divergent",points,np.full(144,-1,dtype=np.int64),owners,(slice(0,144),))
    return coherent, divergent


def variants():
    old_halo = original_module("refined_halo")._preflight_chunk_actions
    old_cache = original_module("completed_halo_sampling")._cache_access_plan
    return (("M1",old_halo,old_cache),("HPR",halo._preflight_chunk_actions,old_cache),
            ("retained",halo._preflight_chunk_actions,cache._cache_access_plan))


def sampling(case,reader,repeats):
    choices = variants()
    records = []
    try:
        for query in queries(case):
            expected = np.empty((len(query.points),3))
            rps_stats, rps_time = timed(lambda: chs.execute_refined_trilinear_points_from_blocks(
                *chs.rps_arguments(case,reader,query,expected)),repeats)
            w = len(np.unique(query.owners))
            sessions = []
            for capacity in sorted({0,max(1,w//2),w,2*w}):
                entry = 3*10**3*8+16
                active = [chs.make_completed_halo_sampling_session(*chs.session_arguments(
                    case,reader,57,capacity*entry)) for _ in choices]
                for warm in (False,True):
                    raw, cpu = [[],[],[]],[[],[],[]]
                    stats = [None,None,None]
                    for repetition in range(repeats+1):
                        for vi in tuple((repetition+i)%3 for i in range(3)):
                            _,halo._preflight_chunk_actions,cache._cache_access_plan = choices[vi]
                            chs.clear_completed_halo_sampling_session(active[vi])
                            if warm:
                                chs.run_chs_sequence(active[vi],query)
                            start, process = time.perf_counter(),time.process_time()
                            values,owners,stats[vi] = chs.run_chs_sequence(active[vi],query)
                            elapsed_cpu, elapsed = time.process_time()-process,time.perf_counter()-start
                            if repetition:
                                raw[vi].append(elapsed); cpu[vi].append(elapsed_cpu)
                            assert np.array_equal(values.view(np.uint64),expected.view(np.uint64))
                            assert np.array_equal(owners,query.owners)
                        assert stats[0]==stats[1]==stats[2]
                    _,halo._preflight_chunk_actions,cache._cache_access_plan = choices[2]
                    chs.clear_completed_halo_sampling_session(active[2])
                    if warm:
                        chs.run_chs_sequence(active[2],query)
                    attribution, preads = {}, chs.PreadCounter()
                    with stages(attribution),chs.count_native_preads(preads,True):
                        chs.run_chs_sequence(active[2],query)
                    sessions.append({"capacity":capacity,"warm":warm,
                        "variants":[{"name":v[0],"wall":lfe.scalar_summary(raw[i]),
                            "cpu":lfe.scalar_summary(cpu[i]),"raw_wall":raw[i],"raw_cpu":cpu[i]}
                            for i,v in enumerate(choices)],"stats":stats[2],
                        "stages_nested":attribution,"pread":preads.as_dict(),
                        "memory":chs.memory_breakdown(active[2]),
                        "live_array_bytes":array_inventory_bytes(case,reader,active,query,expected),
                        "bits_equal_to_rps":True})
                del active
            records.append({"query":query.name,"working_set":w,"point_count":len(query.points),
                "rps":rps_time,"rps_stats":lfe.stats_record(rps_stats),"sessions":sessions})
    finally:
        _,halo._preflight_chunk_actions,cache._cache_access_plan = choices[2]
    return records


def trajectory(case,reader,repeats):
    selected,_,_ = selection(case)
    ids = selected.leaf_ids[np.linspace(0,len(selected.leaf_ids)-1,8,dtype=np.int64)]
    seeds = np.ascontiguousarray(case.leaf_bounds[ids].mean(axis=1))
    spec = sle.Trajectory("WENO-eight-seed",seeds,np.ones(8,dtype=np.int8),
        .25*float(case.leaf_spacing[ids].min()),8,False,8)
    choices = variants()
    records, expected = [], None
    try:
        for capacity in (4,8,16):
            active = [sle.make_completed_halo_sampling_session(*sle.session_arguments(case,reader,capacity))
                      for _ in choices]
            outputs = [sle.allocate_outputs(spec) for _ in choices]
            for warm in (False,True):
                raw = [[],[],[]]
                cpu = [[],[],[]]
                for repeat in range(repeats+1):
                    stats = [None,None,None]
                    for vi in tuple((repeat+i)%3 for i in range(3)):
                        _,halo._preflight_chunk_actions,cache._cache_access_plan = choices[vi]
                        sle.clear_completed_halo_sampling_session(active[vi])
                        if warm:
                            sle.run_trajectory(active[vi],spec,outputs[vi])
                        sle.reset_outputs(outputs[vi])
                        start,process = time.perf_counter(),time.process_time()
                        stats[vi] = sle.run_trajectory(active[vi],spec,outputs[vi])
                        elapsed_cpu,elapsed = time.process_time()-process,time.perf_counter()-start
                        if repeat:
                            raw[vi].append(elapsed)
                            cpu[vi].append(elapsed_cpu)
                        if expected is None:expected = sle.copy_outputs(outputs[vi])
                        assert sle.outputs_equal(outputs[vi],expected)
                    assert stats[0]==stats[1]==stats[2]
                records.append({"capacity":capacity,"warm":warm,
                    "variants":[{"name":v[0],"wall":lfe.scalar_summary(raw[i]),"raw_wall":raw[i],
                                 "cpu":lfe.scalar_summary(cpu[i]),"raw_cpu":cpu[i]}
                                for i,v in enumerate(choices)],"stats":sle.stats_record(stats[2]),
                    "digest":sle.result_digest(outputs[2]),"bits_equal":True,
                    "live_array_bytes":array_inventory_bytes(case,reader,active,outputs,expected)})
    finally:
        _,halo._preflight_chunk_actions,cache._cache_access_plan = choices[2]
    return {"step_size":spec.step_size,"steps":8,"seeds":seeds.tolist(),"records":records,
            "canonical_comparator":"unsupported: original simesh has no matching native tracer contract"}


def cache_grid(case,reader,repeats):
    _,lower,upper = selection(case)
    shape = (16,8,4)
    axes = [lower[a]+(np.arange(shape[a])+.5)*(upper[a]-lower[a])/shape[a] for a in range(3)]
    points = np.ascontiguousarray(np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3))
    owners = chs.locate(case,points)
    query = chs.Query("common-grid",points,np.full(len(points),-1,dtype=np.int64),owners,(slice(0,len(points)),))
    expected = np.empty((len(points),3))
    rps_stats,rps_time = timed(lambda:chs.execute_refined_trilinear_points_from_blocks(
        *chs.rps_arguments(case,reader,query,expected)),repeats)
    capacity = len(np.unique(owners))
    start = time.perf_counter()
    session = chs.make_completed_halo_sampling_session(*chs.session_arguments(
        case,reader,57,capacity*(3*1000*8+16)))
    create_seconds = time.perf_counter()-start
    start = time.perf_counter()
    values,ids,cold = chs.run_chs_sequence(session,query)
    cold_seconds = time.perf_counter()-start
    (values,ids,warm),warm_time = timed(lambda:chs.run_chs_sequence(session,query),repeats)
    assert np.array_equal(values.view(np.uint64),expected.view(np.uint64))
    assert np.array_equal(ids,owners) and warm["cache_miss_count"]==0
    attribution,preads = {},chs.PreadCounter()
    with stages(attribution),chs.count_native_preads(preads,True):
        chs.run_chs_sequence(session,query)
    return {"records":[{"shape":shape,"owner_working_set":capacity,"rps":rps_time,
        "rps_stats":lfe.stats_record(rps_stats),"session_create_seconds":create_seconds,"cold_seconds":cold_seconds,
        "warm":warm_time,"cold_stats":cold,"warm_stats":warm,"stages_nested":attribution,
        "warm_pread":preads.as_dict(),"bits_equal_to_rps":True,"memory":chs.memory_breakdown(session),
        "live_array_bytes":array_inventory_bytes(case,reader,session,query,expected,values,ids)}]}


@contextmanager
def named_stages(module,names,totals):
    saved = {name:getattr(module,name) for name in names}
    try:
        for name,original in saved.items():
            def measure(*args,_name=name,_fn=original,**kwargs):
                start = time.perf_counter()
                try:return _fn(*args,**kwargs)
                finally:totals[_name] = totals.get(_name,0.)+time.perf_counter()-start
            setattr(module,name,measure)
        yield
    finally:
        for name,original in saved.items():setattr(module,name,original)


def attribution(case,reader):
    import simesh_rewrite.field_lines as lines
    selected,_,_ = selection(case)
    ids = selected.leaf_ids[np.linspace(0,len(selected.leaf_ids)-1,8,dtype=np.int64)]
    spec = sle.Trajectory("WENO-eight-seed",np.ascontiguousarray(case.leaf_bounds[ids].mean(axis=1)),
        np.ones(8,dtype=np.int8),.25*float(case.leaf_spacing[ids].min()),8,False,8)
    session = sle.make_completed_halo_sampling_session(*sle.session_arguments(case,reader,8))
    output = sle.allocate_outputs(spec)
    records,expected = [],None
    for warm in (False,True):
        sle.clear_completed_halo_sampling_session(session)
        if warm:sle.run_trajectory(session,spec,output)
        sle.reset_outputs(output)
        totals = {}
        start = time.perf_counter()
        with stages(totals),named_stages(cache,("_validate_dynamic_arrays","_validate_and_locate",
                "_validate_point_plan"),totals),named_stages(lines,(
                "_sample_refined_trilinear_vectors_cached","field_line_rhs_unchecked",
                "field_line_rk4_stage_unchecked","field_line_rk4_finish_unchecked",
                "_classify_field_line_candidate_unchecked","_classify_field_line_stage_state_unchecked",
                "_field_line_termination_from_rhs_status_unchecked"),totals):
            stats = sle.run_trajectory(session,spec,output)
        wall = time.perf_counter()-start
        if expected is None:expected = sle.copy_outputs(output)
        assert sle.outputs_equal(output,expected)
        records.append({"warm":warm,"instrumented_wall":wall,"stages_nested_seconds":totals,
            "stats":sle.stats_record(stats),"live_array_bytes":array_inventory_bytes(case,reader,session,output,expected)})
    return {"scope":"Single instrumented passes only; nested stages are not additive or headline timings.","records":records}


def canonical(case,reader,source,repeats):
    from simesh.amrvac.datio import get_metadata,read_blocks_sequential
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh,openmp_build_info
    (header,flags,tree), metadata = timed(lambda:get_metadata(str(source)),repeats)
    root = header["domain_nx"]//header["block_nx"]
    forest, forest_time = timed(lambda:AMRForest(3,*map(np.uint32,root),flags.astype(np.int32)),repeats)
    controlled_bound = 22614*8*(3*12**3+3*8**3+12**3+3*8**3)+64*1024**2
    if controlled_bound > 2*1024**3:
        raise MemoryError("canonical stage exceeds its frozen 2 GiB controlled budget")
    start = time.perf_counter()
    mesh = AMRMesh(3,header["block_nx"].astype(np.uint32),header["domain_nx"].astype(np.uint32),
        header["xmin"],header["xmax"],2,3,forest)
    allocate_seconds = time.perf_counter()-start
    backing, read_time = timed(lambda:read_blocks_sequential(str(source),[4,5,6]),repeats)
    _, copy_time = timed(lambda:mesh.load_interior_data(backing),repeats)
    _, halo_time = timed(mesh.apply_ghost_cells,repeats)
    np.testing.assert_array_equal(mesh.interior_view().view(np.uint64),backing.view(np.uint64))
    del backing
    gc.collect()
    selections = {name:selection(case,name)[0] for name,_,_ in REGIONS}
    curl_results = {name:np.empty((len(s.leaf_ids),3,8,8,8)) for name,s in selections.items()}
    padded_result = np.empty((case.leaf_count,12,12,12,1))
    derivative_times = []
    for component,(source_fields,axes) in enumerate((((2,1),(1,2)),((0,2),(2,0)),((1,0),(0,1)))):
        fields = np.asarray(source_fields,dtype=np.uint32)
        axis = np.asarray(axes,dtype=np.uint32)
        _, timing = timed(lambda:mesh.first_derivative_fields(padded_result,
            np.zeros(2,dtype=np.uint32),fields,axis,np.array([1.,-1.])),repeats)
        derivative_times.append(timing)
        for name,s in selections.items():
            curl_results[name][:,component] = padded_result[s.leaf_ids,2:10,2:10,2:10,0]
    derivative_output_bytes = padded_result.nbytes
    del padded_result
    comparisons = []
    for name,s in selections.items():
        output = np.full_like(curl_results[name],lfe.SENTINEL)
        acc = np.array([1.25])
        _, m1_time = timed(lambda:lfe.execute_selected_refined_curl_from_blocks(*lfe.execution_args(
            local_fixture(case),reader,lfe.i3(0,1,2),s,256,output,acc)),repeats)
        parts, canon_parts, safe_parts, safe_canon = [],[],[],[]
        for row in range(len(s.leaf_ids)):
            box = tuple(slice(int(a),int(b)) for a,b in zip(s.cell_lower[row],s.cell_upper[row]))
            parts.append(output[(row,slice(None),*box)].ravel())
            canon_parts.append(curl_results[name][(row,slice(None),*box)].ravel())
            lo,hi = np.maximum(s.cell_lower[row],1),np.minimum(s.cell_upper[row],7)
            if np.all(lo<hi):
                box = tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
                safe_parts.append(output[(row,slice(None),*box)].ravel())
                safe_canon.append(curl_results[name][(row,slice(None),*box)].ravel())
        common = differences(np.concatenate(parts),np.concatenate(canon_parts),5e-13)
        safe = differences(np.concatenate(safe_parts),np.concatenate(safe_canon),5e-13) if safe_parts else None
        if safe:assert safe["within_tolerance"]
        comparisons.append({"region":name,"m1_selected_time":m1_time,"all_selected_curl":common,
                            "safe_interior_curl":safe})
    selected = selections["mixed"].leaf_ids
    completed = np.empty((len(selected),3,10,10,10))
    def consume(state,first,ids,payload,*boxes):
        state[first:first+len(ids)] = payload
    consumer = make_completed_primary_consumer(completed,consume,output_arrays=(completed,))
    f = case.forest
    _, rhe_time = timed(lambda:execute_selected_refined_halos_with_consumer(reader,consumer,selected,
        case.field_ids,case.root_shape,case.coord_to_rank,f.root_node_ids,f.node_levels,f.node_coords,
        f.child_node_ids,f.node_leaf_ids,f.leaf_node_ids,lfe.i3(1,1,1),lfe.i3(1,1,1),
        case.boundary_modes,case.normal_field_slots,256),repeats)
    canonical_halos = np.ascontiguousarray(mesh.padded_view()[selected,1:11,1:11,1:11,:].transpose(0,4,1,2,3))
    halo_comparison = differences(completed,canonical_halos,1e-12)
    _, lower,upper = selection(case)
    shape = np.array([16,8,4],dtype=np.uint32)
    axes = [lower[a]+(np.arange(shape[a])+.5)*(upper[a]-lower[a])/shape[a] for a in range(3)]
    points = np.ascontiguousarray(np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3))
    owners = chs.locate(case,points)
    query = chs.Query("common-grid",points,np.full(len(points),-1,dtype=np.int64),owners,(slice(0,len(points)),))
    sampling_reports = []
    for linear in (False,True):
        grid = np.full((3,*map(int,shape)),np.nan)
        if linear:
            operation = lambda:mesh.uniform_grid_linear(grid,shape,lower,upper,np.arange(3,dtype=np.uint32))
        else:
            operation = lambda:mesh.uniform_grid_zero_order(mesh.interior_view(),grid,shape,lower,upper)
        _, ct = timed(operation,repeats)
        values = np.empty((len(points),3))
        fn = chs.execute_refined_trilinear_points_from_blocks if linear else execute_refined_zero_order_points_from_blocks
        arguments = chs.rps_arguments(case,reader,query,values)
        if not linear:
            arguments = (*arguments[:16],57,values)
        _, rt = timed(lambda:fn(*arguments),repeats)
        expected = np.ascontiguousarray(grid.transpose(1,2,3,0).reshape(-1,3))
        spatial = (points-case.leaf_bounds[owners,0])/case.leaf_spacing[owners]-.5
        safe = np.all((spatial>=0)&(spatial<7),axis=1)
        safe_result = differences(values[safe],expected[safe])
        if safe.any():assert safe_result["within_tolerance"]
        sampling_reports.append({"linear":linear,"canonical_query":ct,"m1_query":rt,
            "comparison":differences(values,expected),"safe_interior":safe_result,
            "safe_point_count":int(safe.sum())})
    return {"comparison_scope":"canonical full-domain core composition, not Dataset registry/materialization lifecycle",
        "openmp":openmp_build_info(),"metadata":metadata,"forest_connectivity":forest_time,
        "mesh_allocate_seconds":allocate_seconds,"eager_read":read_time,"interior_copy":copy_time,
        "full_halo":halo_time,"full_derivative_components":derivative_times,
        "derivative_timing_scope":"one component at a time; includes full padded output initialization",
        "canonical_padded_bytes":mesh.padded_view().nbytes,"canonical_coarse_bytes":mesh.datac.nbytes,
        "canonical_derivative_output_bytes":derivative_output_bytes,"admitted_controlled_upper_bytes":controlled_bound,
        "curl_comparisons":comparisons,"mixed_selected_rhe":rhe_time,"mixed_common_halos":halo_comparison,
        "sampling":sampling_reports,"peak_rss_bytes":lfe.peak_rss_bytes()}


def halo_full(case,reader,source,repeats):
    from simesh.amrvac.datio import get_metadata,read_blocks_sequential
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh
    header,flags,_ = get_metadata(str(source))
    root = header["domain_nx"]//header["block_nx"]
    forest = AMRForest(3,*map(np.uint32,root),flags.astype(np.int32))
    bound = case.leaf_count*8*(3*12**3+3*8**3+3*8**3)+128*1024**2
    if bound>2*1024**3:raise MemoryError("full-halo control exceeds 2 GiB")
    mesh = AMRMesh(3,header["block_nx"].astype(np.uint32),header["domain_nx"].astype(np.uint32),
        header["xmin"],header["xmax"],2,3,forest)
    backing = read_blocks_sequential(str(source),[4,5,6])
    mesh.load_interior_data(backing)
    del backing
    _,canonical_time = timed(mesh.apply_ghost_cells,repeats)
    target = mesh.padded_view()
    f = case.forest
    ids = np.arange(case.leaf_count,dtype=np.int64)
    state = {"check":True,"max_abs":0.,"mismatches":0,"values":0,
             "copy_seconds":0.,"first_result":None,"start":0.}
    def consume(state,first,leaves,payload,*boxes):
        values = payload.transpose(0,2,3,4,1)
        if state["check"]:
            expected = target[first:first+len(leaves)]
            assert np.array_equal(leaves,ids[first:first+len(leaves)])
            metrics = differences(values,expected,1e-12)
            if not metrics["within_tolerance"]:
                raise AssertionError(f"full halo differs from canonical: {metrics}")
            state["max_abs"] = max(state["max_abs"],metrics["max_abs"] or 0.)
            state["mismatches"] += metrics["bit_mismatches"]
            state["values"] += metrics["values"]
        start = time.perf_counter()
        target[first:first+len(leaves)] = values
        state["copy_seconds"] += time.perf_counter()-start
        if state["first_result"] is None:state["first_result"] = time.perf_counter()-state["start"]
    consumer = make_completed_primary_consumer(state,consume,output_arrays=(target,))
    arguments = (reader,consumer,ids,case.field_ids,case.root_shape,case.coord_to_rank,f.root_node_ids,
        f.node_levels,f.node_coords,f.child_node_ids,f.node_leaf_ids,f.leaf_node_ids,lfe.i3(2,2,2),
        lfe.i3(2,2,2),case.boundary_modes,case.normal_field_slots,256)
    state["start"] = time.perf_counter()
    expected_stats = execute_selected_refined_halos_with_consumer(*arguments)
    state["check"] = False
    wall,cpu,copies,firsts = [],[],[],[]
    for _ in range(repeats):
        mesh.apply_ghost_cells()
        state["copy_seconds"],state["first_result"] = 0.,None
        state["start"],process = time.perf_counter(),time.process_time()
        stats = execute_selected_refined_halos_with_consumer(*arguments)
        cpu.append(time.process_time()-process)
        wall.append(time.perf_counter()-state["start"])
        copies.append(state["copy_seconds"]);firsts.append(state["first_result"])
        assert stats==expected_stats
    recorder,preads = lfe.StageRecorder(),lfe.PreadCounter()
    state["copy_seconds"],state["first_result"] = 0.,None
    state["start"] = recorder.started_at = time.perf_counter()
    with lfe.instrument_stages(recorder),lfe.count_native_preads(preads):
        execute_selected_refined_halos_with_consumer(*arguments)
    return {"canonical_full_halo":canonical_time,"m1_full_halo":{"wall":lfe.scalar_summary(wall),
        "cpu":lfe.scalar_summary(cpu),"raw_wall":wall,"raw_cpu":cpu,
        "output_copy_raw_seconds":copies,"first_result_raw_seconds":firsts},
        "stats":lfe.stats_record(expected_stats),"comparison":{"values":state["values"],
        "bit_mismatches":state["mismatches"],"max_abs":state["max_abs"],"rtol_atol":1e-12},
        "stages_nested_seconds":recorder.seconds,"instrumented_pread":preads.as_dict(),
        "canonical_padded_bytes":target.nbytes,"canonical_coarse_bytes":mesh.datac.nbytes,
        "admitted_controlled_upper_bytes":bound,"peak_rss_bytes":lfe.peak_rss_bytes(),
        "scope":"Same full-domain three-field width-two requested values. Canonical refresh is resident; RHC reads the bridge and copies all completed padded rows into the supplied output. Preparation and output-copy costs are explicit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section",choices=("canonical","sampling","trajectory","cache-grid","attribution","halo-full"),required=True)
    parser.add_argument("--source",type=Path,default=Path("data/weno509_sub_0000.dat"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=5)
    args = parser.parse_args()
    if args.repeats<1:raise ValueError("repeats must be positive")
    with bridge_case(args.source) as (case,reader,setup):
        if args.section=="canonical":result = canonical(case,reader,args.source,args.repeats)
        elif args.section=="sampling":result = sampling(case,reader,args.repeats)
        elif args.section=="trajectory":result = trajectory(case,reader,args.repeats)
        elif args.section=="cache-grid":result = cache_grid(case,reader,args.repeats)
        elif args.section=="attribution":result = attribution(case,reader)
        else:result = halo_full(case,reader,args.source,args.repeats)
        if args.section not in ("canonical","halo-full"):
            rows = ([r for q in result for r in q["sessions"]] if args.section=="sampling"
                    else result["records"])
            controlled_upper = max(r["live_array_bytes"] for r in rows)+setup["metadata_array_bytes"]+32*1024**2
            if controlled_upper > 512*1024**2:
                raise MemoryError("query stage exceeded the reference controlled budget")
            setup["query_controlled_upper_bytes"] = controlled_upper
            setup["query_bound_scope"] = "Live arrays plus metadata (conservative alias double count) and 32 MiB allowance for transient query/backend/Python storage. RSS is separate."
    record = {"section":args.section,"source":str(args.source),"source_bytes":args.source.stat().st_size,
        "bridge":setup,"environment":lfe.environment_record(),"original_m1_revision":BASELINE,
        "repeats":args.repeats,"result":result,"peak_rss_bytes":lfe.peak_rss_bytes()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(args.section,"complete",flush=True)


if __name__=="__main__":
    main()
