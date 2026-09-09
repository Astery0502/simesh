"""Measure selected application-interface costs on a small analytic fixture.

This is an opt-in implementation audit, not a throughput benchmark. Preparation
is excluded from the measured operations. tracemalloc reports traced allocations
after each operation starts, not process RSS or a native allocator upper bound.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time
import tracemalloc

import numpy as np
import simesh as sm
from simesh import applications as app
from simesh import result_shards, results_io
from simesh.tracing import _PATH_SEGMENT_STEPS
from simesh.physics import mhd


def timed(operation, repeats=9):
    operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
        del result
    return dict(median_seconds=statistics.median(samples),
                minimum_seconds=min(samples), maximum_seconds=max(samples))


def allocated(operation):
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, dict(retained_traced_bytes=current, peak_traced_bytes=peak)


def fixture():
    mesh = sm.mesh_from_forest((2, 2, 2), np.ones(8, dtype=bool),
                              lower=(0., 0., 0.), upper=(1., 1., 1.),
                              block_shape=(16, 16, 16))
    raw = np.zeros((8, 8, 16, 16, 16))
    model = sm.IdealMHD(gamma=5/3, energy_kind="total",
        composition=sm.CoronalComposition(),
        units=sm.MHDUnits(density_g_cm3=1.e-3, momentum_g_cm2_s=.1, energy_erg_cm3=10.,
                         field_gauss=1.e4, length_cm=100.))
    raw[:, 0] = 1.e-12
    raw[:, 3] = 1.e-12 * 2.e4
    raw[:, 7] = 1.e-4
    raw[:, 4] = .01/(model.gamma-1) + .5*1.e-12*(2.e4)**2 + 1.e-8/(2*model.units.magnetic_si.permeability_h_m)
    with sm.source_from_arrays(mesh, raw, ("rho", "m1", "m2", "m3", "e", "b1", "b2", "b3")) as source:
        fields = sm.prepare(source, scheme="coordinate-phase")
    return fields, model


def run(output):
    fields, model = fixture()
    magnetic = sm.select_fields(fields, ("b1", "b2", "b3"))
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  simesh=sm.__version__, platform=platform.platform(),
                  fixture=dict(leaves=8, block_shape=[16,16,16], valid_halo=2),
                  memory_method="tracemalloc incremental peak; excludes pre-existing inputs; not RSS")
    positions = np.random.default_rng(42).uniform(.1, .9, (50000, 3))
    points = sm.PointSet(positions)
    direct = lambda: sm.sample(magnetic, points.positions, components="b3")
    wrapped = lambda: app.sample(magnetic, points, components="b3")
    expected = direct()
    result = wrapped()
    for a, b in zip(expected, (result.values, result.owners, result.valid)):
        np.testing.assert_array_equal(a, b)
    original = app.sample_values
    captured = []
    def observe(*args, **kwargs):
        values = original(*args, **kwargs)
        captured.append(values)
        return values
    app.sample_values = observe
    try:
        result = wrapped()
    finally:
        app.sample_values = original
    report["sampling"] = dict(points=len(points), direct=timed(direct), wrapped=timed(wrapped),
        reuses_native_arrays=all(np.shares_memory(a,b) for a,b in zip(
            captured[0], (result.values, result.owners, result.valid))),
        pointset_bytes=points.nbytes, pointset_copies_coordinates=not np.shares_memory(points.positions,positions))
    subset = sm.select_fields(fields, "b3")
    merged = sm.merge_fields((magnetic, subset), names=("b1", "b2", "b3", "b3_copy"))
    report["field_copies"] = dict(input_payload_bytes=fields.values.nbytes,
        selected_payload_bytes=subset.values.nbytes, merged_payload_bytes=merged.values.nbytes,
        selection_shares_input=np.shares_memory(subset.values,fields.values),
        merge_shares_input=np.shares_memory(merged.values,magnetic.values))

    state = sm.mhd_fields(fields,model=model,outputs=("density","temperature"))
    def compose_old():
        selected_b = sm.select_fields(fields,("b1","b2","b3"))
        current = sm.current_density(selected_b,units=model.units.magnetic_si)
        scaled = sm.derive_many(state,{"temperature_MK":"MK","density_cgs":"g cm^-3"},
            lambda ctx: {"temperature_MK":ctx.field("temperature")*1e-6,
                         "density_cgs":ctx.field("density")})
        return sm.merge_fields((scaled,current))
    def compose_selected():
        current = sm.current_density(fields,units=model.units.magnetic_si,components=("b1","b2","b3"))
        return sm.derive_many({"state":state,"current":current},
            {"temperature_MK":"MK","density_cgs":"g cm^-3","jz":current.fields[2].units},
            lambda ctx: {"temperature_MK":ctx.field("temperature",group="state")*1e-6,
                         "density_cgs":ctx.field("density",group="state"),
                         "jz":ctx.field("jz",group="current")})
    old, before = allocated(compose_old)
    selected, after = allocated(compose_selected)
    np.testing.assert_array_equal(selected.values,old.values[...,[0,1,4]])
    report["selected_composition"] = dict(previous=before,selected=after,
        previous_payload_bytes=old.values.nbytes,selected_payload_bytes=selected.values.nbytes)

    report["mhd"] = {}
    for label, options in (("velocity_only", {"outputs":"velocity"}), ("default", {})):
        operation = lambda options=options: sm.mhd_fields(fields, model=model, **options)
        value, memory = allocated(operation)
        report["mhd"][label] = dict(payload_bytes=value.values.nbytes,
            evaluated_diagnostics=value.preparation_stats["evaluated_diagnostics"],
            controlled_upper_bytes=value.preparation_stats["controlled_upper_bytes"],
            **memory, **timed(operation, 5))
        del value
    produced = set()
    original_recover = mhd._recover
    def observe_recovery(*args, **kwargs):
        value = original_recover(*args, **kwargs)
        produced.update(value[0])
        return value
    mhd._recover = observe_recovery
    try:
        sm.mhd_fields(fields, model=model, outputs="velocity")
    finally:
        mhd._recover = original_recover
    report["mhd"]["returned_keys_for_velocity_request"] = sorted(produced)

    seeds = sm.PointSet.from_plane(sm.Plane([.1,.1,.5], [.8,0,0], [0,.8,0], (8,8)))
    report["trace"] = {}
    reference = None
    for max_steps in (100, 5000):
        lines, memory = allocated(lambda: app.trace(magnetic, seeds, step=.02, max_steps=max_steps))
        if reference is None:
            reference = lines.positions.copy()
        else:
            np.testing.assert_array_equal(lines.positions, reference)
        report["trace"][str(max_steps)] = dict(**memory, stored_points=len(lines.positions),
            final_position_bytes=lines.positions.nbytes,
            one_branch_path_capacity_bytes=len(seeds)*(_PATH_SEGMENT_STEPS+1)*3*8,
            native_dense_capacity_bytes=len(seeds)*(max_steps+1)*3*8)
        del lines

    with tempfile.TemporaryDirectory(prefix="simesh-interface-audit-") as directory:
        directory = Path(directory)
        _, memory = allocated(lambda: sm.save_result(directory/"points.npz", result))
        restored = sm.load_result(directory/"points.npz").result
        np.testing.assert_array_equal(restored.values, result.values)
        encoder = results_io._Encoder()
        encoder.node(result)
        report["save_result"] = dict(**memory, points=len(points),
            snapshot_bytes=sum(a.nbytes for a in encoder.arrays.values() if not np.shares_memory(a,points.positions)
                and not np.shares_memory(a,points.ids)),
            compression_chunk_bytes=results_io._WRITE_CHUNK_BYTES,
            numerical_payload_bytes=points.nbytes+result.values.nbytes+result.owners.nbytes+result.valid.nbytes,
            compressed_file_bytes=(directory/"points.npz").stat().st_size)
        ids = np.arange(10000,dtype=np.int64)
        sizes = []
        original_write = result_shards._write_json
        def observe_write(path, data):
            original_write(path, data)
            sizes.append(path.stat().st_size)
        def batches():
            for first in range(0,len(ids),250):
                selected = ids[first:first+250]
                seeds = sm.PointSet(np.full((len(selected),3),.5), selected)
                yield sm.LineSet(seeds, np.empty((0,3)), np.zeros(2*len(selected)+1,dtype=np.int64),
                                np.full((len(selected),2),sm.LineSet.NOT_REQUESTED,dtype=np.int64),None)
        result_shards._write_json = observe_write
        try:
            shards = sm.save_result_shards(directory/"shards", batches(), seed_ids=ids)
        finally:
            result_shards._write_json = original_write
        assert shards.manifest["complete"]
        report["shard_manifest"] = dict(schema_version=shards.manifest["schema_version"],
            seeds=len(ids), shard_count=40, writes=len(sizes),
            seed_ids_file_bytes=(directory/"shards"/"seed_ids.npy").stat().st_size,
            cumulative_bytes=sum(sizes), final_bytes=sizes[-1],
            npz_bytes=sum(p.stat().st_size for p in (directory/"shards").glob("*.npz")))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
