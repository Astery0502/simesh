"""Separate value kernels from planning inside the retained halo application."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import lfe_001 as b
import simesh_rewrite.refined_halo as halo
import simesh_rewrite.coarser_workspace_application as coarse
from m1_optimization import original_module


@contextmanager
def value_stages(totals):
    saved = []
    targets = [(module, name) for module in (halo, coarse) for name in (
        "copy_region_into_unchecked", "restrict_cartesian_2to1_into_unchecked",
        "apply_cartesian_physical_widening_unchecked")]
    targets.append((halo, "prolong_cartesian_2to1_into_unchecked"))
    targets += [(halo, name) for name in (
        "fill_coarser_workspace_boxes_unchecked", "fill_coarser_slope_support_plan_unchecked",
        "fill_finer_restriction_boxes_unchecked", "fill_same_level_source_boxes_unchecked")]
    try:
        for module, name in targets:
            original = getattr(module, name)
            saved.append((module, name, original))
            def timed(*args, _name=name, _original=original, **kwargs):
                start = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    totals[_name] = totals.get(_name, 0.0) + time.perf_counter() - start
            setattr(module, name, timed)
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = b.make_fixture()
    _, lo, hi = next(row for row in b.roi_bounds(fixture) if row[0] == "medium")
    selection = b.refined_region_windows(*b.selection_args(fixture, lo, hi))
    output = np.full((len(selection.leaf_ids), 3, 4, 4, 4), b.SENTINEL)
    accumulator = np.asarray([1.25])
    retained = halo._preflight_chunk_actions
    baseline = original_module("refined_halo")._preflight_chunk_actions
    reports = []
    with tempfile.TemporaryDirectory(prefix="simesh-halo-attribution-") as directory:
        fd, reader, _ = b.open_native_fixture(Path(directory) / "native.dat", fixture)
        try:
            expected = None
            for label, preflight in (("M1", baseline), ("retained", retained)):
                halo._preflight_chunk_actions = preflight
                accumulator[0] = 1.25
                output.view(np.uint64).fill(b.SENTINEL_BITS)
                recorder, kernels = b.StageRecorder(), {}
                recorder.started_at = time.perf_counter()
                with b.instrument_stages(recorder), value_stages(kernels):
                    stats = b.execute_selected_refined_curl_from_blocks(*b.execution_args(
                        fixture, reader, b.i3(0, 1, 2), selection, 57, output, accumulator))
                bits = (output.view(np.uint64).copy(), accumulator.view(np.uint64).copy())
                if expected is not None:
                    assert all(np.array_equal(a, e) for a, e in zip(bits, expected))
                expected = bits
                reports.append({"variant": label, "stages_nested_seconds": recorder.seconds,
                                "kernel_boundaries_seconds": kernels, "stats": b.stats_record(stats)})
        finally:
            halo._preflight_chunk_actions = retained
            os.close(fd)
    args.output.write_text(json.dumps({"environment": b.environment_record(),
        "records": reports, "scope": "Instrumented attribution only; kernel boundaries include dispatch, not pure FLOP time. Nested stages must not be summed."}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
