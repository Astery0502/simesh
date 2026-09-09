"""WSP-001 exact composition, compatibility, and fixed-cost probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.chunking import (
    workspace_nbytes as compatibility_nbytes,
    workspace_slot_capacity as compatibility_capacity,
)
from simesh_rewrite.workspace import workspace_nbytes, workspace_slot_capacity
from simesh_rewrite.workspace_reference import (
    workspace_nbytes_reference,
    workspace_slot_capacity_reference,
)


def median_seconds_per_call(operation, calls: int, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            operation()
        samples.append((time.perf_counter() - started) / calls)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    slot_capacity = 64
    field_count = 8
    workspace_shape = np.asarray([18, 20, 22], dtype=np.int64)
    per_slot = workspace_nbytes(1, field_count, workspace_shape)
    budget_bytes = per_slot * 137 + per_slot - 1
    block_count = 1000
    expected_bytes = workspace_nbytes_reference(
        slot_capacity,
        field_count,
        workspace_shape,
    )
    expected_capacity = workspace_slot_capacity_reference(
        budget_bytes,
        block_count,
        field_count,
        workspace_shape,
    )
    if workspace_nbytes(
        slot_capacity, field_count, workspace_shape
    ) != expected_bytes or compatibility_nbytes(
        slot_capacity, field_count, workspace_shape
    ) != expected_bytes:
        raise AssertionError("forward accounting disagreement")
    if workspace_slot_capacity(
        budget_bytes, block_count, field_count, workspace_shape
    ) != expected_capacity or compatibility_capacity(
        budget_bytes, block_count, field_count, workspace_shape
    ) != expected_capacity:
        raise AssertionError("inverse accounting disagreement")

    actual_capacity = workspace_slot_capacity(
        2 * 1024 * 1024,
        2048,
        2,
        np.asarray([16, 16, 16], dtype=np.int64),
    )
    payload = np.empty((actual_capacity, 2, 16, 16, 16), dtype=np.float64)
    block_ids = np.empty(actual_capacity, dtype=np.int64)
    actual_allocated_bytes = payload.nbytes + block_ids.nbytes
    actual_formula_bytes = workspace_nbytes(
        actual_capacity,
        2,
        np.asarray([16, 16, 16], dtype=np.int64),
    )
    if actual_allocated_bytes != actual_formula_bytes:
        raise AssertionError("actual arrays disagree with WSP bytes")

    operations = {
        "canonical_nbytes": lambda: workspace_nbytes(
            slot_capacity, field_count, workspace_shape
        ),
        "compatibility_nbytes": lambda: compatibility_nbytes(
            slot_capacity, field_count, workspace_shape
        ),
        "canonical_capacity": lambda: workspace_slot_capacity(
            budget_bytes, block_count, field_count, workspace_shape
        ),
        "compatibility_capacity": lambda: compatibility_capacity(
            budget_bytes, block_count, field_count, workspace_shape
        ),
    }
    latency = {
        name: median_seconds_per_call(operation, args.calls, args.repeats)
        for name, operation in operations.items()
    }

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    workspace_nbytes(slot_capacity, field_count, workspace_shape)
    workspace_slot_capacity(
        budget_bytes, block_count, field_count, workspace_shape
    )
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "capability": "WSP-001",
        "calls_per_sample": args.calls,
        "repeats": args.repeats,
        "latency_seconds_per_call": latency,
        "representative": {
            "slot_capacity": slot_capacity,
            "field_count": field_count,
            "workspace_shape": workspace_shape.tolist(),
            "bytes_per_slot": per_slot,
            "managed_bytes": expected_bytes,
            "budget_bytes": budget_bytes,
            "block_count": block_count,
            "inverse_capacity": expected_capacity,
        },
        "actual_allocation": {
            "budget_bytes": 2 * 1024 * 1024,
            "capacity": actual_capacity,
            "payload_plus_ids_bytes": actual_allocated_bytes,
            "formula_bytes": actual_formula_bytes,
        },
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
