"""PRL-001 loop-organization, composition, and current mapping evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite._prolongation import (
    prolong_cartesian_2to1_into_fine_centric_unchecked,
    prolong_cartesian_2to1_into_unchecked,
)
from simesh_rewrite.limiter_reference import (
    three_point_limited_slope_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.prolongation import prolong_cartesian_2to1_into
from simesh_rewrite.prolongation_reference import (
    prolong_cartesian_2to1_reference,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def interleaved_timings(operations, repeats: int):
    samples = {name: [] for name, _ in operations}
    orders = []
    for repeat in range(repeats):
        order = [
            operations[(repeat + offset) % len(operations)]
            for offset in range(len(operations))
        ]
        orders.append([name for name, _ in order])
        for name, operation in order:
            started = time.perf_counter()
            operation()
            samples[name].append(time.perf_counter() - started)
    return (
        {name: statistics.median(values) for name, values in samples.items()},
        orders,
    )


def phase_distribution(
    fine_lower: np.ndarray,
    fine_upper: np.ndarray,
    fine_origin: np.ndarray,
) -> dict:
    counts = np.zeros(8, dtype=np.int64)
    for i in range(int(fine_lower[0]), int(fine_upper[0])):
        pi = (i - int(fine_origin[0])) % 2
        for j in range(int(fine_lower[1]), int(fine_upper[1])):
            pj = (j - int(fine_origin[1])) % 2
            for k in range(int(fine_lower[2]), int(fine_upper[2])):
                pk = (k - int(fine_origin[2])) % 2
                counts[pi + 2 * pj + 4 * pk] += 1
    return {
        f"{phase & 1}{(phase >> 1) & 1}{(phase >> 2) & 1}": int(counts[phase])
        for phase in range(8)
    }


def throughput(seconds: float, output_cells: int) -> dict:
    return {
        "million_fine_field_cells_per_second": output_cells / seconds / 1.0e6,
        "effective_64byte_gb_per_second": 64 * output_cells / seconds / 1.0e9,
    }


def direct_case(
    label: str,
    slots: int,
    fields: int,
    coarse_shape: tuple[int, int, int],
    coarse_origin: tuple[int, int, int],
    fine_shape: tuple[int, int, int],
    fine_lower: tuple[int, int, int],
    fine_upper: tuple[int, int, int],
    fine_origin: tuple[int, int, int],
    repeats: int,
    seed: int,
    *,
    include_reference: bool = False,
    trace_checked: bool = False,
) -> dict:
    rng = np.random.default_rng(
        seed + 1009 * slots + 9176 * fields + sum(coarse_shape) + sum(fine_shape)
    )
    coarse = np.ascontiguousarray(
        rng.normal(size=(slots, fields, *coarse_shape)), dtype=np.float64
    )
    coarse_valid_lower = i3(0, 0, 0)
    coarse_valid_upper = i3(*coarse_shape)
    coarse_origin_array = i3(*coarse_origin)
    fine_lower_array = i3(*fine_lower)
    fine_upper_array = i3(*fine_upper)
    fine_origin_array = i3(*fine_origin)
    sentinel = np.asarray([0x7FF800000000B501], dtype=np.uint64).view(np.float64)[0]
    checked_output = np.full(
        (slots, fields, *fine_shape), sentinel, dtype=np.float64
    )
    coarse_output = np.full_like(checked_output, sentinel)
    fine_output = np.full_like(checked_output, sentinel)

    def checked() -> None:
        prolong_cartesian_2to1_into(
            coarse,
            coarse_valid_lower,
            coarse_valid_upper,
            coarse_origin_array,
            checked_output,
            fine_lower_array,
            fine_upper_array,
            fine_origin_array,
        )

    def coarse_centric() -> None:
        prolong_cartesian_2to1_into_unchecked(
            coarse,
            coarse_origin_array,
            coarse_output,
            fine_lower_array,
            fine_upper_array,
            fine_origin_array,
        )

    def fine_centric() -> None:
        prolong_cartesian_2to1_into_fine_centric_unchecked(
            coarse,
            coarse_origin_array,
            fine_output,
            fine_lower_array,
            fine_upper_array,
            fine_origin_array,
        )

    operations = [
        ("checked_coarse_centric", checked),
        ("unchecked_coarse_centric", coarse_centric),
        ("unchecked_fine_centric", fine_centric),
    ]
    checked()
    coarse_centric()
    fine_centric()
    exact_coarse = bits_equal(checked_output, coarse_output)
    exact_fine = bits_equal(checked_output, fine_output)
    if not exact_coarse or not exact_fine:
        raise AssertionError("PRL loop organizations disagree bitwise")

    reference_output = None
    if include_reference:
        reference_output = np.full_like(checked_output, sentinel)

        def scalar_reference() -> None:
            prolong_cartesian_2to1_reference(
                coarse,
                coarse_valid_lower,
                coarse_valid_upper,
                coarse_origin_array,
                reference_output,
                fine_lower_array,
                fine_upper_array,
                fine_origin_array,
            )

        scalar_reference()
        if not bits_equal(checked_output, reference_output):
            raise AssertionError("PRL scalar reference disagrees bitwise")
        operations.append(("scalar_reference", scalar_reference))

    timings, timing_orders = interleaved_timings(operations, repeats)
    if not bits_equal(checked_output, coarse_output) or not bits_equal(
        checked_output, fine_output
    ):
        raise AssertionError("timed PRL arrays disagree")
    if reference_output is not None and not bits_equal(
        checked_output, reference_output
    ):
        raise AssertionError("timed PRL scalar array disagrees")

    traced_retained = None
    traced_peak = None
    if trace_checked:
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        checked()
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_retained = after_current - before_current
        traced_peak = peak - before_current

    target_extent = fine_upper_array - fine_lower_array
    output_cells = slots * fields * int(np.prod(target_extent))
    result = {
        "label": label,
        "slots": slots,
        "fields": fields,
        "coarse_shape": list(coarse_shape),
        "coarse_origin": list(coarse_origin),
        "fine_shape": list(fine_shape),
        "fine_lower": list(fine_lower),
        "fine_upper": list(fine_upper),
        "fine_origin": list(fine_origin),
        "target_fine_field_cells": output_cells,
        "fine_output_bytes": checked_output.nbytes,
        "target_output_bytes": 8 * output_cells,
        "phase_distribution_spatial_cells": phase_distribution(
            fine_lower_array, fine_upper_array, fine_origin_array
        ),
        "exact_checked_unchecked_coarse_arrays": exact_coarse,
        "exact_checked_fine_centric_arrays": exact_fine,
        "exact_checked_scalar_reference_array": (
            None
            if reference_output is None
            else bits_equal(checked_output, reference_output)
        ),
        "timing_orders": timing_orders,
        "seconds": timings,
        "checked_over_unchecked_coarse": (
            timings["checked_coarse_centric"]
            / timings["unchecked_coarse_centric"]
        ),
        "fine_centric_over_coarse_centric": (
            timings["unchecked_fine_centric"]
            / timings["unchecked_coarse_centric"]
        ),
        "checked_traced_retained_bytes": traced_retained,
        "checked_traced_peak_bytes": traced_peak,
    }
    result["throughput"] = {
        name: throughput(seconds, output_cells)
        for name, seconds in timings.items()
    }
    return result


def standard_case(repeats: int, seed: int) -> dict:
    return direct_case(
        "standard",
        64,
        4,
        (10, 10, 10),
        (1, 1, 1),
        (16, 16, 16),
        (0, 0, 0),
        (16, 16, 16),
        (0, 0, 0),
        repeats,
        seed,
        trace_checked=True,
    )


def small_reference_case(repeats: int, seed: int) -> dict:
    return direct_case(
        "bounded_scalar_reference",
        1,
        1,
        (6, 6, 6),
        (1, 1, 1),
        (8, 8, 8),
        (0, 0, 0),
        (8, 8, 8),
        (0, 0, 0),
        max(1, min(repeats, 3)),
        seed,
        include_reference=True,
    )


def rst_prl_composition(repeats: int, seed: int) -> dict:
    slots = 64
    fields = 4
    rng = np.random.default_rng(seed + 71001)
    fine_source = np.ascontiguousarray(
        rng.normal(size=(slots, fields, 16, 16, 16)), dtype=np.float64
    )
    assembled_seed = np.ascontiguousarray(
        rng.normal(size=(slots, fields, 10, 10, 10)), dtype=np.float64
    )
    zero = i3(0, 0, 0)
    fine_upper = i3(16, 16, 16)
    coarse_lower = i3(1, 1, 1)
    coarse_valid_upper = i3(10, 10, 10)
    coarse_origin = i3(1, 1, 1)

    resident_coarse = np.empty_like(assembled_seed)
    resident_fine = np.empty_like(fine_source)

    def resident_once() -> dict:
        started = time.perf_counter()
        np.copyto(resident_coarse, assembled_seed)
        assembly_seconds = time.perf_counter() - started
        started = time.perf_counter()
        restrict_cartesian_2to1_into(
            fine_source,
            zero,
            fine_upper,
            resident_coarse,
            coarse_lower,
        )
        restriction_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prolong_cartesian_2to1_into(
            resident_coarse,
            zero,
            coarse_valid_upper,
            coarse_origin,
            resident_fine,
            zero,
            fine_upper,
            zero,
        )
        prolongation_seconds = time.perf_counter() - started
        return {
            "assembly_seconds": assembly_seconds,
            "restriction_seconds": restriction_seconds,
            "prolongation_seconds": prolongation_seconds,
            "wall_seconds": (
                assembly_seconds + restriction_seconds + prolongation_seconds
            ),
        }

    resident_once()
    resident_samples = [resident_once() for _ in range(repeats)]
    resident = min(
        resident_samples,
        key=lambda value: abs(
            value["wall_seconds"]
            - statistics.median(x["wall_seconds"] for x in resident_samples)
        ),
    )
    resident_exact = resident_fine.copy()

    cases = []
    for capacity in (1, 8, 64):
        coarse_batch = np.empty((capacity, fields, 10, 10, 10), dtype=np.float64)
        fine_batch = np.empty((capacity, fields, 16, 16, 16), dtype=np.float64)
        bounded_output = np.empty_like(resident_exact)

        def bounded_once() -> dict:
            first = 0
            batches = 0
            assembly_seconds = 0.0
            restriction_seconds = 0.0
            prolongation_seconds = 0.0
            output_store_seconds = 0.0
            wall_started = time.perf_counter()
            while first < slots:
                count = min(capacity, slots - first)
                started = time.perf_counter()
                np.copyto(
                    coarse_batch[:count], assembled_seed[first : first + count]
                )
                assembly_seconds += time.perf_counter() - started
                started = time.perf_counter()
                restrict_cartesian_2to1_into(
                    fine_source[first : first + count],
                    zero,
                    fine_upper,
                    coarse_batch[:count],
                    coarse_lower,
                )
                restriction_seconds += time.perf_counter() - started
                started = time.perf_counter()
                prolong_cartesian_2to1_into(
                    coarse_batch[:count],
                    zero,
                    coarse_valid_upper,
                    coarse_origin,
                    fine_batch[:count],
                    zero,
                    fine_upper,
                    zero,
                )
                prolongation_seconds += time.perf_counter() - started
                started = time.perf_counter()
                bounded_output[first : first + count] = fine_batch[:count]
                output_store_seconds += time.perf_counter() - started
                first += count
                batches += 1
            wall_seconds = time.perf_counter() - wall_started
            exact = bits_equal(bounded_output, resident_exact)
            if not exact:
                raise AssertionError("bounded RST/PRL output disagrees with resident")
            return {
                "capacity": capacity,
                "batches": batches,
                "assembly_seconds": assembly_seconds,
                "restriction_seconds": restriction_seconds,
                "prolongation_seconds": prolongation_seconds,
                "output_store_seconds": output_store_seconds,
                "wall_seconds": wall_seconds,
                "exact_full_output_array": exact,
            }

        bounded_once()
        samples = [bounded_once() for _ in range(repeats)]
        chosen = min(
            samples,
            key=lambda value: abs(
                value["wall_seconds"]
                - statistics.median(x["wall_seconds"] for x in samples)
            ),
        )
        output_cells = slots * fields * 16**3
        chosen["prolongation_throughput"] = throughput(
            chosen["prolongation_seconds"], output_cells
        )
        chosen["coarse_workspace_bytes"] = coarse_batch.nbytes
        chosen["fine_batch_bytes"] = fine_batch.nbytes
        cases.append(chosen)

    return {
        "description": (
            "RST interior overwrite into an explicitly preassembled coarse "
            "workspace, followed by PRL; no relation or halo policy"
        ),
        "slots": slots,
        "fields": fields,
        "coarse_shape": [10, 10, 10],
        "fine_shape": [16, 16, 16],
        "resident": resident,
        "resident_output_bytes": resident_exact.nbytes,
        "cases": cases,
    }


def current_prolongation_fixture(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
):
    root = i3(4, 4, 4)
    _, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        if tuple(int(value) for value in coordinate) == (1, 1, 1):
            flags.append(False)
            flags.extend([True] * 8)
        else:
            flags.append(True)
    forest = AMRForest(3, 4, 4, 4, np.asarray(flags, dtype=np.int32))
    block = np.asarray([4, 4, 4], dtype=np.uint32)
    mesh = AMRMesh(
        3,
        block,
        np.asarray([16, 16, 16], dtype=np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(2),
        np.uint32(1),
        forest,
    )
    rng = np.random.default_rng(20260902)
    interior = np.ascontiguousarray(
        rng.normal(size=(forest.nleafs, 1, 4, 4, 4))
    )
    mesh.load_interior_data(interior)
    started = time.perf_counter()
    mesh.apply_ghost_cells()
    current_seconds = time.perf_counter() - started
    candidates = np.flatnonzero(np.asarray(forest.neighbor_type)[:, 14] == 2)
    if candidates.size == 0:
        raise AssertionError("current PRL fixture has no COARSER face")
    leaf = int(candidates[0])
    coarse = np.asarray(mesh.datac)[leaf].transpose(3, 0, 1, 2)[None].copy()
    current = mesh.padded_view()[leaf].transpose(3, 0, 1, 2)[None].copy()
    rnode = np.asarray(mesh.rnode)[leaf].copy()
    return coarse, current, rnode, current_seconds


def ordered_ulp_distance(left: float, right: float) -> int:
    values = np.asarray([left, right], dtype=np.float64)
    bits_values = values.view(np.uint64)
    sign_mask = np.uint64(1 << 63)
    ordered = np.where(
        bits_values & sign_mask,
        ~bits_values,
        bits_values | sign_mask,
    )
    return abs(int(ordered[0]) - int(ordered[1]))


def current_comparison() -> dict:
    dyadic_coarse, dyadic_current, _, dyadic_seconds = current_prolongation_fixture(
        np.zeros(3), np.ones(3)
    )
    dyadic_exact = np.full_like(dyadic_current, -61.0)
    prolong_cartesian_2to1_into(
        dyadic_coarse,
        i3(3, 1, 1),
        i3(6, 5, 5),
        i3(2, 2, 2),
        dyadic_exact,
        i3(6, 2, 2),
        i3(8, 6, 6),
        i3(2, 2, 2),
    )
    dyadic_region = (slice(None), slice(None), slice(6, 8), slice(2, 6), slice(2, 6))
    dyadic_equal = bits_equal(
        dyadic_exact[dyadic_region], dyadic_current[dyadic_region]
    )
    if not dyadic_equal:
        raise AssertionError("dyadic current PRL comparison is not bitwise")

    coarse, current, rnode, nondyadic_seconds = current_prolongation_fixture(
        np.asarray([-0.7, 1.1, -2.3]),
        np.asarray([1.9, 4.7, 7.2]),
    )
    exact = np.full_like(current, -67.0)
    prolong_cartesian_2to1_into(
        coarse,
        i3(3, 1, 1),
        i3(6, 5, 5),
        i3(2, 2, 2),
        exact,
        i3(6, 2, 2),
        i3(8, 6, 6),
        i3(2, 2, 2),
    )
    eps = np.finfo(np.float64).eps
    gamma6 = (6.0 * eps) / (1.0 - 6.0 * eps)
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    maximum_eta_delta = 0.0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_ulp = 0
    maximum_bound = 0.0
    all_within_bound = True
    for i in range(6, 8):
        for j in range(2, 6):
            for k in range(2, 6):
                centers = []
                current_eta = []
                exact_eta = []
                for axis, index in enumerate((i, j, k)):
                    h_fine = rnode[6 + axis]
                    h_coarse = 2.0 * h_fine
                    inverse = 1.0 / h_coarse
                    x_fine_min = rnode[axis] - 2.0 * h_fine
                    x_coarse_min = rnode[axis] - 2.0 * h_coarse
                    x_fine = x_fine_min + (float(index) + 0.5) * h_fine
                    center_index = int((x_fine - x_coarse_min) * inverse)
                    expected_index = 2 + ((index - 2) // 2)
                    if center_index != expected_index:
                        raise AssertionError("current/exact coarse indices differ")
                    x_coarse = x_coarse_min + (
                        float(center_index) + 0.5
                    ) * h_coarse
                    eta = (x_fine - x_coarse) * inverse
                    exact_phase = -0.25 if (index - 2) % 2 == 0 else 0.25
                    current_eta.append(eta)
                    exact_eta.append(exact_phase)
                    centers.append(center_index)
                    maximum_eta_delta = max(
                        maximum_eta_delta, abs(eta - exact_phase)
                    )
                I, J, K = centers
                center = coarse[0, 0, I, J, K]
                slopes = (
                    three_point_limited_slope_reference(
                        coarse[0, 0, I - 1, J, K],
                        center,
                        coarse[0, 0, I + 1, J, K],
                    ),
                    three_point_limited_slope_reference(
                        coarse[0, 0, I, J - 1, K],
                        center,
                        coarse[0, 0, I, J + 1, K],
                    ),
                    three_point_limited_slope_reference(
                        coarse[0, 0, I, J, K - 1],
                        center,
                        coarse[0, 0, I, J, K + 1],
                    ),
                )
                b_eta = sum(
                    abs(float(slope))
                    * abs(current_eta[axis] - exact_eta[axis])
                    for axis, slope in enumerate(slopes)
                )
                m_exact = abs(float(center)) + sum(
                    abs(float(np.float64(slope) * np.float64(exact_eta[axis])))
                    for axis, slope in enumerate(slopes)
                )
                m_current = abs(float(center)) + sum(
                    abs(float(np.float64(slope) * np.float64(current_eta[axis])))
                    for axis, slope in enumerate(slopes)
                )
                bound = b_eta + gamma6 * (m_exact + m_current) + 8.0 * smallest
                current_value = float(current[0, 0, i, j, k])
                exact_value = float(exact[0, 0, i, j, k])
                error = abs(current_value - exact_value)
                all_within_bound = bool(all_within_bound and error <= bound)
                maximum_bound = max(maximum_bound, bound)
                maximum_absolute = max(maximum_absolute, error)
                scale = max(abs(current_value), abs(exact_value))
                if scale:
                    maximum_relative = max(maximum_relative, error / scale)
                maximum_ulp = max(
                    maximum_ulp,
                    ordered_ulp_distance(current_value, exact_value),
                )
    if not all_within_bound:
        raise AssertionError("non-dyadic current PRL error exceeds bound")
    return {
        "description": (
            "current AMRMesh aggregate comparison; current physical-coordinate "
            "eta differs intentionally from exact integer phase"
        ),
        "not_a_direct_kernel_runtime_comparator": True,
        "dyadic": {
            "current_apply_ghost_cells_seconds": dyadic_seconds,
            "compared_fine_field_cells": 32,
            "bitwise_equal": dyadic_equal,
        },
        "nondyadic": {
            "current_apply_ghost_cells_seconds": nondyadic_seconds,
            "compared_fine_field_cells": 32,
            "all_within_contract_bound": bool(all_within_bound),
            "maximum_eta_absolute_delta": maximum_eta_delta,
            "maximum_absolute_difference": maximum_absolute,
            "maximum_relative_difference": maximum_relative,
            "maximum_ulp_distance": maximum_ulp,
            "maximum_bound": maximum_bound,
        },
    }


def weno_mapping_metrics() -> dict:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        return {
            "available": False,
            "description": "WENO metadata file is unavailable",
        }
    from simesh.amrvac.datio import get_metadata

    header, flags, _ = get_metadata(str(path))
    root = header["domain_nx"] // header["block_nx"]
    forest = AMRForest(3, *map(int, root), np.asarray(flags, dtype=np.int32))
    mesh = AMRMesh(
        3,
        header["block_nx"].astype(np.uint32),
        header["domain_nx"].astype(np.uint32),
        header["xmin"].astype(np.float64),
        header["xmax"].astype(np.float64),
        np.uint32(0),
        np.uint32(1),
        forest,
    )
    rnode = np.asarray(mesh.rnode)
    maximum_eta_delta = 0.0
    wrong_indices = 0
    eta_patterns: list[int] = []
    samples = 0
    started = time.perf_counter()
    for leaf in range(rnode.shape[0]):
        for axis in range(3):
            h_fine = rnode[leaf, 6 + axis]
            h_coarse = 2.0 * h_fine
            inverse = 1.0 / h_coarse
            x_fine_min = rnode[leaf, axis] - 2.0 * h_fine
            x_coarse_min = rnode[leaf, axis] - 2.0 * h_coarse
            for index in range(int(header["block_nx"][axis]) + 4):
                x_fine = x_fine_min + (float(index) + 0.5) * h_fine
                center_index = int((x_fine - x_coarse_min) * inverse)
                expected_index = 2 + ((index - 2) // 2)
                wrong_indices += center_index != expected_index
                x_coarse = x_coarse_min + (
                    float(center_index) + 0.5
                ) * h_coarse
                eta = np.float64((x_fine - x_coarse) * inverse)
                exact_eta = -0.25 if (index - 2) % 2 == 0 else 0.25
                maximum_eta_delta = max(
                    maximum_eta_delta, abs(float(eta) - exact_eta)
                )
                pattern = bits(eta)
                if pattern not in eta_patterns:
                    eta_patterns.append(pattern)
                samples += 1
    seconds = time.perf_counter() - started
    if wrong_indices:
        raise AssertionError("WENO current/integer mapping indices differ")
    return {
        "available": True,
        "description": (
            "mapping/eta metadata only; staggered WENO payload is outside PRL support"
        ),
        "leaves": int(rnode.shape[0]),
        "mapping_samples": samples,
        "seconds": seconds,
        "wrong_coarse_indices": int(wrong_indices),
        "eta_bit_pattern_count": len(eta_patterns),
        "maximum_eta_absolute_delta": maximum_eta_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--composition-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.repeats < 1 or args.composition_repeats < 1:
        raise ValueError("repeat counts must be positive")

    standard = standard_case(args.repeats, args.seed)
    shape_cases = [
        direct_case(
            f"shape_{extent}",
            64,
            4,
            (extent // 2 + 2,) * 3,
            (1, 1, 1),
            (extent,) * 3,
            (0, 0, 0),
            (extent,) * 3,
            (0, 0, 0),
            args.repeats,
            args.seed,
        )
        for extent in (4, 8, 32)
    ]
    field_cases = [
        direct_case(
            f"fields_{fields}",
            64,
            fields,
            (10, 10, 10),
            (1, 1, 1),
            (16, 16, 16),
            (0, 0, 0),
            (16, 16, 16),
            (0, 0, 0),
            args.repeats,
            args.seed,
        )
        for fields in (1, 8)
    ]
    slot_cases = [
        direct_case(
            f"slots_{slots}",
            slots,
            4,
            (10, 10, 10),
            (1, 1, 1),
            (16, 16, 16),
            (0, 0, 0),
            (16, 16, 16),
            (0, 0, 0),
            args.repeats,
            args.seed,
        )
        for slots in (1, 8, 256)
    ]
    target_cases = [
        direct_case(
            "odd_partial_target",
            64,
            4,
            (10, 10, 10),
            (1, 1, 1),
            (16, 16, 16),
            (1, 0, 1),
            (16, 15, 14),
            (0, 0, 0),
            args.repeats,
            args.seed,
        ),
        direct_case(
            "translated_target",
            64,
            4,
            (10, 10, 10),
            (1, 1, 1),
            (20, 20, 20),
            (2, 3, 1),
            (18, 17, 16),
            (2, 3, 1),
            args.repeats,
            args.seed,
        ),
        direct_case(
            "negative_relative_mapping",
            64,
            4,
            (10, 10, 10),
            (2, 2, 2),
            (16, 16, 16),
            (0, 0, 0),
            (16, 16, 16),
            (2, 2, 2),
            args.repeats,
            args.seed,
        ),
    ]

    report = {
        "capability": "PRL-001",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats": args.repeats,
        "composition_repeats": args.composition_repeats,
        "standard": standard,
        "shape_scaling": [shape_cases[0], shape_cases[1], standard, shape_cases[2]],
        "field_scaling": [field_cases[0], standard, field_cases[1]],
        "slot_scaling": [slot_cases[0], slot_cases[1], standard, slot_cases[2]],
        "target_scaling": [standard, *target_cases],
        "bounded_scalar_reference": small_reference_case(args.repeats, args.seed),
        "rst_coarse_workspace_prl_composition": rst_prl_composition(
            args.composition_repeats,
            args.seed,
        ),
        "current_comparison_descriptive": current_comparison(),
        "weno_mapping_only": weno_mapping_metrics(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
