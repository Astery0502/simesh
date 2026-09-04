from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import simesh_rewrite.amrvac_dat as amrvac_dat
from simesh_rewrite.amrvac_dat import (
    AMRVACV5ForestBinding,
    AMRVACV5Index,
    bind_amrvac_v5_forest,
    read_amrvac_v5_index,
)
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays


ROOT = Path(__file__).resolve().parents[2]


def _mixed_index() -> AMRVACV5Index:
    flags = np.asarray((False,) + (True,) * 8 + (True,), dtype=np.bool_)
    levels = np.asarray((2,) * 8 + (1,), dtype=np.int64)
    coordinates = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, 0),
        ),
        dtype=np.int64,
    )
    return AMRVACV5Index(
        byte_order="<",
        file_identity=(7, 11, 4096, 101, 103),
        offset_tree=200,
        offset_blocks=480,
        field_count=1,
        direction_count=3,
        dimension_count=3,
        declared_max_level=4,
        leaf_count=9,
        parent_count=1,
        iteration=2,
        time=0.5,
        domain_lower=np.asarray([-1.0, -1.0, -1.0], dtype=np.float64),
        domain_upper=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        domain_cell_counts=np.asarray([4, 2, 2], dtype=np.int64),
        block_cell_counts=np.asarray([2, 2, 2], dtype=np.int64),
        periodic=np.asarray([False, True, False], dtype=np.bool_),
        geometry="not-a-supported-numerical-geometry",
        staggered=True,
        field_names=("rho",),
        physics_type="hd",
        parameter_values=np.empty(0, dtype=np.float64),
        parameter_names=(),
        snapshot_next=0,
        slice_next=0,
        collapse_next=0,
        forest_flags=flags,
        block_levels=levels,
        block_coordinates=coordinates,
        block_offsets=np.arange(9, dtype=np.int64) * 128 + 480,
    )


def _binding_arrays(binding: AMRVACV5ForestBinding) -> tuple[np.ndarray, ...]:
    forest = binding.forest
    return (
        binding.root_shape,
        binding.coord_to_rank,
        binding.rank_to_coord,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        forest.root_node_ids,
    )


def test_mixed_depth_forest_binds_exact_canonical_leaf_order() -> None:
    index = _mixed_index()
    binding = bind_amrvac_v5_forest(index)

    assert type(binding) is AMRVACV5ForestBinding
    assert binding.source_file_identity == index.file_identity
    assert binding.source_file_identity is not index.file_identity
    assert np.array_equal(binding.root_shape, [2, 1, 1])
    assert np.array_equal(binding.coord_to_rank[:, 0, 0], [0, 1])
    assert np.array_equal(binding.rank_to_coord, [[0, 0, 0], [1, 0, 0]])

    forest = binding.forest
    assert forest.max_level == 2
    assert np.array_equal(
        forest.node_levels,
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    )
    assert np.array_equal(
        forest.node_coords,
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
    )
    assert np.array_equal(forest.parent_node_ids, [-1] + [0] * 8 + [-1])
    expected_children = np.full((10, 8), -1, dtype=np.int64)
    expected_children[0] = np.arange(1, 9, dtype=np.int64)
    assert np.array_equal(forest.child_node_ids, expected_children)
    assert np.array_equal(forest.node_leaf_ids, [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert np.array_equal(forest.leaf_node_ids, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.array_equal(forest.root_node_ids, [0, 9])
    assert np.array_equal(
        index.block_levels,
        forest.node_levels[forest.leaf_node_ids],
    )
    assert np.array_equal(
        index.block_coordinates,
        forest.node_coords[forest.leaf_node_ids],
    )
    assert (
        validate_refined_forest_arrays(
            binding.root_shape,
            binding.coord_to_rank,
            binding.rank_to_coord,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.parent_node_ids,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
        )
        == 2
    )

    arrays = _binding_arrays(binding)
    for value in arrays:
        assert value.flags.owndata
        assert value.flags.c_contiguous
        assert value.dtype == np.dtype(np.int64)
    for position, value in enumerate(arrays):
        assert not any(np.shares_memory(value, prior) for prior in arrays[:position])


def test_binding_retains_no_input_array_or_index_reference() -> None:
    index = _mixed_index()
    expected_levels = index.block_levels.copy()
    expected_coordinates = index.block_coordinates.copy()
    binding = bind_amrvac_v5_forest(index)

    index.domain_cell_counts[:] = 2
    index.block_cell_counts[:] = 1
    index.forest_flags[:] = True
    index.block_levels[:] = 4
    index.block_coordinates[:] = 99
    index.block_offsets[:] = -1

    assert np.array_equal(binding.root_shape, [2, 1, 1])
    assert np.array_equal(
        binding.forest.node_levels[binding.forest.leaf_node_ids],
        expected_levels,
    )
    assert np.array_equal(
        binding.forest.node_coords[binding.forest.leaf_node_ids],
        expected_coordinates,
    )


def test_geometry_periodicity_staggering_and_unoccupied_declared_levels_are_metadata() -> None:
    index = _mixed_index()._replace(
        geometry="spherical",
        periodic=np.ones(3, dtype=np.bool_),
        staggered=True,
        declared_max_level=63,
    )
    binding = bind_amrvac_v5_forest(index)
    assert binding.forest.max_level == 2


def test_binding_performs_no_file_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_pread(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("DAT-002 must not perform file I/O")

    monkeypatch.setattr(amrvac_dat.os, "pread", fail_pread)
    bind_amrvac_v5_forest(_mixed_index())


def test_binding_requires_exact_index_type() -> None:
    index = _mixed_index()
    with pytest.raises(TypeError, match="exactly an AMRVACV5Index"):
        bind_amrvac_v5_forest(tuple(index))  # type: ignore[arg-type]

    class DerivedIndex(AMRVACV5Index):
        pass

    derived = DerivedIndex(*index)
    with pytest.raises(TypeError, match="exactly an AMRVACV5Index"):
        bind_amrvac_v5_forest(derived)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        (
            "domain_cell_counts",
            np.asarray([4, 2, 2], dtype=np.int32),
            TypeError,
            "dtype int64",
        ),
        (
            "block_cell_counts",
            np.asarray([2, 2], dtype=np.int64),
            ValueError,
            "shape",
        ),
        (
            "forest_flags",
            np.ones(10, dtype=np.int64),
            TypeError,
            "dtype bool",
        ),
        (
            "block_levels",
            np.ones(8, dtype=np.int64),
            ValueError,
            "shape",
        ),
        (
            "block_coordinates",
            np.asfortranarray(np.ones((9, 3), dtype=np.int64)),
            ValueError,
            "C-contiguous",
        ),
        (
            "block_offsets",
            np.ones((9, 1), dtype=np.int64),
            ValueError,
            "shape",
        ),
    ],
)
def test_binding_validates_index_array_representations(
    field: str,
    replacement: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        bind_amrvac_v5_forest(_mixed_index()._replace(**{field: replacement}))


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        ("leaf_count", np.int64(9), TypeError, "exact Python int"),
        ("parent_count", -1, ValueError, "nonnegative"),
        ("declared_max_level", 0, ValueError, "positive"),
        ("file_identity", (1, 2, 3), TypeError, "five-int tuple"),
        ("file_identity", (1, 2, 3, 4, np.int64(5)), TypeError, "five-int tuple"),
    ],
)
def test_binding_validates_scalar_provenance_and_counts(
    field: str,
    replacement: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        bind_amrvac_v5_forest(_mixed_index()._replace(**{field: replacement}))


def test_root_shape_failures_are_reported_before_construction() -> None:
    index = _mixed_index()
    with pytest.raises(ValueError, match="positive"):
        bind_amrvac_v5_forest(
            index._replace(domain_cell_counts=np.asarray([0, 2, 2], dtype=np.int64))
        )
    with pytest.raises(ValueError, match="positive"):
        bind_amrvac_v5_forest(
            index._replace(block_cell_counts=np.asarray([-1, 2, 2], dtype=np.int64))
        )
    with pytest.raises(ValueError, match="divisible"):
        bind_amrvac_v5_forest(
            index._replace(domain_cell_counts=np.asarray([5, 2, 2], dtype=np.int64))
        )
    with pytest.raises(OverflowError, match="root-grid volume"):
        bind_amrvac_v5_forest(
            index._replace(
                domain_cell_counts=np.asarray(
                    [2_147_483_647] * 3,
                    dtype=np.int64,
                ),
                block_cell_counts=np.ones(3, dtype=np.int64),
            )
        )


def test_malformed_forest_and_declared_maximum_are_rejected() -> None:
    index = _mixed_index()
    with pytest.raises(ValueError, match="trailing nodes"):
        bind_amrvac_v5_forest(
            index._replace(forest_flags=np.ones(10, dtype=np.bool_))
        )
    with pytest.raises(ValueError, match="exceeds declared_max_level"):
        bind_amrvac_v5_forest(index._replace(declared_max_level=1))


def test_reordered_or_corrupted_disk_leaf_rows_are_rejected() -> None:
    index = _mixed_index()
    levels = index.block_levels.copy()
    levels[[0, 8]] = levels[[8, 0]]
    with pytest.raises(ValueError, match="level.*leaf 0"):
        bind_amrvac_v5_forest(index._replace(block_levels=levels))

    coordinates = index.block_coordinates.copy()
    coordinates[[0, 1]] = coordinates[[1, 0]]
    with pytest.raises(ValueError, match="coordinate.*leaf 0, axis 0"):
        bind_amrvac_v5_forest(index._replace(block_coordinates=coordinates))


@pytest.mark.parametrize(
    "relative_path",
    ["data/tdm.dat", "data/weno509_sub_0000.dat", "reference/bw.dat"],
)
def test_available_real_tree_binds_exactly(relative_path: str) -> None:
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"optional real fixture is unavailable: {relative_path}")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(descriptor)
    finally:
        os.close(descriptor)
    binding = bind_amrvac_v5_forest(index)
    forest = binding.forest

    assert binding.source_file_identity == index.file_identity
    assert forest.max_level <= index.declared_max_level
    assert forest.leaf_node_ids.size == index.leaf_count
    assert np.array_equal(
        forest.node_levels[forest.leaf_node_ids],
        index.block_levels,
    )
    assert np.array_equal(
        forest.node_coords[forest.leaf_node_ids],
        index.block_coordinates,
    )
