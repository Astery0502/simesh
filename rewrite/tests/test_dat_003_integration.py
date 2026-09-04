from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from simesh.amrvac.datio import (
    header_template,
    read_blocks_sequential,
    write_datfile_from_sfc,
)
from simesh_rewrite.amrvac_dat import (
    bind_amrvac_v5_forest,
    read_amrvac_v5_index,
)
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import (
    array_block_reader,
    array_block_writer,
    read_blocks_into,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.refined_halo import execute_selected_refined_halos_from_blocks
from simesh_rewrite.repeated_sampling import (
    execute_refined_trilinear_points_from_blocks,
    execute_refined_zero_order_points_from_blocks,
)
from simesh_rewrite.storage import gather_blocks_into


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def refined_file_components():
    root_shape = i3(4, 4, 4)
    block_shape = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        split = tuple(int(value) for value in coordinate) == (1, 1, 1)
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    flag_array = np.asarray(flags, dtype=np.bool_)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flag_array,
    )
    leaf_nodes = forest.leaf_node_ids
    block_levels = forest.node_levels[leaf_nodes].astype(np.int32)
    block_coordinates = (forest.node_coords[leaf_nodes] + 1).astype(np.int32)
    leaf_count = int(leaf_nodes.size)
    field_count = 3
    x, y, z = np.indices((4, 4, 4), dtype=np.float64)
    data = np.empty((leaf_count, field_count, 4, 4, 4), dtype=np.float64)
    for leaf in range(leaf_count):
        for field in range(field_count):
            data[leaf, field] = (
                leaf * 100000.0
                + field * 10000.0
                + x * 100.0
                + y * 10.0
                + z
                + 1.0
            )

    header = header_template.copy()
    header.update(
        datfile_version=5,
        nw=field_count,
        ndir=3,
        ndim=3,
        levmax=forest.max_level,
        nleafs=leaf_count,
        nparents=int(forest.node_levels.size - leaf_count),
        xmin=np.zeros(3, dtype=np.float64),
        xmax=root_shape.astype(np.float64),
        domain_nx=(root_shape * block_shape).astype(np.int32),
        block_nx=block_shape.astype(np.int32),
        periodic=np.zeros(3, dtype=np.bool_),
        geometry="Cartesian_3D",
        staggered=False,
        w_names=["q0", "q1", "q2"],
    )
    tree = (
        block_levels,
        block_coordinates,
        np.zeros(leaf_count, dtype=np.int64),
    )
    return data, header, flag_array.astype(np.int32), tree


def write_refined_file(path: Path) -> np.ndarray:
    data, header, flags, tree = refined_file_components()
    write_datfile_from_sfc(
        str(path), data, header, flags, tree, overwrite=True
    )
    return data


def assert_bits_equal(left: np.ndarray, right: np.ndarray) -> None:
    assert np.array_equal(left.view(np.uint64), right.view(np.uint64))


def test_real_tdm_selected_fields_match_current_eager_reader() -> None:
    path = REPOSITORY_ROOT / "data/tdm.dat"
    if not path.exists():
        pytest.skip("real non-staggered tdm.dat fixture is unavailable")
    field_ids = i3(6, 4, 6)
    block_ids = i3(26, 0, 13, 0)
    current = read_blocks_sequential(str(path), field_ids.tolist())
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.lseek(file_descriptor, 17, os.SEEK_SET)
        index = read_amrvac_v5_index(file_descriptor)
        binding = bind_amrvac_v5_forest(index)
        reader = make_amrvac_v5_block_reader(
            file_descriptor, index, binding
        )
        destination = np.empty((4, 3, 10, 10, 10), dtype=np.float64)
        read_blocks_into(
            reader,
            i3(0, 0, 0),
            i3(10, 10, 10),
            block_ids,
            field_ids,
            destination,
            i3(0, 0, 0),
        )
        assert os.lseek(file_descriptor, 0, os.SEEK_CUR) == 17
    finally:
        os.close(file_descriptor)
    assert_bits_equal(destination, np.ascontiguousarray(current[block_ids]))


def test_native_refined_reader_matches_array_rhe_and_repeated_sampling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "refined-v5.dat"
    backing = write_refined_file(path)
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(file_descriptor)
        binding = bind_amrvac_v5_forest(index)
        reader = make_amrvac_v5_block_reader(
            file_descriptor, index, binding
        )
        forest = binding.forest
        validate_refined_all_touch_2to1(
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
        )
        leaf_count = index.leaf_count
        all_leaf_ids = np.arange(leaf_count, dtype=np.int64)
        fields = i3(2, 0, 2)
        modes = np.zeros((fields.size, 6), dtype=np.uint8)
        normals = i3(-1, -1, -1)
        one = i3(1, 1, 1)

        # Direct translated/partial selected reads keep STO-003 semantics.
        selected_blocks = i3(70, 0, 70, 5)
        source_lower = i3(1, 0, 1)
        source_upper = i3(4, 3, 4)
        destination_lower = i3(1, 2, 3)
        actual_direct = np.full((4, 3, 6, 7, 8), -97.0)
        expected_direct = actual_direct.copy()
        read_blocks_into(
            reader,
            source_lower,
            source_upper,
            selected_blocks,
            fields,
            actual_direct,
            destination_lower,
        )
        gather_blocks_into(
            backing,
            source_lower,
            source_upper,
            selected_blocks,
            fields,
            expected_direct,
            destination_lower,
        )
        assert_bits_equal(actual_direct, expected_direct)

        padded_shape = (6, 6, 6)
        native_halos = np.empty(
            (leaf_count, fields.size, *padded_shape), dtype=np.float64
        )
        array_halos = np.empty_like(native_halos)
        native_halo_stats = execute_selected_refined_halos_from_blocks(
            reader,
            array_block_writer(native_halos),
            all_leaf_ids,
            fields,
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            one,
            one,
            modes,
            normals,
            57,
        )
        array_halo_stats = execute_selected_refined_halos_from_blocks(
            array_block_reader(backing),
            array_block_writer(array_halos),
            all_leaf_ids,
            fields,
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            one,
            one,
            modes,
            normals,
            leaf_count,
        )
        assert native_halo_stats.chunk_count == 2
        assert array_halo_stats.chunk_count == 1
        assert_bits_equal(native_halos, array_halos)

        bounds, spacing = refined_leaf_geometry(
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            all_leaf_ids,
        )
        fractions = np.asarray(
            ((0.25, 0.75, 3.75), (3.75, 0.25, 0.75)),
            dtype=np.float64,
        )
        points = np.empty((leaf_count + 11, 3), dtype=np.float64)
        for leaf in range(leaf_count):
            points[leaf] = bounds[leaf, 0] + fractions[leaf % 2] * spacing[leaf]
        points[leaf_count : leaf_count + 10] = points[:10]
        points[-1] = (-1.0, 0.0, 0.0)

        native_zero = np.full((points.shape[0], fields.size), -101.0)
        array_zero = native_zero.copy()
        execute_refined_zero_order_points_from_blocks(
            reader,
            points,
            fields,
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            8,
            native_zero,
        )
        execute_refined_zero_order_points_from_blocks(
            array_block_reader(backing),
            points,
            fields,
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            leaf_count,
            array_zero,
        )
        assert_bits_equal(native_zero, array_zero)

        native_tri = np.full_like(native_zero, -103.0)
        array_tri = native_tri.copy()
        execute_refined_trilinear_points_from_blocks(
            reader,
            points,
            fields,
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            modes,
            normals,
            57,
            native_tri,
        )
        execute_refined_trilinear_points_from_blocks(
            array_block_reader(backing),
            points,
            fields,
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            modes,
            normals,
            leaf_count,
            array_tri,
        )
        assert_bits_equal(native_tri, array_tri)
        assert np.all(native_tri[-1] == -103.0)
    finally:
        os.close(file_descriptor)
