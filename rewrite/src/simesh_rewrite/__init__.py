"""Functional AMR rewrite core."""

from .access import (
    AccessPattern,
    required_input_region,
    supports_output_region,
    valid_output_region,
    validate_access_requirement,
)
from .foundation import (
    AXIS_NAMES,
    INDEX_DTYPE,
    PAYLOAD_DTYPE,
    copy_region_into,
    interior_region,
    ravel_cell,
    unravel_cell,
)
from .morton import fill_level1_morton, level1_morton
from .geometry import fill_level1_block_geometry, level1_block_geometry
from .topology import (
    PHYSICAL_BOUNDARY_ID,
    fill_level1_face_neighbors,
    level1_face_neighbors,
)
from .forest import RefinedForest, fill_refined_forest, refined_forest
from .forest_conformance import validate_refined_forest_arrays
from .contacts import fill_refined_contact_targets, refined_contact_targets
from .balance import validate_refined_all_touch_2to1
from .relations import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    balanced_refined_relations,
    fill_balanced_refined_relations,
)
from .refined_support import (
    maximum_balanced_refined_support_slots,
    plan_balanced_refined_support_prefix,
)
from .selected_refined_support import (
    maximum_selected_refined_support_slots,
    plan_selected_refined_support_prefix,
)
from .restriction import restrict_cartesian_2to1_into
from .limiter import three_point_limited_slope
from .prolongation import prolong_cartesian_2to1_into
from .relation_slots import resolve_refined_relation_source_slots
from .target_boxes import fill_directed_halo_target_boxes
from .relation_phases import fill_refined_relation_phase_codes
from .same_level_boxes import fill_same_level_source_boxes
from .finer_boxes import fill_finer_restriction_boxes
from .coarser_workspace import fill_coarser_workspace_boxes
from .coarser_support import fill_coarser_slope_support_plan
from .coarser_workspace_application import apply_coarser_workspace_plan
from .refined_halo import (
    RefinedHaloExecutionStats,
    execute_selected_refined_halos_from_blocks,
)
from .completed_primary import (
    CompletedPrimaryConsumer,
    CompletedPrimaryExecutionStats,
    execute_selected_refined_halos_with_consumer,
    make_completed_primary_consumer,
)
from .completed_halo_sampling import (
    CachedVectorSamplingStats,
    CompletedHaloSamplingSession,
    clear_completed_halo_sampling_session,
    make_completed_halo_sampling_session,
    sample_refined_trilinear_vectors_cached,
)
from .refined_geometry import (
    fill_refined_leaf_geometry,
    refined_leaf_geometry,
)
from .region_selection import (
    RefinedRegionSelection,
    count_refined_region_windows,
    fill_refined_region_windows,
    refined_region_windows,
)
from .point_location import (
    fill_refined_point_leaf_ids,
    refined_point_leaf_ids,
)
from .hinted_location import (
    HintedLocationStats,
    fill_refined_point_leaf_ids_with_hints,
)
from .refined_sampling import (
    sample_refined_trilinear_point_groups,
    sample_refined_zero_order_point_groups,
)
from .repeated_sampling import (
    RepeatedPointExecutionStats,
    execute_refined_trilinear_points_from_blocks,
    execute_refined_zero_order_points_from_blocks,
)
from .storage import gather_blocks_into, scatter_blocks_from
from .blockio import (
    BlockReader,
    BlockWriter,
    array_block_reader,
    array_block_writer,
    make_block_reader,
    make_block_writer,
    read_blocks_into,
    write_blocks_from,
)
from .amrvac_dat import (
    AMRVACV5ForestBinding,
    AMRVACV5Index,
    bind_amrvac_v5_forest,
    read_amrvac_v5_index,
)
from .amrvac_dat_reader import make_amrvac_v5_block_reader
from .chunking import (
    minimum_face_closed_slots,
    minimum_halo_closed_slots,
    plan_level1_chunk,
    plan_level1_halo_chunk,
)
from .workspace import workspace_nbytes, workspace_slot_capacity
from .primary import fill_ascending_primary_prefix
from .face_closure import (
    minimum_direct_face_closed_slots,
    plan_direct_face_closed_prefix,
)
from .halo_closure import (
    minimum_level1_halo_closed_slots,
    plan_level1_halo_closed_prefix,
)
from .halos import (
    BoundaryMode,
    common_physical_valid_region,
    fill_physical_halos,
    fill_same_level_halos,
)
from .boundary_rules import (
    physical_halo_source_index,
    transform_physical_halo_value,
)
from .physical_widening import apply_cartesian_physical_widening
from .halo_plans import (
    fill_level1_halo_relation_plan,
    level1_halo_relation_plan,
)
from .halo_apply import apply_level1_same_level_halo_plan
from .sampling import (
    place_level1_blocks,
    sample_level1_trilinear,
    sample_level1_zero_order,
)
from .operators import central_difference_into, scaled_difference_into
from .curl import cartesian_curl_into
from .reductions import (
    accumulate_field_sum,
    finalize_field_sum,
    merge_field_sums,
)
from .local_field import (
    SelectedCurlExecutionStats,
    execute_selected_refined_curl_from_blocks,
)
from .pipeline import execute_level1_m0, execute_level1_m0_from_blocks

__all__ = [
    "AccessPattern",
    "accumulate_field_sum",
    "AMRVACV5ForestBinding",
    "AMRVACV5Index",
    "AXIS_NAMES",
    "BoundaryMode",
    "BlockReader",
    "BlockWriter",
    "CachedVectorSamplingStats",
    "CompletedHaloSamplingSession",
    "CompletedPrimaryConsumer",
    "CompletedPrimaryExecutionStats",
    "HintedLocationStats",
    "INDEX_DTYPE",
    "PAYLOAD_DTYPE",
    "PHYSICAL_BOUNDARY_ID",
    "RefinedForest",
    "RefinedRegionSelection",
    "SelectedCurlExecutionStats",
    "copy_region_into",
    "array_block_reader",
    "array_block_writer",
    "apply_level1_same_level_halo_plan",
    "apply_cartesian_physical_widening",
    "apply_coarser_workspace_plan",
    "bind_amrvac_v5_forest",
    "cartesian_curl_into",
    "clear_completed_halo_sampling_session",
    "count_refined_region_windows",
    "execute_selected_refined_curl_from_blocks",
    "execute_selected_refined_halos_from_blocks",
    "execute_selected_refined_halos_with_consumer",
    "execute_refined_trilinear_points_from_blocks",
    "execute_refined_zero_order_points_from_blocks",
    "execute_level1_m0",
    "execute_level1_m0_from_blocks",
    "fill_balanced_refined_relations",
    "fill_directed_halo_target_boxes",
    "fill_refined_relation_phase_codes",
    "fill_same_level_source_boxes",
    "fill_finer_restriction_boxes",
    "fill_coarser_workspace_boxes",
    "fill_coarser_slope_support_plan",
    "fill_ascending_primary_prefix",
    "common_physical_valid_region",
    "central_difference_into",
    "fill_level1_face_neighbors",
    "fill_level1_block_geometry",
    "fill_level1_morton",
    "fill_level1_halo_relation_plan",
    "fill_refined_forest",
    "fill_refined_leaf_geometry",
    "fill_refined_point_leaf_ids",
    "fill_refined_point_leaf_ids_with_hints",
    "fill_refined_region_windows",
    "fill_refined_contact_targets",
    "fill_physical_halos",
    "fill_same_level_halos",
    "finalize_field_sum",
    "gather_blocks_into",
    "interior_region",
    "level1_morton",
    "level1_halo_relation_plan",
    "level1_face_neighbors",
    "refined_forest",
    "refined_leaf_geometry",
    "refined_point_leaf_ids",
    "refined_contact_targets",
    "refined_region_windows",
    "balanced_refined_relations",
    "level1_block_geometry",
    "minimum_face_closed_slots",
    "minimum_direct_face_closed_slots",
    "minimum_halo_closed_slots",
    "minimum_level1_halo_closed_slots",
    "maximum_balanced_refined_support_slots",
    "maximum_selected_refined_support_slots",
    "make_block_reader",
    "make_block_writer",
    "make_completed_halo_sampling_session",
    "make_completed_primary_consumer",
    "make_amrvac_v5_block_reader",
    "merge_field_sums",
    "plan_level1_chunk",
    "plan_direct_face_closed_prefix",
    "plan_level1_halo_chunk",
    "plan_level1_halo_closed_prefix",
    "plan_balanced_refined_support_prefix",
    "plan_selected_refined_support_prefix",
    "place_level1_blocks",
    "physical_halo_source_index",
    "prolong_cartesian_2to1_into",
    "ravel_cell",
    "required_input_region",
    "read_blocks_into",
    "read_amrvac_v5_index",
    "restrict_cartesian_2to1_into",
    "resolve_refined_relation_source_slots",
    "sample_level1_zero_order",
    "sample_level1_trilinear",
    "sample_refined_zero_order_point_groups",
    "sample_refined_trilinear_point_groups",
    "sample_refined_trilinear_vectors_cached",
    "scaled_difference_into",
    "scatter_blocks_from",
    "supports_output_region",
    "three_point_limited_slope",
    "transform_physical_halo_value",
    "unravel_cell",
    "valid_output_region",
    "validate_access_requirement",
    "validate_refined_all_touch_2to1",
    "validate_refined_forest_arrays",
    "RELATION_PHYSICAL",
    "RELATION_COARSER",
    "RELATION_SAME",
    "RELATION_FINER",
    "RefinedHaloExecutionStats",
    "RepeatedPointExecutionStats",
    "workspace_nbytes",
    "workspace_slot_capacity",
    "write_blocks_from",
]
