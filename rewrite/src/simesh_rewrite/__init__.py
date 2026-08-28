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
from .refined_geometry import (
    fill_refined_leaf_geometry,
    refined_leaf_geometry,
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
from .reductions import (
    accumulate_field_sum,
    finalize_field_sum,
    merge_field_sums,
)
from .pipeline import execute_level1_m0, execute_level1_m0_from_blocks

__all__ = [
    "AccessPattern",
    "accumulate_field_sum",
    "AXIS_NAMES",
    "BoundaryMode",
    "BlockReader",
    "BlockWriter",
    "INDEX_DTYPE",
    "PAYLOAD_DTYPE",
    "PHYSICAL_BOUNDARY_ID",
    "RefinedForest",
    "copy_region_into",
    "array_block_reader",
    "array_block_writer",
    "apply_level1_same_level_halo_plan",
    "execute_level1_m0",
    "execute_level1_m0_from_blocks",
    "fill_balanced_refined_relations",
    "fill_ascending_primary_prefix",
    "common_physical_valid_region",
    "central_difference_into",
    "fill_level1_face_neighbors",
    "fill_level1_block_geometry",
    "fill_level1_morton",
    "fill_level1_halo_relation_plan",
    "fill_refined_forest",
    "fill_refined_leaf_geometry",
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
    "refined_contact_targets",
    "balanced_refined_relations",
    "level1_block_geometry",
    "minimum_face_closed_slots",
    "minimum_direct_face_closed_slots",
    "minimum_halo_closed_slots",
    "minimum_level1_halo_closed_slots",
    "maximum_balanced_refined_support_slots",
    "make_block_reader",
    "make_block_writer",
    "merge_field_sums",
    "plan_level1_chunk",
    "plan_direct_face_closed_prefix",
    "plan_level1_halo_chunk",
    "plan_level1_halo_closed_prefix",
    "plan_balanced_refined_support_prefix",
    "place_level1_blocks",
    "physical_halo_source_index",
    "ravel_cell",
    "required_input_region",
    "read_blocks_into",
    "sample_level1_zero_order",
    "sample_level1_trilinear",
    "scaled_difference_into",
    "scatter_blocks_from",
    "supports_output_region",
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
    "workspace_nbytes",
    "workspace_slot_capacity",
    "write_blocks_from",
]
