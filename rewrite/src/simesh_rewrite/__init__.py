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
from .storage import gather_blocks_into, scatter_blocks_from
from .chunking import (
    minimum_face_closed_slots,
    minimum_halo_closed_slots,
    plan_level1_chunk,
    plan_level1_halo_chunk,
    workspace_nbytes,
    workspace_slot_capacity,
)
from .halos import (
    BoundaryMode,
    common_physical_valid_region,
    fill_physical_halos,
    fill_same_level_halos,
)
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

__all__ = [
    "AccessPattern",
    "accumulate_field_sum",
    "AXIS_NAMES",
    "BoundaryMode",
    "INDEX_DTYPE",
    "PAYLOAD_DTYPE",
    "PHYSICAL_BOUNDARY_ID",
    "copy_region_into",
    "common_physical_valid_region",
    "central_difference_into",
    "fill_level1_face_neighbors",
    "fill_level1_block_geometry",
    "fill_level1_morton",
    "fill_physical_halos",
    "fill_same_level_halos",
    "finalize_field_sum",
    "gather_blocks_into",
    "interior_region",
    "level1_morton",
    "level1_face_neighbors",
    "level1_block_geometry",
    "minimum_face_closed_slots",
    "minimum_halo_closed_slots",
    "merge_field_sums",
    "plan_level1_chunk",
    "plan_level1_halo_chunk",
    "place_level1_blocks",
    "ravel_cell",
    "required_input_region",
    "sample_level1_zero_order",
    "sample_level1_trilinear",
    "scaled_difference_into",
    "scatter_blocks_from",
    "supports_output_region",
    "unravel_cell",
    "valid_output_region",
    "validate_access_requirement",
    "workspace_nbytes",
    "workspace_slot_capacity",
]
