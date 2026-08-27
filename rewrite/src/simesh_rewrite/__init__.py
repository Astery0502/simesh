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

__all__ = [
    "AccessPattern",
    "AXIS_NAMES",
    "INDEX_DTYPE",
    "PAYLOAD_DTYPE",
    "PHYSICAL_BOUNDARY_ID",
    "copy_region_into",
    "fill_level1_face_neighbors",
    "fill_level1_block_geometry",
    "fill_level1_morton",
    "interior_region",
    "level1_morton",
    "level1_face_neighbors",
    "level1_block_geometry",
    "ravel_cell",
    "required_input_region",
    "supports_output_region",
    "unravel_cell",
    "valid_output_region",
    "validate_access_requirement",
]
