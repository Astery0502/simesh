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

__all__ = [
    "AccessPattern",
    "AXIS_NAMES",
    "INDEX_DTYPE",
    "PAYLOAD_DTYPE",
    "copy_region_into",
    "interior_region",
    "ravel_cell",
    "required_input_region",
    "supports_output_region",
    "unravel_cell",
    "valid_output_region",
    "validate_access_requirement",
]
