from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Mesh:
    """Base mesh class that defines common attributes for all mesh types."""
    
    # Spatial range of the mesh
    xrange: Tuple[float, float]  # (min_x, max_x)
    yrange: Tuple[float, float]  # (min_y, max_y)
    zrange: Tuple[float, float]  # (min_z, max_z)
    
    # Field names that can be defined on the mesh
    field_names: List[str]

    def __post_init__(self):
        """Validate ranges and initialize empty field names if none provided."""
        # Validate ranges
        for range_name, range_val in [
            ('x_range', self.xrange),
            ('y_range', self.yrange),
            ('z_range', self.zrange) if self.zrange is not None else (None, None)
        ]:
            if range_val and range_val[0] >= range_val[1]:
                raise ValueError(f"{range_name} minimum must be less than maximum")
        
        # Initialize empty field names list if none provided
        if self.field_names is None:
            self.field_names = []
