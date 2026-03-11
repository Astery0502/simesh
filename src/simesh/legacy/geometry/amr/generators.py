"""
Manually generate mesh structures, mesh values and datasets for AMR (Adaptive Mesh Refinement).

This module provides utilities and functions to create and manipulate AMR mesh structures,
including generation of:
- Base mesh grids
- Mesh value assignments
- Dataset structures for AMR hierarchies

The generators in this module allow manual creation and testing of AMR mesh configurations
without requiring a full simulation and data file.
"""

import numpy as np

class AMRGenerator:

    """
    Generators for AMR mesh structures, mesh values and datasets.
    """

    def __init__(self, ng1:int, ng2:int, ng3:int):

