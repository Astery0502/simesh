"""
Simesh: A Python package for handling AMR (Adaptive Mesh Refinement) data structures.

This package provides tools for working with AMR meshes, particularly focused on
AMRVAC data handling and visualization.
"""

__version__ = "0.1.0"
__author__ = "Hao Wu"
__license__ = "GPL-3.0"

# Version information tuple
VERSION_INFO = tuple(map(int, __version__.split(".")))

# Expose main functionality at package level
from .frontends.amrvac.io import amr_loader, load_from_uarrays
from .geometry.amr.amr_forest import AMRForest
from .dataset.data_set import AMRDataSet

# Define what should be available in "from simesh import *"
__all__ = [
    'amr_loader',
    'load_from_uarrays',
    'AMRForest',
    'AMRDataSet'
]
