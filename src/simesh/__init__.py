"""
Simesh: A Python package for handling AMR (Adaptive Mesh Refinement) data structures.

This package provides tools for working with AMR meshes, particularly focused on
AMRVAC data handling and visualization.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"  # or whatever license you're using

# Version information tuple
VERSION_INFO = tuple(map(int, __version__.split(".")))

# Expose main functionality at package level
from .frontends.amrvac.io import amr_loader
from .geometry.amr.amr_forest import AMRForest
from .dataset.data_set import AMRDataSet

# Define what should be available in "from simesh import *"
__all__ = [
    'amr_loader',
    'AMRForest',
    'AMRDataSet'
]
