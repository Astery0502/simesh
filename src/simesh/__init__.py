"""
Simesh: a Python/Cython toolkit for AMRVAC-style AMR data.

The package combines Python user interfaces with compiled AMR data structures
for reading, writing, exploring, and manipulating block-structured adaptive
mesh refinement datasets.
"""

__version__ = "0.1.0"
__author__ = "Hao Wu"
__license__ = "GPL-3.0"

# Version information tuple
VERSION_INFO = tuple(map(int, __version__.split(".")))

# Expose main functionality at package level
# from .utils import configurations
# from .frontends.amrvac.io import amr_loader, load_from_uarrays, header_template
# from .geometry.amr.amr_forest import AMRForest
# from .dataset.data_set import AMRDataSet

# # Define what should be available in "from simesh import *"
# __all__ = [
#     "header_template",
#     'configurations',
#     'amr_loader',
#     'load_from_uarrays',
#     'AMRForest',
#     'AMRDataSet',
# ]
