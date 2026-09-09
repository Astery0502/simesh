"""Independent experimental native AMR analysis.

The delivered profiles supply explicit sources, regional and whole-domain fields,
scientific consumers and opt-in bounded workflows. See the README for numerical
scope and MIGRATION.md for retained Dataset interfaces and moved imports.
"""

from .fields import FieldDefinition, Fields
from .mesh import Mesh, Selection, mesh_from_forest, select_region
from .io.source import Source, source_from_arrays, read_fields, select_source
from .io.amrvac import open_amrvac
from .io.metadata import SnapshotMetadata
from .io.products import source_from_dataset, write_amrvac
from .io.cache import cache_source
from .io.jobs import global_curl_file
from .preparation import prepare
from .preparation.plans import FillPlan, plan_preparation
from .bounded import iter_prepared, global_curl
from .operators.sampling import sample
from .operators.derivatives import derivative, curl
from .operators.derived import derive, derive_many, DerivedContext
from .field_ops import select_fields, merge_fields
from .connectivity import qsl, iter_qsl, line_diagnostics, iter_line_diagnostics, QSLResult, Boundary, ConnectivityTermination
from .diagnostics import (magnitude, dot, gradient, divergence, MagneticUnits,
                          current_density, magnetic_pressure, magnetic_energy_density)
from .slices import Plane, SliceResult, sample_plane, iter_uniform
from .tracing import Termination, TraceResult, trace, iter_traces, retrace
from .projection import LOSResult, LOSStatus, integrate_los, integrate_los_views, orthographic_plane
from .physics.thermal import AIA171, CoronalComposition, ThermalLOSResult, thermal_fields, emissivity_fields, integrate_thermal_los
from .geometry import PointSet, RaySet, LineSet
from .line_profiles import LineProfile, LineProfiles, sample_line_profiles, iter_line_profiles
from .reductions import (LengthUnits, AxisAlignedSurface, volume_integral,
                         weighted_mean, extrema, histogram, surface_flux)
from .physics.mhd import MHDUnits, IdealMHD, MHDStatus, MHDStateError, mhd_fields
from .results_io import ResultFile, ResultFileError, save_result, load_result

__all__ = ["FieldDefinition", "Fields", "Mesh", "Selection", "mesh_from_forest", "select_region",
           "Source", "source_from_arrays", "read_fields", "open_amrvac", "prepare", "iter_prepared",
           "sample", "derivative", "curl", "Plane", "SliceResult", "sample_plane",
           "Termination", "TraceResult", "trace", "iter_traces"]
__all__ += ["select_source", "cache_source", "FillPlan", "plan_preparation", "global_curl", "global_curl_file",
            "iter_uniform", "retrace", "LOSResult", "LOSStatus", "integrate_los", "integrate_los_views",
            "orthographic_plane", "AIA171", "CoronalComposition", "ThermalLOSResult", "thermal_fields",
            "emissivity_fields", "integrate_thermal_los"]
__all__ += ["SnapshotMetadata", "source_from_dataset", "write_amrvac"]
__all__ += ["derive", "derive_many", "DerivedContext", "select_fields", "merge_fields"]
__all__ += ["qsl", "iter_qsl", "QSLResult", "Boundary", "ConnectivityTermination"]
__all__ += ["PointSet", "RaySet", "LineSet"]
__all__ += ["LineProfile", "LineProfiles", "sample_line_profiles"]
__all__ += ["line_diagnostics","iter_line_diagnostics","magnitude","dot","gradient","divergence",
            "MagneticUnits","current_density","magnetic_pressure","magnetic_energy_density"]
__all__ += ["LengthUnits","AxisAlignedSurface","volume_integral","weighted_mean","extrema","histogram","surface_flux",
            "MHDUnits","IdealMHD","MHDStatus","MHDStateError","mhd_fields",
            "ResultFile","ResultFileError","save_result","load_result"]
__version__ = "0.2.0.dev0"

from .result_shards import ResultShards, save_result_shards, open_result_shards
__all__ += ["iter_line_profiles", "ResultShards", "save_result_shards", "open_result_shards"]
