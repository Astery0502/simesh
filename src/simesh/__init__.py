"""Public entry points for native Cartesian AMR analysis.

Use Source and prepare for explicit input lifetimes, Fields for completed data,
and numerical consumers for analysis. The applications module associates results
with identified geometry. Public imports remain available here; implementations
import their dependencies directly from the defining modules.
"""

from .fields import FieldDefinition, Fields
from .mesh import Mesh, Selection, mesh_from_forest, select_region, select_roots
from .io.source import Source, source_from_arrays, read_fields, select_source
from .io.amrvac import open_amrvac
from .io.metadata import SnapshotMetadata
from .io.products import write_amrvac, crop_amrvac
from .io.cache import cache_source
from .io.uniform import export_uniform, export_uniform_vtk
from .io.vtk import write_uniform_vtk
from .io.jobs import global_curl_file
from .preparation import prepare
from .preparation.plans import FillPlan, plan_preparation
from .bounded import iter_prepared, global_curl
from .operators.sampling import sample
from .operators.derivatives import derivative, curl
from .operators.derived import derive, derive_many, DerivedContext
from .field_ops import select_fields, merge_fields
from .connectivity import qsl, iter_qsl, line_diagnostics, iter_line_diagnostics, QSLResult, Boundary, ConnectivityTermination
from .diagnostics import (magnitude, dot, gradient, divergence,
                          current_density, magnetic_pressure, magnetic_energy_density)
from .slices import SliceResult, sample_plane, iter_uniform, AMRSliceResult, slice_axis
from .tracing import Termination, TraceResult, trace, iter_traces, retrace
from .projection import LOSResult, LOSStatus, integrate_los, integrate_los_views, orthographic_plane
from .physics.composition import CoronalComposition
from .physics.emission import AIA171, EUV
from .physics.thermodynamics import thermal_fields, emissivity_fields
from .physics.thermal import ThermalLOSResult, integrate_thermal_los
from .physics.radiation import HHeAbsorption, RadioFreeFree, radiation_fields
from .spatial import Plane, PointSet, RaySet, LineSet, LengthUnits, AxisAlignedSurface
from .geometry import AxisSlice, native_bottom_seeds
from .line_profiles import LineProfile, LineProfiles, sample_line_profiles, iter_line_profiles
from .reductions import (volume_integral,
                         weighted_mean, extrema, histogram, surface_flux)
from .physics.units import MagneticUnits, MHDUnits
from .physics.mhd import IdealMHD, MHDStatus, MHDStateError, mhd_fields
from .results_io import ResultFile, ResultFileError, save_result, load_result
from .result_shards import ResultShards, save_result_shards, open_result_shards
from .applications import RadiationResult, radiative_los
from .current_proxy import (current_proxy, iter_current_proxy, current_proxy_from_lines,
                            CurrentProxyDiagnostics, CurrentProxyBatch, CurrentProxyResult)
from .operators.line import line_integral, line_derivative

__version__ = "0.2.0rc1"

__all__ = [
    'AIA171', 'AMRSliceResult', 'AxisAlignedSurface', 'AxisSlice', 'Boundary',
    'ConnectivityTermination', 'CoronalComposition', 'CurrentProxyBatch',
    'CurrentProxyDiagnostics', 'CurrentProxyResult', 'DerivedContext', 'EUV',
    'FieldDefinition', 'Fields', 'FillPlan', 'HHeAbsorption', 'IdealMHD', 'LOSResult',
    'LOSStatus', 'LengthUnits', 'LineProfile', 'LineProfiles', 'LineSet',
    'MHDStateError', 'MHDStatus', 'MHDUnits', 'MagneticUnits', 'Mesh', 'Plane',
    'PointSet', 'QSLResult', 'RadiationResult', 'RadioFreeFree', 'RaySet',
    'ResultFile', 'ResultFileError', 'ResultShards', 'Selection', 'SliceResult',
    'SnapshotMetadata', 'Source', 'Termination', 'ThermalLOSResult', 'TraceResult',
    'cache_source', 'crop_amrvac', 'curl', 'current_density', 'current_proxy',
    'current_proxy_from_lines', 'derivative', 'derive', 'derive_many', 'divergence',
    'dot', 'emissivity_fields', 'export_uniform', 'export_uniform_vtk', 'extrema',
    'global_curl', 'global_curl_file', 'gradient', 'histogram', 'integrate_los',
    'integrate_los_views', 'integrate_thermal_los', 'iter_current_proxy',
    'iter_line_diagnostics', 'iter_line_profiles', 'iter_prepared', 'iter_qsl',
    'iter_traces', 'iter_uniform', 'line_derivative', 'line_diagnostics',
    'line_integral', 'load_result', 'magnetic_energy_density', 'magnetic_pressure',
    'magnitude', 'merge_fields', 'mesh_from_forest', 'mhd_fields',
    'native_bottom_seeds', 'open_amrvac', 'open_result_shards', 'orthographic_plane',
    'plan_preparation', 'prepare', 'qsl', 'radiation_fields', 'radiative_los',
    'read_fields', 'retrace', 'sample', 'sample_line_profiles', 'sample_plane',
    'save_result', 'save_result_shards', 'select_fields', 'select_region',
    'select_roots', 'select_source', 'slice_axis', 'source_from_arrays',
    'surface_flux', 'thermal_fields', 'trace', 'volume_integral', 'weighted_mean',
    'write_amrvac', 'write_uniform_vtk',
]
