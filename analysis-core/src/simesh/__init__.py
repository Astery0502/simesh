"""Independent experimental native AMR analysis.

The delivered profiles supply explicit sources, regional and whole-domain fields,
scientific consumers and opt-in bounded workflows. See the README for numerical
scope and the remaining migration work.
"""

from .fields import FieldDefinition, Fields
from .mesh import Mesh, Selection, mesh_from_forest, select_region
from .io.source import Source, source_from_arrays, read_fields
from .io.amrvac import open_amrvac
from .io.products import source_from_dataset, write_amrvac
from .io.cache import cache_source
from .io.jobs import global_curl_file
from .preparation import prepare
from .preparation.plans import FillPlan, plan_preparation
from .bounded import iter_prepared, global_curl
from .operators.sampling import sample
from .operators.derivatives import derivative, curl
from .operators.derived import derive, DerivedContext
from .connectivity import qsl, iter_qsl, QSLResult, Boundary, ConnectivityTermination
from .slices import Plane, SliceResult, sample_plane, iter_uniform
from .tracing import Termination, TraceResult, trace, iter_traces, retrace
from .projection import LOSResult, LOSStatus, integrate_los, integrate_los_views, orthographic_plane
from .physics.thermal import AIA171, CoronalComposition, ThermalLOSResult, thermal_fields, emissivity_fields, integrate_thermal_los

__all__ = ["FieldDefinition", "Fields", "Mesh", "Selection", "mesh_from_forest", "select_region",
           "Source", "source_from_arrays", "read_fields", "open_amrvac", "prepare", "iter_prepared",
           "sample", "derivative", "curl", "Plane", "SliceResult", "sample_plane",
           "Termination", "TraceResult", "trace", "iter_traces"]
__all__ += ["cache_source", "FillPlan", "plan_preparation", "global_curl", "global_curl_file",
            "iter_uniform", "retrace", "LOSResult", "LOSStatus", "integrate_los", "integrate_los_views",
            "orthographic_plane", "AIA171", "CoronalComposition", "ThermalLOSResult", "thermal_fields",
            "emissivity_fields", "integrate_thermal_los"]
__all__ += ["source_from_dataset", "write_amrvac"]
__all__ += ["derive", "DerivedContext"]
__all__ += ["qsl", "iter_qsl", "QSLResult", "Boundary", "ConnectivityTermination"]
__version__ = "0.2.0.dev0"
