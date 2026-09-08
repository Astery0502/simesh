"""Experimental native analysis; providers are explicit, immutable inputs.

Canonical ``simesh.amrvac`` workflows remain unchanged. See
``docs/analysis-core/native-core-design.md`` for supported scope.
"""

from .fields import FieldDefinition, FieldSource, PreparedFields, PreparedPool, prepare, iter_prepared
from .mesh import MeshIndex
from .sampling import sample
from .derivatives import derivative, curl
from .field_lines import trace, iter_traces, retrace, TraceResult, Termination
from .diagnostics import with_curl, CurlPool
from .global_fields import global_curl
from .slices import Plane, SliceResult, sample_plane, iter_uniform
from .los import integrate_los, integrate_los_views, orthographic_plane, LOSResult, LOSStatus

__all__ = ["FieldDefinition", "FieldSource", "MeshIndex", "PreparedFields",
           "PreparedPool", "prepare", "iter_prepared", "sample", "derivative", "curl",
           "trace", "iter_traces", "TraceResult", "Termination", "global_curl",
           "Plane", "SliceResult", "sample_plane", "with_curl", "CurlPool", "retrace",
           "integrate_los", "integrate_los_views", "orthographic_plane", "LOSResult", "LOSStatus", "iter_uniform"]


def open_source(path,*,field_names=None,field_indices=None,field_units=None,
                support_capacity=128,value_cache_capacity=0,budget_bytes=2*1024**3):
    """Lazily open an owned immutable v5 ordinary-field source context."""
    from simesh.amrvac.analysis_io import open_source as implementation
    return implementation(path,field_names=field_names,field_indices=field_indices,
                          field_units=field_units,support_capacity=support_capacity,
                          value_cache_capacity=value_cache_capacity,budget_bytes=budget_bytes)


def open_prepared(path,*,field_names=None,field_indices=None,field_units=None,budget_bytes=2*1024**3):
    """Read selected fields into an owned full-domain canonical prepared product."""
    from simesh.amrvac.analysis_io import open_prepared as implementation
    return implementation(path,field_names=field_names,field_indices=field_indices,
                          field_units=field_units,budget_bytes=budget_bytes)


def global_curl_file(path,*,field_names=('b1','b2','b3'),field_units=None,
                     backend='process',workers=2,task_size=512,batch_size=256,
                     support_capacity=256,output=None,budget_bytes=2*1024**3):
    """Complete file-backed curl through bounded independent preparation tasks."""
    from simesh.amrvac.analysis_execution import global_curl_file as implementation
    return implementation(path,field_names=field_names,field_units=field_units,backend=backend,
        workers=workers,task_size=task_size,batch_size=batch_size,support_capacity=support_capacity,
        output=output,budget_bytes=budget_bytes)


__all__ += ["open_source","open_prepared","global_curl_file"]
