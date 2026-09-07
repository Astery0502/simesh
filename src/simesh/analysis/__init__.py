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
from .slices import Plane, SliceResult, sample_plane

__all__ = ["FieldDefinition", "FieldSource", "MeshIndex", "PreparedFields",
           "PreparedPool", "prepare", "iter_prepared", "sample", "derivative", "curl",
           "trace", "iter_traces", "TraceResult", "Termination", "global_curl",
           "Plane", "SliceResult", "sample_plane", "with_curl", "CurlPool", "retrace"]
