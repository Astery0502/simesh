"""Detached, immutable snapshot descriptions without numerical or I/O state."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from types import MappingProxyType

import numpy as np


_INDEX_HEADER = {
    "offset_tree": "offset_tree", "offset_blocks": "offset_blocks",
    "nw": "field_count", "ndir": "direction_count", "ndim": "dimension_count",
    "levmax": "declared_max_level", "nleafs": "leaf_count", "nparents": "parent_count",
    "it": "iteration", "time": "time", "xmin": "domain_lower", "xmax": "domain_upper",
    "domain_nx": "domain_cell_counts", "block_nx": "block_cell_counts",
    "periodic": "periodic", "geometry": "geometry", "staggered": "staggered",
    "w_names": "field_names", "physics_type": "physics_type",
    "params": "parameter_values", "param_names": "parameter_names",
    "snapshotnext": "snapshot_next", "slicenext": "slice_next", "collapsenext": "collapse_next",
}
_ARRAY_DTYPES = {
    "xmin": float, "xmax": float, "domain_nx": np.int64,
    "block_nx": np.int64, "periodic": bool, "params": float,
}


def _freeze(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("snapshot header values must be scalars or sequences of scalars")


@dataclass(frozen=True)
class SnapshotMetadata:
    """A copied AMRVAC header and optional source-file observations.

    Header sequences are immutable tuples. This describes the original snapshot,
    not the selected or derived field directory. It infers no units or EOS.
    ``file_identity`` records device, inode, size, mtime_ns and ctime_ns when read;
    it is not a content digest or a verified association with any later Fields.
    """

    header: Mapping
    path: str | None = None
    byte_order: str | None = None
    file_identity: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.header, Mapping) or not all(isinstance(k, str) for k in self.header):
            raise TypeError("snapshot header must be a string-keyed mapping")
        header = {key: _freeze(value) for key, value in self.header.items()}
        required = {"datfile_version", *_INDEX_HEADER, "n_par"}
        if not required <= header.keys():
            raise ValueError(f"snapshot header is missing {sorted(required - header.keys())}")
        if (len(header["w_names"]) != header["nw"] or
                len(header["param_names"]) != len(header["params"]) or
                len(header["params"]) != header["n_par"]):
            raise ValueError("snapshot field and parameter counts must match their directories")
        if self.byte_order not in (None, "<", ">", "="):
            raise ValueError("byte_order must be '<', '>' or '=' when supplied")
        identity = None if self.file_identity is None else tuple(self.file_identity)
        if identity is not None:
            if len(identity) != 5 or any(isinstance(x, (bool, np.bool_)) or
                                       not isinstance(x, (int, np.integer)) for x in identity):
                raise ValueError("file_identity must contain five integers")
            identity = tuple(map(int, identity))
        object.__setattr__(self, "header", MappingProxyType(header))
        object.__setattr__(self, "file_identity", identity)
        if self.path is not None:
            object.__setattr__(self, "path", os.path.abspath(os.fsdecode(self.path)))

    @classmethod
    def _from_amrvac_index(cls, index, path):
        header = {key: getattr(index, attribute) for key, attribute in _INDEX_HEADER.items()}
        header.update(datfile_version=5, n_par=len(index.parameter_names))
        return cls(header, path=path, byte_order=index.byte_order, file_identity=index.file_identity)

    @property
    def time(self):
        return self.header["time"]

    @property
    def iteration(self):
        return self.header["it"]

    @property
    def physics_type(self):
        return self.header["physics_type"]

    @property
    def field_names(self):
        return self.header["w_names"]

    @property
    def parameters(self):
        names = self.header["param_names"]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter names; use the ordered header parameter arrays")
        return MappingProxyType(dict(zip(names, self.header["params"])))

    def to_header(self):
        """Return an independent mutable header for the existing AMRVAC writer."""
        header = dict(self.header)
        for key, dtype in _ARRAY_DTYPES.items():
            header[key] = np.array(header[key], dtype=dtype)
        for key in ("w_names", "param_names"):
            header[key] = list(header[key])
        return header

    def to_dict(self):
        """Return detached JSON data for result metadata; reject nonfinite values.

        Header time and parameters remain in their recorded units. Source-file
        observations are descriptive, not a persisted value-identity certificate.
        """
        return json.loads(json.dumps({
            "format": "amrvac", "header": dict(self.header), "path": self.path,
            "byte_order": self.byte_order, "file_identity": self.file_identity,
        }, allow_nan=False))
