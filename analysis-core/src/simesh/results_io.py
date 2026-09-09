"""Versioned, pickle-free files for owned application geometry and results.

Import explicitly from ``simesh.results_io``. A loaded ``ResultFile`` separates
caller metadata and unverified provenance from the reconstructed result.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile
import zlib

import numpy as np

from .applications import SampledPoints, ConnectivityMap, RayResult, UniformResult
from .connectivity import QSLResult, Boundary, ConnectivityTermination
from .fields import FieldDefinition
from .geometry import PointSet, RaySet, LineSet
from .projection import LOSStatus
from .slices import Plane
from .tracing import Termination

__all__ = ["save_result", "load_result", "ResultFile", "ResultFileError", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1
_FORMAT = "simesh-result"
_MANIFEST = "__manifest__"
_MAX_MANIFEST_BYTES = 1024 * 1024


class ResultFileError(ValueError):
    """An invalid or unsupported result file or result representation."""


@dataclass(frozen=True)
class ResultFile:
    """Reconstructed result, caller metadata, and explicitly unverified source.

    ``source`` is a caller-supplied JSON object or None. No file, digest, model,
    or in-memory source identity is verified by this module.
    """

    result: object
    metadata: dict
    source: dict | None
    schema_version: int = SCHEMA_VERSION

    @property
    def source_verification(self):
        return "unspecified" if self.source is None else "unverified"


# Closed schemas: no imports, callbacks, arbitrary attributes, or object arrays
# are selected by file content. Dtypes are canonical on disk (little endian).
_ARRAYS = {
    Plane: {"origin": "f8", "u": "f8", "v": "f8"},
    PointSet: {"positions": "f8", "ids": "i8", "normals": "f8"},
    RaySet: {"directions": "f8", "near": "f8", "far": "f8"},
    SampledPoints: {"values": "f8", "owners": "i8", "valid": "?"},
    ConnectivityMap: {},
    QSLResult: {**dict.fromkeys(("seeds", "q", "log10_q", "q_perp", "log10_q_perp",
                               "twist", "length", "footpoints", "endpoint_fields"), "f8"),
                **dict.fromkeys(("boundary", "termination", "steps"), "i8"),
                **dict.fromkeys(("complete", "valid", "stencil_valid"), "?")},
    RayResult: {**dict.fromkeys(("values", "entry", "exit"), "f8"),
                **dict.fromkeys(("status", "samples", "misses"), "i8")},
    LineSet: {"positions": "f8", "offsets": "i8", "termination": "i8"},
    UniformResult: {"values": "f8", "valid": "?", "lower": "f8", "upper": "f8"},
}
_CHILDREN = {
    PointSet: {"plane": Plane}, RaySet: {"origins": PointSet},
    SampledPoints: {"points": PointSet}, ConnectivityMap: {"points": PointSet, "data": QSLResult},
    RayResult: {"rays": RaySet}, LineSet: {"seeds": PointSet},
}
_PROPERTIES = {
    Plane: ("shape",), PointSet: ("shape",), SampledPoints: ("definitions",),
    QSLResult: ("normalization", "local_radius", "method"),
    RayResult: ("units", "quadrature", "metadata"), UniformResult: ("definitions",),
}
_OPTIONAL = {PointSet: {"normals"}, QSLResult: {"q", "log10_q", "q_perp", "log10_q_perp", "twist"}}
_IDENTIFIED = {SampledPoints, ConnectivityMap, RayResult, LineSet, UniformResult}
_TYPES = {cls.__name__: cls for cls in _ARRAYS}


def _require(condition, message):
    if not condition:
        raise ResultFileError(message)


def _json_value(value, depth=0):
    """Accept JSON values only; normalize tuple containers to JSON arrays."""
    _require(depth <= 64, "JSON metadata nesting exceeds 64 levels")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_json_value(item, depth+1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_value(item, depth+1) for key, item in value.items()}
    raise ResultFileError("metadata must contain only finite JSON values and string object keys")


def _json_object(value, label):
    _require(type(value) is dict, f"{label} must be a JSON object")
    return _json_value(value)


def _keys(value, keys, label):
    _require(type(value) is dict and value.keys() == set(keys), f"invalid {label} members")


def _shape(value, *, dimensions=None, positive=False):
    _require(type(value) in (list, tuple) and len(value) > 0 and
             all(type(n) is int and n >= int(positive) for n in value), "invalid layout shape")
    _require(dimensions is None or len(value) == dimensions, "invalid layout dimension")
    return tuple(value)


def _text(value, label):
    _require(type(value) is str and bool(value), f"{label} must be a nonempty string")


def _array_shape(array, shape, label, *, finite=False):
    _require(array is not None and array.shape == shape, f"invalid {label} shape; expected {shape}")
    if finite:
        _require(np.isfinite(array).all(), f"{label} must be finite")


def _coordinates(array, label):
    _require(array is not None and array.ndim == 2 and array.shape[1] == 3 and
             np.isfinite(array).all(), f"{label} must be finite (n,3) coordinates")
    return len(array)


def _unit_vectors(array, count, label):
    _require(array is not None and array.shape in ((3,), (count, 3)) and np.isfinite(array).all(),
             f"invalid {label} shape or values")
    _require(np.allclose(np.linalg.norm(array, axis=-1), 1., rtol=0, atol=8*np.finfo(float).eps),
             f"{label} must contain normalized vectors")


def _codes(array, enum, label, extra=()):
    _require(np.isin(array, [int(code) for code in enum] + list(extra)).all(), f"unknown {label} code")


def _definitions(value):
    _require(type(value) is list and bool(value), "definitions must be a nonempty list")
    definitions = []
    for item in value:
        _keys(item, ("name", "units", "interpretation"), "field definition")
        definitions.append(FieldDefinition(**item))
    return tuple(definitions)


def _validate(cls, data):
    """Validate structural and status invariants before constructing objects."""
    if cls is Plane:
        for name in ("origin", "u", "v"):
            _array_shape(data[name], (3,), name, finite=True)
        _shape(data["shape"], dimensions=2, positive=True)
    elif cls is PointSet:
        n = _coordinates(data["positions"], "positions")
        _array_shape(data["ids"], (n,), "IDs")
        _require(len(np.unique(data["ids"])) == n, "IDs must be unique")
        _require(math.prod(_shape(data["shape"])) == n, "shape does not match point count")
        if data["normals"] is not None:
            _unit_vectors(data["normals"], n, "normals")
        # A sparse set may retain its parent plane without having its shape.
    elif cls is RaySet:
        n = len(data["origins"])
        _unit_vectors(data["directions"], n, "directions")
        _array_shape(data["near"], (n,), "near", finite=True)
        _array_shape(data["far"], (n,), "far")
        _require(np.all(data["near"] >= 0) and not np.isnan(data["far"]).any() and
                 np.all(data["far"] >= data["near"]), "invalid ray intervals")
    elif cls is SampledPoints:
        n = len(data["points"])
        _array_shape(data["values"], (n, len(data["definitions"])), "sample values")
        _array_shape(data["owners"], (n,), "owners")
        _array_shape(data["valid"], (n,), "valid")
        _require(np.all(data["owners"] >= -1) and np.all(data["owners"][data["valid"]] >= 0),
                 "invalid sample owners")
    elif cls is ConnectivityMap:
        _require(np.array_equal(data["points"].positions, data["data"].seeds),
                 "map points and diagnostic seeds do not agree")
    elif cls is QSLResult:
        _validate_qsl(data)
    elif cls is RayResult:
        n = len(data["rays"].origins)
        for name in _ARRAYS[cls]:
            _array_shape(data[name], (n,), name)
        _codes(data["status"], LOSStatus, "LOS status")
        _require(not np.any(data["status"] == LOSStatus.RUNNING), "published LOS cannot be running")
        successful = np.isin(data["status"], (LOSStatus.COMPLETE, LOSStatus.EMPTY))
        _require(np.isfinite(data["values"][successful]).all(), "successful LOS values must be finite")
        _require(np.all(data["values"][data["status"] == LOSStatus.EMPTY] == 0), "empty LOS must be zero")
        _require(np.all(data["samples"] >= 0) and np.all(data["misses"] >= 0), "negative ray counts")
        _text(data["units"], "units")
        _text(data["quadrature"], "quadrature")
        _json_object(data["metadata"], "ray metadata")
    elif cls is LineSet:
        n = len(data["seeds"])
        total = _coordinates(data["positions"], "line positions")
        offsets = data["offsets"]
        _array_shape(offsets, (2*n+1,), "offsets")
        _array_shape(data["termination"], (n, 2), "termination")
        _require(offsets[0] == 0 and offsets[-1] == total and np.all(offsets >= 0) and
                 np.all(offsets <= total) and np.all(offsets[1:] >= offsets[:-1]), "invalid packed offsets")
        _codes(data["termination"], Termination, "line termination", (-1, -2))
        _require(not np.any(data["termination"] == Termination.RUNNING), "published lines cannot be running")
        counts = offsets[1:] - offsets[:-1]
        skipped = np.isin(data["termination"].ravel(), (-1, -2, int(Termination.OUTSIDE_SEED)))
        _require(np.all(counts[skipped] == 0), "untraced branches must be empty")
        _require(np.all(counts[~skipped] > 0), "traced branches must include their seed")
        rows = np.flatnonzero(counts)
        _require(np.array_equal(data["positions"][offsets[rows]], data["seeds"].positions[rows//2]),
                 "packed branches must start at their identified seed")
    elif cls is UniformResult:
        _require(data["valid"].ndim == 3 and all(n > 0 for n in data["valid"].shape),
                 "uniform validity requires a nonempty 3D layout")
        _array_shape(data["values"], (*data["valid"].shape, len(data["definitions"])), "uniform values")
        for name in ("lower", "upper"):
            _array_shape(data[name], (3,), name, finite=True)
        _require(np.all(data["upper"] > data["lower"]), "uniform bounds must be ordered")


def _validate_qsl(data):
    n = _coordinates(data["seeds"], "seeds")
    q_names = ("q", "log10_q", "q_perp", "log10_q_perp")
    has_q = data["q"] is not None
    _require(all((data[name] is not None) == has_q for name in q_names), "inconsistent optional Q arrays")
    _require(has_q or data["twist"] is not None, "at least Q or twist is required")
    for name in (*q_names, "twist", "length", "complete", "valid", "stencil_valid"):
        if data[name] is not None:
            _array_shape(data[name], (n,), name)
    for name in ("footpoints", "endpoint_fields"):
        _array_shape(data[name], (n, 2, 3), name)
    for name in ("boundary", "termination", "steps"):
        _array_shape(data[name], (n, 2), name)
    _codes(data["boundary"], Boundary, "boundary")
    _codes(data["termination"], ConnectivityTermination, "connectivity termination")
    _require(np.all(data["steps"] >= 0), "negative integration steps")
    _require(data["normalization"] in ("mapping", "flux"), "unknown Q normalization")
    expected_methods = ("finite-difference", "variational") if has_q else ("twist-only",)
    _require(data["method"] in expected_methods, "inconsistent diagnostic method")
    radius = data["local_radius"]
    _require(radius is None or (type(radius) in (int, float) and math.isfinite(radius) and radius > 0),
             "local_radius must be positive and finite or null")
    complete = np.all(np.isin(data["termination"], (3, 12)), axis=1)
    _require(np.array_equal(data["complete"], complete), "complete mask conflicts with termination")
    if has_q:
        valid = (complete & data["stencil_valid"] & np.all(np.isin(data["boundary"], (1,2,3,4,5,6,9)), axis=1)
                 & ~np.isnan(data["log10_q"]))
        _require(np.all(data["q"][valid] > 0), "valid Q must be positive and not NaN")
    else:
        valid = complete & np.isfinite(data["twist"])
    _require(np.array_equal(data["valid"], valid), "valid mask conflicts with diagnostic status")


class _Encoder:
    def __init__(self):
        self.arrays = {}

    def node(self, result):
        cls = type(result)
        _require(cls in _ARRAYS, "unsupported result type")
        node = {"type": cls.__name__}
        for name, dtype in _ARRAYS[cls].items():
            array = getattr(result, name)
            if array is None:
                _require(name in _OPTIONAL.get(cls, ()), f"{name} cannot be None")
                node[name] = None
            else:
                _require(isinstance(array, np.ndarray) and array.dtype.kind == np.dtype(dtype).kind and
                         array.dtype.itemsize == np.dtype(dtype).itemsize, f"invalid {name} dtype")
                key = f"a{len(self.arrays):04d}"
                self.arrays[key] = np.array(array, dtype=np.dtype(dtype).newbyteorder("<"), order="C", copy=True)
                node[name] = key
        for name in _CHILDREN.get(cls, {}):
            child = getattr(result, name)
            node[name] = None if child is None else self.node(child)
        for name in _PROPERTIES.get(cls, ()):
            value = getattr(result, name)
            if name == "local_radius" and isinstance(value, (np.integer, np.floating)):
                value = float(value)
            if name == "definitions":
                _require(type(value) in (list, tuple) and all(type(d) is FieldDefinition for d in value),
                         "invalid field definitions")
                value = [{"name": d.name, "units": d.units, "interpretation": d.interpretation} for d in value]
            node[name] = _json_value(value)
        return node


class _Decoder:
    def __init__(self, arrays):
        self.arrays = arrays
        self.used = set()

    def node(self, node, expected=None):
        _require(type(node) is dict and type(node.get("type")) is str, "invalid result type")
        cls = _TYPES.get(node["type"])
        _require(cls is not None and (expected is None or cls is expected), "unsupported or misplaced result type")
        _keys(node, ("type", *_ARRAYS[cls], *_CHILDREN.get(cls, {}), *_PROPERTIES.get(cls, ())), "result")
        data = {}
        for name, dtype in _ARRAYS[cls].items():
            key = node[name]
            if key is None:
                _require(name in _OPTIONAL.get(cls, ()), f"{name} cannot be null")
                data[name] = None
                continue
            _require(type(key) is str and key in self.arrays and key != _MANIFEST and key not in self.used,
                     f"invalid or duplicate array reference for {name}")
            self.used.add(key)
            array = self.arrays[key]
            _require(isinstance(array, np.ndarray) and array.dtype == np.dtype(dtype).newbyteorder("<"),
                     f"invalid {name} dtype")
            # NPZ reads and encoder snapshots already own their arrays; only
            # byte-order/layout conversion requires another allocation.
            data[name] = np.asarray(array, dtype=dtype, order="C")
            data[name].flags.writeable = False
        for name, child_type in _CHILDREN.get(cls, {}).items():
            if node[name] is None:
                _require(cls is PointSet and name == "plane", f"{name} cannot be null")
                data[name] = None
            else:
                data[name] = self.node(node[name], child_type)
        for name in _PROPERTIES.get(cls, ()):
            data[name] = _definitions(node[name]) if name == "definitions" else node[name]
        _validate(cls, data)
        if cls in _IDENTIFIED:
            data["source_identity"] = None
        result = cls(**data)
        # Constructors normalize vectors. Preserve the already-validated bits,
        # since renormalization can change ray clipping at grazing surfaces.
        for name in ("normals",) if cls is PointSet else ("directions",) if cls is RaySet else ():
            object.__setattr__(result, name, data[name])
        return result


def _manifest_result(manifest, arrays):
    _keys(manifest, ("format", "schema_version", "result", "metadata", "provenance"), "manifest")
    _require(manifest["format"] == _FORMAT, "unsupported result format")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == SCHEMA_VERSION,
             "unsupported schema version")
    metadata = _json_object(manifest["metadata"], "metadata")
    provenance = manifest["provenance"]
    _keys(provenance, ("source", "verification"), "provenance")
    source = provenance["source"]
    if source is not None:
        source = _json_object(source, "source")
    _require(provenance["verification"] == ("unspecified" if source is None else "unverified"),
             "source verification cannot be asserted by a result file")
    decoder = _Decoder(arrays)
    result = decoder.node(manifest["result"])
    _require(type(result) is not Plane, "standalone Plane is not a supported result")
    _require(decoder.used == set(arrays) - {_MANIFEST}, "unexpected array members")
    return ResultFile(result, metadata, source)


def save_result(path, result, *, metadata=None, source=None, overwrite=False):
    """Save one supported object to a versioned NPZ file and return its Path.

    Metadata and optional source descriptions must contain finite JSON values.
    Tuples become lists. Supply calculation controls, units absent from the
    result, and model assumptions explicitly; source_identity is never saved.
    Existing files are refused unless overwrite=True. Publication follows the
    temporary-sibling/link-or-replace protocol of io.products.write_amrvac.
    """
    _require(type(overwrite) is bool, "overwrite must be boolean")
    destination = Path(path)
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(destination)
    metadata = _json_object({} if metadata is None else metadata, "metadata")
    source = None if source is None else _json_object(source, "source")
    encoder = _Encoder()
    manifest = {"format": _FORMAT, "schema_version": SCHEMA_VERSION,
                "result": encoder.node(result),
                "metadata": metadata,
                "provenance": {"source": source,
                               "verification": "unspecified" if source is None else "unverified"}}
    # Validate the same representation that load_result will accept, before
    # opening any destination. No caller object is changed by validation.
    encoded = json.dumps(manifest, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    _require(len(encoded) <= _MAX_MANIFEST_BYTES, "manifest exceeds the 1 MiB limit")
    _manifest_result(manifest, encoder.arrays)
    encoder.arrays[_MANIFEST] = np.frombuffer(encoded, dtype=np.uint8)
    fd, temporary = tempfile.mkstemp(prefix=".simesh-result-", suffix=".npz", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **encoder.arrays)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return destination
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def load_result(path):
    """Load and validate a result without pickle or source reconstruction.

    Return ResultFile; access its .result to use the ordinary application API.
    Malformed/unsupported files raise ResultFileError; filesystem errors such as
    missing files or denied access remain OSError. Arrays are owned/read-only.
    This whole-result format has no streaming or bounded-memory guarantee.
    """
    try:
        # Validate ZIP names before NumPy strips .npy suffixes, preventing
        # duplicate/ambiguous members. No archive entries are extracted to disk.
        with open(path, "rb") as stream, zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            names = [info.filename for info in members]
            _require(len(names) == len(set(names)) and f"{_MANIFEST}.npy" in names and
                     all(name == f"{_MANIFEST}.npy" or re.fullmatch(r"a[0-9]{4,}\.npy", name)
                         for name in names),
                     "invalid NPZ members")
            _require(archive.getinfo(f"{_MANIFEST}.npy").file_size <= _MAX_MANIFEST_BYTES + 16384,
                     "manifest exceeds the 1 MiB limit")
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as arrays:
                raw = arrays[_MANIFEST]
                _require(raw.dtype == np.uint8 and raw.ndim == 1 and raw.size <= _MAX_MANIFEST_BYTES,
                         "manifest must be a UTF-8 uint8 vector of at most 1 MiB")
                manifest = json.loads(raw.tobytes().decode("utf-8"), object_pairs_hook=_unique_object,
                                      parse_constant=_invalid_constant)
                return _manifest_result(manifest, arrays)
    except ResultFileError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError,
            zipfile.BadZipFile, EOFError, zlib.error, NotImplementedError, RuntimeError) as error:
        raise ResultFileError(f"invalid result file: {error}") from error


def _invalid_constant(value):
    raise ResultFileError(f"nonfinite JSON constant: {value}")
