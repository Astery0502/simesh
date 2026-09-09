# Snapshot metadata

`open_amrvac` exposes the already parsed file header as `source.metadata`, a
detached `SnapshotMetadata`. It owns no file handle, mesh, field values, or
reader. It remains usable after the Source closes. Header arrays become
immutable tuples, and the header and named parameter mappings are read-only.

```python
import simesh as sm

with sm.open_amrvac("snapshot.dat", fields=("b1", "b2", "b3")) as source:
    metadata = source.metadata
    magnetic = sm.prepare(source, scheme="coordinate-phase")

print(metadata.time, metadata.iteration, metadata.physics_type)
print(metadata.parameters)
curl_b = sm.curl(magnetic)
sm.write_amrvac("curl.dat", curl_b, metadata=metadata)
```

`time`, `iteration`, `physics_type`, `field_names`, and `parameters` are views of
the recorded header. `header` contains the original AMRVAC keys, including
geometry, periodicity, CT flag, field directory, parameters, and output counters.
Duplicate parameter names remain preserved in `header["param_names"]` and
`header["params"]`; the named `parameters` view rejects that ambiguity.
Recorded time and parameters retain their original units. Metadata does not
infer normalization, an EOS, or the meaning of the energy variable.

`source.fields` describes the currently available source columns. In contrast,
`metadata.field_names` retains the original complete file directory even after
`open_amrvac(fields=...)`, `select_source`, or `cache_source(fields=...)`.
Source selection and cache adapters share the same metadata object. Preparing
a region does not change the original header or attach it to the resulting
Fields; retain the description separately when the application needs it.

Array Sources have `metadata=None` by default. Their optional `metadata`
argument accepts an explicit `SnapshotMetadata(header, path=...)` description.
`source_from_dataset` copies the Dataset header without retaining its mutable
arrays or the Dataset itself. For an edited Dataset or caller-supplied arrays,
the description does not certify that values still match the original file.

`path`, `byte_order`, and `file_identity` record available file observations.
The identity tuple contains device, inode, byte size, modification time in
nanoseconds, and change time in nanoseconds. It is not a content hash, a live
file check, or a persisted Fields identity certificate. Dataset adapters do
not re-open or stat the source path to invent fresh observations.

## Output and persistence

`write_amrvac` accepts either `SnapshotMetadata` or the existing header dict.
It still validates full mesh coverage and matching geometry, rebuilds field
names and tree offsets, and writes ordinary values with `staggered=False`.
It does not mutate the original metadata or guarantee simulation restart
compatibility. `metadata.to_header()` provides an independent mutable header
when an existing compatibility interface requires NumPy arrays and lists.

`to_dict()` returns detached JSON data for the existing result-file interface:

```python
points = sm.PointSet([[0.5, 0.5, 0.5]])
sm.save_result(
    "points.npz", points,
    metadata={"snapshot": metadata.to_dict(), "description": "Selected seeds"},
)
```

The JSON description contains `format`, `header`, `path`, `byte_order`, and
`file_identity`. `to_dict()` rejects nonfinite header values rather than
silently replacing them; inspecting the in-memory metadata remains possible.
Loading a result restores the description as JSON data, not as an automatically
verified association between a file and the saved numerical result.
