# Architecture

```text
file / arrays / explicit Dataset copy
                 |
               Source ---- Mesh / Selection
                 |
        read_fields / prepare
                 |
               Fields
                 |
     scientific / application consumers
                 |
     geometry, values, status -> result I/O
```

| Boundary | Contract |
| --- | --- |
| Mesh / values | Shared immutable geometry has no field storage |
| Source / Fields | Reads and preparation are explicit; owned published fields survive Source closure |
| Coverage / region | Selections retain complete original leaves; region edges are not physical boundaries |
| Storage / validity | Allocated padding is not valid halo; a first derivative consumes one valid layer |
| Owned / borrowed | Iterator and pool views expire with their lease; caller output retains alias constraints |
| Science / execution | Numerical schemes, models and units remain explicit; consumers do not fetch missing input |
| Input / output batching | Batching results does not make all input consumers support bounded memory |
| Dataset / native | Mutable AMRMesh stays in `amrvac`; crossing uses explicit copies or ordinary file export |

`_kernels` and checked `_amr` wrappers implement active native arithmetic.
`_kernels/primitives` and `amrvac/_mesh` retain separate compiler semantics.
See [core contracts](api.md) and [build groups](cython-build.md).
