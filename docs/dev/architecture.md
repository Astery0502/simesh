# Architecture

```text
           file / arrays
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

`_kernels` and checked `_amr` wrappers implement active native arithmetic.
`io/products.py` assembles complete root subtrees in the output's Morton order;
regional Fields retain their original Mesh. File crops build export indices,
not a second analysis Mesh, and transfer only selected ordinary fields through
bounded payload batches. Global input Mesh/index storage remains resident.
`io/_v5/writer.py` shares batch serialization between file and Fields exports,
without a mutable mesh dependency. Historical Dataset implementations are excluded from the package.
`_kernels/primitives` retains its original compiler semantics.
`physics/thermal.py` owns tabulated EUV responses and explicit thermal nodes;
`physics/radiation.py` constructs EUV absorption and radio coefficients without
modifying the simulation EOS. Both thermal and ordered transfer consumers reuse
`_kernels/thermal_rays.pyx` for leaf ownership and interpolation-knot traversal.
See [core contracts](api.md) and [build groups](cython-build.md).

`geometry.py` owns native bottom-face seed positions and area weights, reusable
by magnetic diagnostics and tracing. `current_proxy.py` is a composed magnetic
application: its convenience iterator uses existing tracing, while
`current_proxy_from_lines` reuses stored `LineSet` geometry and samples curl with
`sample_line_profiles`. The proxy module owns only its closed-line policy,
arc-length average and display-cell deposition. It keeps geometric closure,
contribution acceptance and original trace termination distinct. Source reads,
checkpoint files and view selection remain in the calling example.
