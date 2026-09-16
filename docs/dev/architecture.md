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


The field-line dependency direction is:

```text
applications / connectivity / current_proxy
          |                   |
      tracing           operators.line / line_profiles
          |
_kernels.streamlines / _kernels.connectivity
          |                   |
          +---- rk4.pxd -------+
```

`rk4.pxd` owns the only classical RK4 stage loop. It accepts a compiled stage
function and caller-owned buffers of arbitrary state width. It imports neither
mesh operations nor applications. Prepared-field adapters own interpolation,
local-cell step admission and field-specific right-hand sides; the drivers own
accepted-prefix or localized-event admission and commit. QSL vector rescaling
and endpoint projection remain in connectivity. `operators.line` provides
array-only sampled-curve calculus; current-proxy deposition reuses its trapezoid
operation after selecting closed paths, preserving its existing discretization.

`_execution` continues to own thread-pool lifetimes and disjoint seed ranges.
Adapters bind inputs once per range, allocate thread-private scratch before the
hot loop, and call the shared kernel without the GIL. Missing coverage and full
path buffers preserve integration state. Resident integrands in bounded tracing
must cover the entire Mesh; only primary/curl coverage is supplied by its pool.
