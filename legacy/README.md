# Historical implementations

This directory is excluded from the root package build and default tests.
Current implementation work belongs in `../src/simesh/`.

| Directory | Origin and purpose |
| --- | --- |
| `previous/` | The preceding root package, its `rewrite/` provider, tests, build tools and documents |
| `python-first/` | The earlier `src/simesh/legacy/` Python implementation and `archive/` sources, separated from the preceding main package |
| `core-development/` | Historical N1–N4 comparison runners, logs, environments and generated artifacts from the former `analysis-core/` workspace |

`python-first/src/simesh/legacy/` preserves the original source namespace for
reference. It is not the new package's legacy namespace. The older main
package's comparisons originally imported that implementation from the same
source tree; that historical layout is available in Git. Archived scripts keep
their historical assumptions and are not promised to run from these new paths.
Recreate an isolated environment and use the original revision/layout for exact
historical reproduction. Saved virtual environments and generated extensions
are local artifacts, not portable installations.

Retained code that still provides current functionality is maintained in the
active package: stateful AMRVAC operations under `simesh.amrvac`, their private
Cython mesh under `simesh.amrvac._mesh`, and array configurations under
`simesh.tools.configurations`. Native analysis uses `simesh._kernels`.
