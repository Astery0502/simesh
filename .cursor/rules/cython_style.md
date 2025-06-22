# Cython Style Guide for simesh/utils/lib

These rules apply **exclusively** to `.pyx` / `.pxd` files inside the Cython performance module `src/simesh/utils/lib/` and its sub-packages (e.g., `amr/`).

Other Python modules in the simesh project follow standard Python conventions.

---
## 1. Compiler Directives (mandatory header)
```cython
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
```

---
## 2. Typing Discipline
- **Always** declare argument and return types (`cdef`, `cpdef`, `ctypedef`).
- Prefer `uint32_t` / `int32_t` for indices and small integers.
- Use `bint` for booleans.
- Expose Python-visible APIs via `cpdef`; keep inner loops `cdef inline nogil`.

---
## 3. Memory & Data
1. **No NumPy in hot loops** – use memoryviews (`double[:]`, `uint32_t[:]`) or raw pointers.
2. Allocate with `malloc` / `free` for large C arrays; document ownership.
3. Provide a `__dealloc__` method when allocating manually.
4. Use `packed struct` wrappers for pointer handles (see `octptr`).
5. **Loop ordering for cache efficiency** – iterate arrays in memory order: for `array[nx, ny]`, loop `nx` first, then `ny` (row-major order).

---
## 4. Performance Checklist
- Add `nogil` on any function that never touches Python objects.
- Mark tiny helpers `inline`.
- Avoid Python exceptions in performance-critical paths; use return codes.
- Limit nested Python attribute access in `.pyx`.

---
## 5. File Layout Convention
1. `.pxd` – C declarations, structs, fused types.
2. `.pyx` – implementation.
3. Keep one top-level class/feature per file when feasible.

---
## 6. Naming Rules
| Entity          | Format      | Example              |
|-----------------|-------------|----------------------|
| C struct        | PascalCase  | `OctreeNode`         |
| Cython cdef var | snake_case  | `neighbor_type`      |
| Pointer wrapper | PascalCase* | `NodePtr`            |
| Functions       | snake_case  | `fill_morton_mapping`|
| Constants       | UPPER_SNAKE | `NEIGHBOR_FINE`      |

\* use `packed struct` where needed.

---
## 7. Documentation
- Provide a one-line docstring for every public (`cpdef`) function.
- For complex algorithms, add inline C-style comments **inside** the function.

---
## 8. Testing & Benchmarking
- Each new Cython function **must** have:
  1. Correctness unit test (in `tests/utils/lib/`)
  2. A micro-benchmark (pytest-benchmark or manual `time.perf_counter`) ensuring no regression.

---
## 9. Auto-enforcement
Cursor should refuse to:
- Insert NumPy operations inside loops over >100 000 iterations.
- Leave bounds-checking enabled in such loops.
- Add Python-level try/except in `nogil` blocks.

Cursor **must** update this guide if the style evolves.

---
_This file lives at `.cursor/rules/cython_style.md` and guides Cython edits within `src/simesh/utils/lib/` only._ 