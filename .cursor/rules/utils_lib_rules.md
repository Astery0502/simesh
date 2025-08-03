# Project Rules for simesh/utils/lib

> This file is parsed by Cursor. These rules apply **only** to the Cython module at `src/simesh/utils/lib/` and its sub-packages.

---
## 1. Codebase Outline (auto-generated)
- **src/simesh/utils/lib** (Cython performance module)
    - **tree.pxd** : core data structures (`OctreeNode`, pointer wrappers)
    - **math.pxd** : small inlined math helpers (bit utilities)
    - **amr/** (Adaptive-Mesh-Refinement sub-package)
        - **morton.(pxd|pyx)** : Morton encoding utilities
        - **forest.(pxd|pyx)** : `AMRForest` – octree construction & connectivity
        - **mesh.(pxd|pyx)** : `AMRMesh` – block data storage, boundary handling
- **tests/utils/lib**
    - **test_amr.py** : validates Morton ordering, forest construction, basic mesh BCs

---
## 2. Task Board  
Cursor must keep these two lists up-to-date whenever it edits code.

### 🟡 Ongoing
<!-- cursor-ongoing:start -->
* _(none – add new tasks here)_
<!-- cursor-ongoing:end -->

### ✅ Completed
<!-- cursor-done:start -->
* Initial AMR prototype (Morton mapping, forest connectivity, basic BC) – tests pass
* Fixed 2D compatibility issues and memory management in AMR modules
<!-- cursor-done:end -->

---
## 3. Implementation Rules
1. **Tests first** Every new or modified public function/class **must** receive:
   * Unit tests under `tests/utils/lib/`
   * Performance test (pytest-benchmark or explicit timing) if the function is performance-critical.
2. **Status update** After implementing/tests pass, move item from *Ongoing* to *Completed*.
3. **Answer hygiene** Each assistant response **must NOT** contain:
   * Requests for user praise or feedback
   * Irrelevant chatter
4. **Deliverables in each answer** If code is added/changed:
   * The code edit(s)
   * The corresponding test file/section
   * A short note confirming both compile & tests added
5. **Double-check** Before finishing a turn, re-read the diff and ensure it compiles & tests reference correct symbols.
6. **Spec adherence** If a listed function in *Function Catalog* is altered, ensure tests still cover the "Must-test characteristics"; update the table when behaviour changes.

---
## 4. Dependency Management
- Runtime dependencies are declared in `pyproject.toml` / `requirements.txt`.
- Cython extensions are built via `scripts/build_ext.py`; new modules must be added to that script or grouped similarly.

---
## 5. Lint & Style (Python part)
- Black 88, flake8, mypy strict
- Docstrings in Google style
- Max cyclomatic complexity 12

(For Cython-specific style see `cython_style.md`.)

---
## 6. Enforcement Hooks
Cursor should run:
```bash
make utils   # build Cython utils group at src/simesh/utils/lib
pytest tests/utils/lib -q
```
before marking a task as completed.

---
## 7. Function Catalog
_This catalogue helps future agents understand current APIs and what must be tested.  Update when APIs change._

#### `AMRForest`

**Main Functions - Core Procedures:**
| Function | Main Feature Design & Procedure | Key Design Aspects |
|----------|--------------------------------|-------------------|
| `__cinit__(ndim, ng1, ng2, ng3, is_leaf)` | **Forest initialization pipeline** - Entry point that orchestrates the complete AMR forest construction from user specification. | ✔ **Unified interface** - single call creates fully functional forest <br> ✔ **Pipeline coordination** - calls read_forest → build_connectivity sequence <br> ✔ **Neighbors orchestration** - sequence from tree node neighbors at normal direction to leaf blocks all neighbors |
| `read_forest(is_leaf)` | **Tree structure construction** - Converts flat boolean leaf specification into hierarchical octree structure with proper parent-child relationships. | ✔ **Hierarchical building** - recursively constructs tree levels on morton ordering 1st level block with iterating children blocks <br>✔ **Neighbor allocation** - allocate normal node neighbors before iterating the children nodes (note that here the neighbor is structural sibling node, but the neighbor_type is for real leaf block). |
| `asign_tree_neighbor(tree)` | **Direct neighbor assignment** - Establishes immediate (same-level enough) neighbor relationships for each tree node in the tree, forming the foundation of AMR connectivity. | ✔ **Bidirectional linking** - ensures consistent neighbor relationships, success once in bidirections is ok <br>✔ **Sibling neighbors only** - in the Octree only sibling neighbor is required, and don't account for neighbor_coarsen because finer node does not have sibling nodes. |
| `build_connectivity()` | **Complete leaf connectivity construction** - Builds comprehensive neighbor information for real leaf blocks, enabling efficient AMR operations. | ✔ **Multi-resolution neighbors** - handles coarse/fine/sibling relationships from Octree to leaf blocks depending on leaf spatial indices <br>✔ **Comprehensive mapping** - populates neighbor_index/type/children arrays |

**Test & Validation Functions:**
| Function | Validation Purpose | Test Design Features |
|----------|-------------------|---------------------|
| `write_forest()` | **Serialization for testing** - Exports forest to boolean array for round-trip validation, I/O testing, and debugging visualization. | ✔ **Round-trip verification** - enables testing of read_forest correctness <br>✔ **Debugging support** - provides compact forest representation <br>✔ **I/O interface** - supports forest persistence and transfer |
| `test_neighbors()` | **Connectivity validation** - Comprehensive consistency checker for all neighbor relationships, essential for AMR algorithm correctness. | ✔ **Internal consistency** - validates bidirectional neighbor links <br>✔ **Geometric correctness** - checks neighbor positions match expectations |

**Helper Functions - Technical Implementation:**
| Category | Functions | Implementation Details & Attention Points |
|----------|-----------|------------------------------------------|
| **Recursive Tree Operations** | `read_node()`, `write_node()` | **Depth-first tree traversal**. ⚠️ **Critical ordering**: read_node and write_node must use identical traversal order. Memory management during recursion requires careful pointer handling. |
| **Fine-Level Connectivity** | `build_neighbor_children()` | **Multi-resolution neighbor mapping**. ⚠️ **Complex indexing**: handles 4^ndim neighbor children with intricate index calculations. Performance-sensitive due to nested loops. |
| **Neighbor Search Algorithms** | `find_neighbor()`, `find_root_neighbor()` | **Hierarchical neighbor discovery**. ⚠️ **Multi-level logic**: different search paths for level 1 vs. deeper levels. Handles boundary conditions, periodic wrapping, and cross-parent traversal. |
| **Index Management** | `findex()`, `nindex()`, `ncindex()` | **Memory layout utilities**. ⚠️ **Dimension sensitivity**: must handle 2D/3D cases correctly. Inline for performance in tight loops. Critical for array access correctness. |

#### `AMRMesh`
| Function | Purpose | Must-test characteristics |
|----------|---------|---------------------------|
| `getbc()` | Fill ghost-cells & physical boundaries | ✔ Data unchanged in interior <br>✔ Proper synchronization for sibling/coarse/fine blocks <br>✔ O(leafs·block) runtime |
| `_init_block_gridindex()` | Pre-compute index tables | ✔ Indices within bounds <br>✔ Consistent with `ng`, `bsize` |
| `export_slab_uniform()` | Sample AMR data to uniform slab | ✔ Correct interpolation vs. analytic function <br>✔ Handles blocks completely outside slab |

#### Morton utilities
| Function | Purpose | Must-test characteristics |
|----------|---------|---------------------------|
| `morton3D(x,y,z)` | Encode coordinates to Morton code | ✔ Matches bit-interleave reference <br>✔ Works up to 10-bit inputs |
| `fill_morton_mapping3D()` | Populate lookup tables | ✔ `ig2morton[i,j,k] == morton3D` <br>✔ `morton2ig[morton] == (i,j,k)` <br>✔ Memory safe for non-power-of-two grids |

---
_This file lives at `.cursor/rules/utils_lib_rules.md` and governs the Cython module `src/simesh/utils/lib/`._ 