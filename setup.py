"""Build only this generation's extensions, retaining primitive semantics."""

from pathlib import Path
import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent.resolve()
SOURCE = ROOT / "src"
PRIMITIVES = "simesh._kernels.primitives."
openmp = os.environ.get("SIMESH_OPENMP", "0").lower() in {"1", "true", "yes"}
include = [np.get_include()]
compile_args = ["-O3"]
link_args = []
library_dirs = []
if openmp:
    if sys.platform == "darwin":
        compile_args += ["-Xpreprocessor", "-fopenmp"]
        link_args += ["-lomp"]
        for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
            if (prefix / "include/omp.h").is_file():
                include.append(str(prefix / "include"))
            if (prefix / "lib/libomp.dylib").is_file():
                library_dirs.append(str(prefix / "lib"))
    else:
        compile_args += ["-fopenmp"]
        link_args += ["-fopenmp"]

primitives, kernels = [], []
for path in sorted((SOURCE / "simesh/_kernels").rglob("*.pyx")):
    name = ".".join(path.relative_to(SOURCE).with_suffix("").parts)
    primitive = name.startswith(PRIMITIVES)
    extension = Extension(
        name, [path.relative_to(ROOT).as_posix()],
        include_dirs=[np.get_include()] if primitive else include,
        extra_compile_args=[] if primitive else compile_args,
        extra_link_args=[] if primitive else link_args,
        library_dirs=[] if primitive else library_dirs,
    )
    (primitives if primitive else kernels).append(extension)

extensions = cythonize(
    primitives, include_path=[str(SOURCE)],
    compiler_directives={"language_level": 3},
)
extensions += cythonize(
    kernels, include_path=[str(SOURCE)],
    compiler_directives={"language_level": 3, "boundscheck": False,
                         "wraparound": False, "cdivision": True},
)
setup(ext_modules=extensions)
