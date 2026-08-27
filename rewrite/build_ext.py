"""Build only the Cython extensions owned by the isolated rewrite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(ROOT)
    extensions = [
        Extension(
            "simesh_rewrite._foundation",
            [str(ROOT / "src/simesh_rewrite/_foundation.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._access",
            [str(ROOT / "src/simesh_rewrite/_access.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._morton",
            [str(ROOT / "src/simesh_rewrite/_morton.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._topology",
            [str(ROOT / "src/simesh_rewrite/_topology.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._geometry",
            [str(ROOT / "src/simesh_rewrite/_geometry.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._storage",
            [str(ROOT / "src/simesh_rewrite/_storage.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._chunking",
            [str(ROOT / "src/simesh_rewrite/_chunking.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._halos",
            [str(ROOT / "src/simesh_rewrite/_halos.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._sampling",
            [str(ROOT / "src/simesh_rewrite/_sampling.pyx")],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "simesh_rewrite._operators",
            [str(ROOT / "src/simesh_rewrite/_operators.pyx")],
            include_dirs=[np.get_include()],
        ),
    ]
    setup(
        name="simesh-rewrite-core",
        package_dir={"": "src"},
        packages=["simesh_rewrite"],
        ext_modules=cythonize(
            extensions,
            build_dir=str(ROOT / "build/cython"),
            compiler_directives={"language_level": "3"},
        ),
        script_args=[
            "build_ext",
            "--inplace",
            "--force",
            "--build-temp",
            str(ROOT / "build/temp"),
        ],
    )


if __name__ == "__main__":
    main()
