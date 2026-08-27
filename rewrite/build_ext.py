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
