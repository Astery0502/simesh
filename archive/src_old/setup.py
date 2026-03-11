# setup.py
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "amr_selection", 
        ["amr_selection.pyx"], 
        include_dirs=[np.get_include()],
        extra_compile_args=["-g", "-O0"],
        extra_link_args=["-g"]
    )
]

setup(
    name="amr_selection",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'linetrace':True, 'binding':True},
        annotate=True
    ),
    zip_safe=False,
)
