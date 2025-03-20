from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import glob
import os

# Find all .pyx files in src/yt/utils/lib/
cython_files = glob.glob('src/yt/utils/lib/*.pyx')
extensions = [
    Extension(
        f"yt.utils.lib.{os.path.splitext(os.path.basename(f))[0]}", 
        [f]
    ) for f in cython_files
]

setup(
    name="simesh",
    version="0.1.0",
    description="A mesh handling package for simulation data from AMRVAC datfile.",
    author="Hao Wu",
    author_email="aster0502@outlook.com",
    license="GPL-3.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    
    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Python version requirement
    python_requires=">=3.11",
    
    # Core dependencies
    install_requires=[
        "numpy>=2.1.1",
        "cython>=3.0.0",
    ],
    
    # Test dependencies
    extras_require={
        "test": [
            "pytest>=8.3.4",
        ],
    },
    
    # Cython extension configuration
    ext_modules=cythonize(extensions),
) 