from setuptools import setup, find_packages

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
    ],
    
    # Test dependencies
    extras_require={
        "test": [
            "pytest>=8.3.4",
            "f90nml>=1.4.4",
            "jupyterlab>=4.3.4"
        ],
    },
) 