from setuptools import setup, find_packages

setup(
    name="simesh",
    version="0.1.0",
    description="A mesh handling package for simulation data",
    author="Astery",
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
        "numpy>=2.2.1",
    ],
    
    # Test dependencies
    extras_require={
        "test": [
            "pytest>=8.3.4",
            "yt>=4.4.0",
            "f90nml>=1.4.4",
            "jupyterlab>=4.3.4",
            "ipykernel>=6.29.5",
        ],
    },
) 