from setuptools import Extension
from Cython.Build import cythonize
import glob
import os
import sys

def build(setup_kwargs):
    # Define different groups of Cython files
    cython_groups = {
        'core': ['src/yt/utils/lib/core*.pyx'],
        'io': ['src/yt/utils/lib/io*.pyx'],
        'test': ['src/yt/utils/lib/test*.pyx'],
        # Add more groups as needed
    }

    # Check if a specific group is requested via environment variable
    build_group = os.environ.get('CYTHON_BUILD_GROUP', 'all')
    
    extensions = []
    
    if build_group == 'all':
        # Compile all .pyx files
        cython_files = glob.glob('src/yt/utils/lib/*.pyx')
        extensions = [
            Extension(
                f"yt.utils.lib.{os.path.splitext(os.path.basename(f))[0]}", 
                [f]
            ) for f in cython_files
        ]
    else:
        # Compile only files from the specified group
        if build_group in cython_groups:
            for pattern in cython_groups[build_group]:
                files = glob.glob(pattern)
                extensions.extend([
                    Extension(
                        f"yt.utils.lib.{os.path.splitext(os.path.basename(f))[0]}", 
                        [f]
                    ) for f in files
                ])
        else:
            print(f"Warning: Unknown group '{build_group}'. Available groups: {list(cython_groups.keys())}")
            sys.exit(1)
    
    # Add extensions to setup_kwargs
    setup_kwargs.update({
        "ext_modules": cythonize(extensions),
    }) 