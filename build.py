from setuptools import Extension
from Cython.Build import cythonize
import glob
import os
import sys
from pathlib import Path
import numpy as np

def get_extensions(group=None):
    """Get extension modules, optionally filtered by group"""
    # Simple group definitions
    cython_groups = {
        'utils': ['src/simesh/utils/lib/*.pyx'],
        'amr': ['src/simesh/utils/lib/amr/*.pyx'],
    }
    
    # Get absolute paths for includes
    root_dir = Path(__file__).parent
    include_dirs = [
        str(root_dir / 'src/simesh/utils/lib'),
        np.get_include(),
    ]
    
    extensions = []
    
    if group is None or group == 'all':
        # Compile all .pyx files
        patterns = [pat for patterns in cython_groups.values() for pat in patterns]
    else:
        # Compile only files from the specified group
        if group not in cython_groups:
            print(f"Warning: Unknown group '{group}'. Available groups: {list(cython_groups.keys())}")
            return []
        patterns = cython_groups[group]
    
    for pattern in patterns:
        for f in glob.glob(pattern):
            module_path = str(Path(f).relative_to('src')).replace(os.path.sep, '.')[:-4]
            extensions.append(
                Extension(
                    module_path,
                    [f],
                    extra_compile_args=['-O3'],
                    include_dirs=include_dirs,
                )
            )
    
    return extensions

def build(setup_kwargs=None):
    """Main build function called by poetry"""
    extensions = get_extensions()
    if setup_kwargs is not None:
        setup_kwargs.update({
            "package_dir": {"": "src"},
            "ext_modules": cythonize(
                extensions,
                compiler_directives={
                    'language_level': "3",
                    'boundscheck': False,
                    'wraparound': False,
                    'cdivision': True,
                    'embedsignature': True,
                    'profile': False,
                    'linetrace': False,
                },
                nthreads=4,
            ),
        })
    return setup_kwargs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', help='Specific group to compile (core, io, amr, etc.)')
    parser.add_argument('--inplace', action='store_true', help='Build in-place for development')
    args = parser.parse_args()
    
    from setuptools import setup
    
    extensions = get_extensions(args.group)
    if not extensions:
        sys.exit(1)
        
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'embedsignature': True,
            'profile': True,
            'linetrace': True,
        },
        nthreads=4,
    )
    
    if args.inplace:
        setup(
            name="simesh",
            package_dir={"": "src"},
            ext_modules=ext_modules,
            script_args=['build_ext', '--inplace']
        )
    else:
        setup(
            name="simesh",
            ext_modules=ext_modules,
            script_args=['build']
        ) 