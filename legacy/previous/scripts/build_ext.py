import subprocess
import sys
from pathlib import Path
import shutil

def clean_build_artifacts(root_dir):
    """Remove only this project's generated extension/build artifacts."""
    root_dir = Path(root_dir).resolve()
    source_roots = (root_dir / 'src/simesh/utils/lib', root_dir / 'rewrite/src/simesh_rewrite')
    for source_root in source_roots:
        if not source_root.is_dir() or not source_root.resolve().is_relative_to(root_dir):
            continue
        for source in source_root.rglob('*.pyx'):
            for suffix in ('.c', '.cpp', '.so', '.pyd'):
                generated = source.with_suffix(suffix)
                if generated.is_file():
                    generated.unlink()
            for pattern in (source.stem + '.*.so', source.stem + '.*.pyd'):
                for generated in source.parent.glob(pattern):
                    generated.unlink()
    for relative in ('build', 'dist', 'rewrite/build', 'src/simesh.egg-info',
                     'simesh.egg-info', 'rewrite/src/simesh_rewrite_core.egg-info',
                     'rewrite/simesh_rewrite_core.egg-info'):
        artifact = root_dir / relative
        if artifact.is_symlink():
            artifact.unlink()
        elif artifact.is_dir():
            shutil.rmtree(artifact)


def build_cython(group=None, clean=False, openmp=False):
    root_dir = Path(__file__).parent.parent
    build_script = root_dir / 'build.py'  # Get absolute path to build.py
    
    if clean:
        clean_build_artifacts(root_dir)
    
    cmd = [sys.executable, str(build_script), '--inplace']  # Use absolute path
    if group:
        cmd.extend(['--group', group])
    if openmp:
        cmd.append('--openmp')
    
    subprocess.run(cmd, cwd=str(root_dir), check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', help='Specific subdirectory under src/simesh/utils/lib to compile')
    parser.add_argument('--clean', action='store_true', help='Clean before building')
    parser.add_argument('--clean-only', action='store_true', help='Remove project build artifacts without rebuilding')
    parser.add_argument('--openmp', action='store_true', help='Build with OpenMP compiler and linker flags')
    args = parser.parse_args()
    
    if args.clean_only:
        clean_build_artifacts(Path(__file__).parent.parent)
    else:
        build_cython(args.group, args.clean, args.openmp)
