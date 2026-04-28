import subprocess
import sys
from pathlib import Path
import shutil

def build_cython(group=None, clean=False, openmp=False):
    root_dir = Path(__file__).parent.parent
    build_script = root_dir / 'build.py'  # Get absolute path to build.py
    
    if clean:
        # Clean previous builds
        for ext in ['*.so', '*.pyd', '*.c']:
            for file in root_dir.rglob(ext):
                file.unlink()
        for artifact in ['build', 'dist']:
            artifact_path = root_dir / artifact
            if artifact_path.exists():
                shutil.rmtree(artifact_path)
        for egg_info in root_dir.rglob('*.egg-info'):
            shutil.rmtree(egg_info)
    
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
    parser.add_argument('--openmp', action='store_true', help='Build with OpenMP compiler and linker flags')
    args = parser.parse_args()
    
    build_cython(args.group, args.clean, args.openmp)
