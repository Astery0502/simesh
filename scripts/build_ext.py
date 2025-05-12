import subprocess
import sys
from pathlib import Path

def build_cython(group=None, clean=False):
    root_dir = Path(__file__).parent.parent
    build_script = root_dir / 'build.py'  # Get absolute path to build.py
    
    if clean:
        # Clean previous builds
        for ext in ['*.so', '*.pyd', '*.c']:
            for file in root_dir.rglob(ext):
                file.unlink()
    
    cmd = [sys.executable, str(build_script), '--inplace']  # Use absolute path
    if group:
        cmd.extend(['--group', group])
    
    subprocess.run(cmd, cwd=str(root_dir))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', help='Specific group to compile (core, io, amr, etc.)')
    parser.add_argument('--clean', action='store_true', help='Clean before building')
    args = parser.parse_args()
    
    build_cython(args.group, args.clean) 