"""Verify a built wheel without editable hooks or parent package imports."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.wheel) as archive:
        archive.extractall(args.target)
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys
from pathlib import Path
installed, dependencies, tests = map(Path, sys.argv[1:])
sys.path[:0] = [str(installed), str(dependencies), str(tests)]
import simesh
assert Path(simesh.__file__).resolve().is_relative_to(installed)
assert not any(name.startswith(('simesh.amrvac', 'simesh.utils')) for name in sys.modules)
import pytest
status = pytest.main([str(tests), '-q', '-p', 'no:cacheprovider'])
for name, module in tuple(sys.modules.items()):
    if name == 'simesh' or name.startswith('simesh.'):
        path = getattr(module, '__file__', None)
        if path:
            assert Path(path).resolve().is_relative_to(installed), (name, path)
assert not any(name.startswith(('simesh_rewrite', 'simesh.utils', 'simesh.legacy')) for name in sys.modules)
raise SystemExit(status)
'''
    subprocess.run([sys.executable, "-I", "-S", "-c", code, str(args.target.resolve()),
                    sysconfig.get_path("purelib"), str(root/"tests")], check=True,
                   env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                            # Child interpreters started by tests must also load
                            # this wheel before any local editable installation.
                            PYTHONPATH=os.pathsep.join((str(args.target.resolve()),
                                                      sysconfig.get_path("purelib")))))
    result = {"wheel": str(args.wheel), "wheel_bytes": args.wheel.stat().st_size,
              "installed": str(args.target), "isolated_tests_passed": True}
    (args.target.parent/"install-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
