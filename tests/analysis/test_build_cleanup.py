"""Cleanup must never recurse into the user's virtualenv or result artifacts."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


class CleanupTests(unittest.TestCase):
    def test_only_owned_generated_artifacts_are_removed(self):
        path=Path(__file__).resolve().parents[2]/'scripts/build_ext.py'
        spec=importlib.util.spec_from_file_location('build_helper_under_test',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            keep=['src/simesh/utils/lib/analysis/native.pyx',
                  '.venv/lib/numpy/_multiarray.so','benchmark-results/user-data.npy',
                  'reference/handwritten.c']
            remove=['src/simesh/utils/lib/analysis/native.c',
                    'src/simesh/utils/lib/analysis/native.cpython-311-darwin.so',
                    'build/generated.o','src/simesh.egg-info/SOURCES.txt']
            for name in keep+remove:
                file=root/name
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_bytes(b'preserve or remove exactly as declared')
            module.clean_build_artifacts(root)
            self.assertTrue(all((root/name).exists() for name in keep))
            self.assertTrue(all(not (root/name).exists() for name in remove))


if __name__=='__main__':
    unittest.main()
