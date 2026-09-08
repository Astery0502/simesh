"""Private thread/process providers deliver the same complete independent curl."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from simesh.analysis import open_source, global_curl, sample_plane, Plane
from simesh.amrvac.analysis_execution import global_curl_file
from lfe_001 import write_fixture_dat
from test_prepared import source_fixture


class FileExecutionTests(unittest.TestCase):
    def test_complete_output_task_boundaries_and_admission(self):
        _,fixture=source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.dat'
            write_fixture_dat(path,fixture)
            with open_source(path,field_names=['b1','b2','b3']) as source:
                reference=global_curl(source,batch_size=7)
            for backend in ('thread','process'):
                actual=global_curl_file(path,backend=backend,workers=2,task_size=11,batch_size=7)
                np.testing.assert_array_equal(actual.values,reference.values)
                self.assertFalse(actual.values.flags.writeable)
                lower,upper=actual.mesh.lower,actual.mesh.upper
                extent=upper-lower
                plane=Plane(lower+[0,0,.5*extent[2]],[extent[0],0,0],[0,extent[1],0],(16,16))
                np.testing.assert_array_equal(sample_plane(actual,plane).values,
                                              sample_plane(reference,plane).values)
            sentinel=np.full_like(reference.values,73.)
            with self.assertRaises(MemoryError):
                global_curl_file(path,output=sentinel,budget_bytes=1)
            self.assertTrue(np.all(sentinel==73.))


if __name__=='__main__':
    unittest.main()
