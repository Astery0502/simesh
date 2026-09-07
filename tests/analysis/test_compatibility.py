"""Retained public workflows and explicit unsupported native geometry scope."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from simesh.amrvac import write_datfile_from_uniform,read_uniform,write_datfile,datfile_to_vtk
from simesh.amrvac.datio import get_metadata
from simesh.analysis import open_source


class CompatibilityTests(unittest.TestCase):
    def test_retained_cartesian_2d_roundtrip_and_bilinear_workflow(self):
        x,y=np.indices((8,8))
        data=np.stack((x+2*y,3*x-y),axis=-1).astype(float)[:,:,None,:]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'two-dimensional.dat'
            write_datfile_from_uniform(path,data,['rho','p'],[0.,0.],[1.,1.],[4,4])
            values=read_uniform(path,resolution=(8,8),ghost_width=2,interpolation='linear')
            np.testing.assert_allclose(values,data,atol=1e-13,rtol=1e-13)
            copied=Path(directory)/'copy.dat'
            write_datfile(path,copied)
            np.testing.assert_array_equal(read_uniform(copied,resolution=(8,8)),data)
            with self.assertRaises(ValueError):
                with open_source(path):
                    pass

    def test_periodic_metadata_copy_and_retained_vtk_values(self):
        x,y,z=np.indices((8,8,8))
        data=(x+10*y+100*z).astype(float)[...,None]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'periodic.dat'
            write_datfile_from_uniform(path,data,['rho'],[0.,0.,0.],[1.,1.,1.],[4,4,4],
                                      periodic=np.array([True,False,False]))
            copied=Path(directory)/'copy.dat'
            write_datfile(path,copied)
            np.testing.assert_array_equal(get_metadata(str(copied))[0]['periodic'],[True,False,False])
            np.testing.assert_array_equal(read_uniform(copied,resolution=(8,8,8)),data)
            with self.assertRaisesRegex(ValueError,'nonperiodic'):
                with open_source(path):
                    pass
            vtk=Path(directory)/'grid.vtk'
            datfile_to_vtk(copied,vtk)
            raw=vtk.read_bytes()
            marker=b'LOOKUP_TABLE default\n'
            offset=raw.index(marker)+len(marker)
            values=np.frombuffer(raw,dtype='>f8',count=512,offset=offset)
            np.testing.assert_array_equal(values,data[...,0].transpose(2,1,0).ravel())


if __name__=='__main__':
    unittest.main()
