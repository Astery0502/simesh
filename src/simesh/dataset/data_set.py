import abc
import numpy as np
from simesh.meshes.mesh import Mesh
from simesh.meshes.amr_mesh import AMRMesh
from simesh.frontends.amrvac.datio import get_single_block_data
from typing import Tuple

class DataSet(abc.ABC):
    """
    Abstract base dataset class that serves as a parent for all dataset implementations.
    """
    def __init__(self, sfile: str = "datfile", fieldnames: list[str] | None = None):
        """
        Initialize the DataSet with required mesh and file location.
        
        Args:
            mesh (Mesh): Instance of Mesh class
            sfile (str, optional): Path to data files. Defaults to "datfile"
            fieldnames (list, optional): List of field names. Defaults to None
        """
        self.sfile = sfile
        self.fieldnames = fieldnames if fieldnames is not None else []

    @property
    def sfile(self):
        """Getter for sfile attribute"""
        return self._sfile

    @sfile.setter
    def sfile(self, value):
        """Setter for sfile attribute with type checking"""
        if not isinstance(value, str):
            raise TypeError("sfile must be a string")
        self._sfile = value

    @property
    def fieldnames(self):
        """Getter for fieldnames attribute"""
        return self._fieldnames

    @fieldnames.setter
    def fieldnames(self, value):
        """Setter for fieldnames attribute with type checking"""
        if not isinstance(value, list):
            raise TypeError("fieldnames must be a list")
        self._fieldnames = value

class AMRDataSet(DataSet):
    """
    AMR (Adaptive Mesh Refinement) specific implementation of DataSet.
    """
    def __init__(self, amr_mesh: AMRMesh, sfile: str, header: dict, forest: np.ndarray, 
                 tree: Tuple, fieldnames: list[str] | None = None):
        """
        Initialize the AMRDataSet with required mesh and file location.
        
        Args:
            mesh (AMRMesh): Instance of AMRMesh class
            sfile (str): Path to data files
            header (dict): Header information
            forest (np.ndarray): Forest data
            tree (Tuple): Tree structure
            fieldnames (list[str] | None, optional): List of field names. Defaults to None
        """
        super().__init__(sfile, fieldnames)

        # assert isinstance(self.mesh, AMRMesh), "mesh must be an instance of AMRMesh class"
        self.mesh = amr_mesh
        self.header = header
        self.forest = forest
        self.tree = tree

        self.amr_forest = amr_mesh.forest

        assert self.fieldnames == self.header['w_names'], "fieldnames must match the number of fields in the header"
        assert self.header['nleafs'] == self.mesh.nleafs, "nleafs must match the number of leafs in the forest"

    def load_data(self, load_ghost: bool = True):

        offsets = self.tree[2]
        with open(self.sfile, 'rb') as fb:
            nw = len(self.fieldnames)

            for i in range(self.header['nleafs']):
                offset = offsets[i]
                ghostcells, block_data = get_single_block_data(fb, offset)

                if self.header['staggered']:
                    field_data, staggered_data = block_data
                else:
                    field_data = block_data

                if (np.any(ghostcells)):
                    assert np.all(ghostcells[np.where(ghostcells)] == self.mesh.nghostcells), "ghostcells must match the number of ghostcells in the mesh"

                ixOmin = self.mesh.ixGmin
                ixOmax = self.mesh.ixMmax-self.mesh.ixMmin+ghostcells[1]+ghostcells[0]
                lixO = ixOmax-ixOmin+1

                field_data = np.array(field_data).reshape(nw, *lixO)
                field_data = np.transpose(field_data, (3,2,1,0))

                if not load_ghost:
                    ixRmin = self.mesh.ixMmin
                    ixRmax = self.mesh.ixMmax
                    ixOmin = self.mesh.ixGmin+ghostcells[0]
                    ixOmax = self.mesh.ixGmin+lixO-ghostcells[1]-1
                else:
                    ixRmin = self.mesh.ixMmin - ghostcells[0]
                    ixRmax = self.mesh.ixMmax + ghostcells[1]

                self.mesh.data[i][ixRmin[0]:ixRmax[0]+1, ixRmin[1]:ixRmax[1]+1, ixRmin[2]:ixRmax[2]+1] = \
                    field_data[ixOmin[0]:ixOmax[0]+1, ixOmin[1]:ixOmax[1]+1, ixOmin[2]:ixOmax[2]+1]

        print("Load Clear")

    def update(self):

        self.mesh.getbc()
