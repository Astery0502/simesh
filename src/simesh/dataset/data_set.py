import abc
import os
import numpy as np
from simesh.meshes.mesh import Mesh
from simesh.meshes.amr_mesh import AMRMesh
from simesh.frontends.amrvac.datio import get_single_block_data, find_uniform_fields, write_header, write_forest_tree, write_blocks
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


class AMRDataSet(DataSet):
    """
    AMR (Adaptive Mesh Refinement) specific implementation of DataSet.
    """
    def __init__(self, amr_mesh: AMRMesh, header: dict, forest: np.ndarray, 
                 tree: Tuple, sfile: str="datfile", ):
        """
        Initialize the AMRDataSet with required mesh and file location.
        
        Args:
            mesh (AMRMesh): Instance of AMRMesh class
            header (dict): Header information
            forest (np.ndarray): Forest data
            tree (Tuple): Tree structure
            sfile: Path to data files, defaults to "datfile", "load_data" will load the data from the file
            fieldnames (list[str] | None, optional): List of field names. Defaults to None
        """
        super().__init__(sfile, header['w_names'])

        # assert isinstance(self.mesh, AMRMesh), "mesh must be an instance of AMRMesh class"
        self.mesh = amr_mesh
        self.header = header
        self.forest = forest
        self.tree = tree

        self.amr_forest = amr_mesh.forest

        assert self.fieldnames == self.header['w_names'], "fieldnames must match the number of fields in the header"
        assert self.header['nleafs'] == self.mesh.nleafs, "nleafs must match the number of leafs in the forest"

    def load_data(self, load_ghost: bool = False):

        # if the mesh is uniform, load the uniform grid into the mesh in the mesh.udata

        offsets = self.tree[2]
        with open(self.sfile, 'rb') as fb:
            nw = len(self.fieldnames)

            if self.header['levmax'] == 1 and not load_ghost:
                # load the slab uniform grid into a uniform grid, no incorporation of ghostcells here
                self.mesh.udata = find_uniform_fields(fb, self.header, self.tree)

                block_nx = self.header['block_nx']
                domain_nx = self.header['domain_nx']

                for ileaf in range(self.header['nleafs']):

                    block_idx = self.tree[1][ileaf]
                    x0, y0, z0 = (block_idx-1) * block_nx
                    x1, y1, z1 = block_idx * block_nx

                    self.mesh.data[ileaf][self.mesh.ixMmin[0]:self.mesh.ixMmax[0]+1, 
                                 self.mesh.ixMmin[1]:self.mesh.ixMmax[1]+1, 
                                 self.mesh.ixMmin[2]:self.mesh.ixMmax[2]+1] = \
                        self.mesh.udata[x0:x1, y0:y1, z0:z1]
                return 

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

                field_data = np.array(field_data).reshape(nw, *lixO[::-1])
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

    def write_datfile(self):

        if os.path.exists(self.sfile):
            raise FileExistsError(f"File {self.sfile} already exists")
        with open(self.sfile, 'wb') as fb:
            write_header(fb, self.header) 
            write_forest_tree(fb, self.header, self.forest, self.tree)
            # the non-ghostcells data0
            data0 = self.mesh.data[:,self.mesh.ixMmin[0]:self.mesh.ixMmax[0]+1,self.mesh.ixMmin[1]:self.mesh.ixMmax[1]+1,self.mesh.ixMmin[2]:self.mesh.ixMmax[2]+1,:]
            write_blocks(fb, data0, self.header['ndim'], self.tree[2])

    def update(self):
        # update the mesh with ghostcells
        self.mesh.getbc()

    def update_header(self, **kwargs):
        """
        Modify the header of the dataset
        Note the modifications should be compatible with the header template 
        and other attributes of the dataset
        """
        for key, value in kwargs.items():
            if key in self.header:
                self.header[key] = value
            else:
                raise ValueError(f"Key '{key}' not found in header")
