import abc
import os
import numpy as np

class DataSet(abc.ABC):
    """
    Abstract base dataset class that serves as a parent for all dataset implementations.
    """
    def __init__(self, sfile: str):
        """
        Initialize the DataSet with required mesh and file location.
        
        Args:
            sfile (str): Path to data files.
        """
        self.sfile = sfile
        self.load_metadata()

        self.data = None  # Initialize data attribute
        self.field_indices = None  # Initialize field_indices attribute

    
    def load_metadata(self):
        """
        Load the metadata from the data file.
        implemented in the subclass
        will load in at least: physical domain; domain size; field names
        """
        raise NotImplementedError("load_metadata must be implemented in the subclass")

    def print_metadata(self):
        """
        Print the metadata in a readable format
        """
        print("====basic metadata=====")
        print(f"Data file: {self.sfile}")
        print(f"Number of dimensions: {self.ndim}")
        print(f"Physical domain: from {self.physical_domain[0]} to {self.physical_domain[1]}")
        print(f"Domain size: {self.domain_nx}")
        print(f"Field names: {self.wnames}")
        print(f"Geometry: {self.geometry}")

        self.print_metadata_impl()

    def print_metadata_impl(self):
        """
        Print the metadata in a readable format
        implemented in the subclass
        will print additional metadata if needed
        """
        pass

    def load_data(self):
        """
        Load the data from the data file.
        implemented in the subclass
        will load in at least: data
        """
        raise NotImplementedError("load_data must be implemented in the subclass")
        