import abc


class DataSet(abc.ABC):
    """Minimal base class for canonical AMRVAC datasets."""

    def __init__(self, sfile: str):
        self.sfile = sfile
        self.load_metadata()
        self.data = None

    def load_metadata(self):
        raise NotImplementedError("load_metadata must be implemented in the subclass")

    def print_metadata(self):
        print("====basic metadata=====")
        print(f"Data file: {self.sfile}")
        print(f"Number of dimensions: {self.ndim}")
        print(f"Physical domain: from {self.physical_domain[0]} to {self.physical_domain[1]}")
        print(f"Domain size: {self.domain_nx}")
        print(f"Field names: {self.wnames}")
        print(f"Geometry: {self.geometry}")
        self.print_metadata_impl()

    def print_metadata_impl(self):
        pass

    def load_data(self):
        raise NotImplementedError("load_data must be implemented in the subclass")
