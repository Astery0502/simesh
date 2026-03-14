from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import setup


def _load_build_helper():
    build_path = Path(__file__).with_name("build.py")
    spec = spec_from_file_location("simesh_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load build helper from {build_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup(
    **_load_build_helper().get_setup_kwargs(),
)
