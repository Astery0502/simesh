from pathlib import Path
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT_DIR = Path(__file__).parent
CYTHON_ROOT = ROOT_DIR / "src/simesh/utils/lib"


def _normalize_group(group: str | None) -> str | None:
    if group in (None, "all"):
        return None
    group_dir = CYTHON_ROOT / group
    if not group_dir.is_dir():
        available = ", ".join(["all", *sorted(p.name for p in CYTHON_ROOT.iterdir() if p.is_dir())])
        raise ValueError(f"Unknown group '{group}'. Available groups: {available}")
    return group


def get_extension_sources(group: str | None = None) -> list[Path]:
    normalized = _normalize_group(group)
    source_root = CYTHON_ROOT if normalized is None else CYTHON_ROOT / normalized
    return sorted(source_root.rglob("*.pyx"))


def get_extensions(group: str | None = None) -> list[Extension]:
    include_dirs = [
        str(CYTHON_ROOT),
        np.get_include(),
    ]

    extensions: list[Extension] = []
    for source in get_extension_sources(group):
        module_path = ".".join(source.relative_to(ROOT_DIR / "src").with_suffix("").parts)
        extensions.append(
            Extension(
                module_path,
                [str(source)],
                extra_compile_args=["-O3"],
                include_dirs=include_dirs,
            )
        )
    return extensions


def cythonize_extensions(
    group: str | None = None,
    *,
    profile: bool = False,
    linetrace: bool = False,
):
    extensions = get_extensions(group)
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
            "profile": profile,
            "linetrace": linetrace,
        },
        nthreads=4,
    )


def build(setup_kwargs=None):
    """Poetry build hook: always compile all Cython modules under utils/lib."""
    if setup_kwargs is not None:
        setup_kwargs.update(
            {
                "package_dir": {"": "src"},
                "ext_modules": cythonize_extensions(),
            }
        )
    return setup_kwargs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        help="Specific subdirectory under src/simesh/utils/lib to compile in development mode",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Build extensions in place for development",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable profiling and line tracing directives",
    )
    args = parser.parse_args()

    try:
        ext_modules = cythonize_extensions(
            args.group,
            profile=args.debug,
            linetrace=args.debug,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.inplace:
        setup(
            name="simesh",
            package_dir={"": "src"},
            ext_modules=ext_modules,
            script_args=["build_ext", "--inplace"],
        )
    else:
        setup(
            name="simesh",
            ext_modules=ext_modules,
            script_args=["build"],
        )
