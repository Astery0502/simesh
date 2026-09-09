def openmp_enabled() -> bool:
    """Return True when the AMR Cython extension was built with OpenMP."""
    try:
        from simesh.utils.lib.amr.mesh import openmp_enabled as _openmp_enabled
    except ImportError:
        return False

    return bool(_openmp_enabled())


def openmp_build_info() -> dict[str, bool | int | str]:
    """Return build-time OpenMP status for the compiled AMR extension."""
    try:
        from simesh.utils.lib.amr.mesh import openmp_build_info as _openmp_build_info
    except ImportError as exc:
        return {
            "enabled": False,
            "openmp_version": 0,
            "available": False,
            "error": str(exc),
        }

    info = dict(_openmp_build_info())
    info["available"] = True
    return info
