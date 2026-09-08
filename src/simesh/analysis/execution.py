"""Small explicit dispatch boundary; caches remain coordinator-owned."""


def native_dispatch(backend, schedule):
    if backend not in ('threadpool','openmp') or schedule not in ('static','dynamic'):
        raise ValueError('choose threadpool/openmp and static/dynamic scheduling')
    if backend == 'threadpool':
        return False, int(schedule=='dynamic')
    from simesh.utils.lib.analysis.native import openmp_build_info
    if not openmp_build_info()['enabled']:
        raise RuntimeError('OpenMP analysis backend requires an OpenMP-enabled analysis build')
    return True, int(schedule=='dynamic')
