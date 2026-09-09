"""Small explicit dispatch boundary; caches remain coordinator-owned."""


def native_dispatch(backend, schedule, *, build_info=None):
    """Choose one parallel layer; auto uses the consumer's compiled capability."""
    if backend not in ('auto','threadpool','openmp') or schedule not in ('static','dynamic'):
        raise ValueError('choose auto/threadpool/openmp and static/dynamic scheduling')
    if backend == 'threadpool':
        return False, int(schedule=='dynamic')
    if build_info is None:
        from simesh.utils.lib.analysis.native import openmp_build_info as build_info
    enabled = build_info()['enabled']
    if backend == 'auto':
        return enabled, int(schedule=='dynamic')
    if not enabled:
        raise RuntimeError('OpenMP analysis backend requires an OpenMP-enabled analysis build')
    return True, int(schedule=='dynamic')
