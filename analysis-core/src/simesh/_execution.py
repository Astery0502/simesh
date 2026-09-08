"""One level of portable workers over disjoint, already prepared ranges."""

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager

from ._validation import workers_count


@contextmanager
def worker_context(workers):
    workers_count(workers)
    if workers == 1:
        yield None
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            yield executor


def run_ranges(count, workers, operation, executor=None):
    if not count:
        return
    if executor is None:
        operation(0, count)
        return
    tasks = min(workers, count)
    futures = []
    try:
        for task in range(tasks):
            futures.append(executor.submit(operation, task*count//tasks, (task+1)*count//tasks))
        for future in futures:
            future.result()
    finally:
        # A failed worker must not allow a borrowed input to be reused early.
        wait(futures)


def native_dispatch(backend, schedule, *, build_info=None):
    if backend not in ("threadpool", "openmp") or schedule not in ("static", "dynamic"):
        raise ValueError("choose threadpool/openmp and static/dynamic scheduling")
    if backend == "threadpool":
        return False, int(schedule == "dynamic")
    if build_info is None:
        from ._kernels.native import openmp_build_info as build_info
    if not build_info()["enabled"]:
        raise RuntimeError("OpenMP analysis backend requires an OpenMP build")
    return True, int(schedule == "dynamic")
