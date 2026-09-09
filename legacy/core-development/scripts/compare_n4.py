"""One isolated full-domain ordinary-field export and reopen comparison."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def array_hash(array):
    array = np.ascontiguousarray(array)
    result = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    result.update(memoryview(array).cast('B'))
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--flavor', choices=('donor', 'new'), required=True)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError('run with -I -S')
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(args.source_root.resolve()), str(args.dependencies.resolve())]
    global np
    import numpy as np
    start = time.perf_counter()
    import simesh
    from simesh.amrvac import open_dataset
    from simesh.amrvac.datio import get_metadata
    assert Path(simesh.__file__).resolve().is_relative_to(args.source_root.resolve())
    imported = time.perf_counter()
    names = ['b3', 'rho']
    limit = 2*1024**3
    scratch = args.output.with_suffix('.dat')
    if scratch.exists():
        raise FileExistsError(scratch)
    try:
        if args.flavor == 'donor':
            dataset = open_dataset(str(args.file))
            ids = [dataset.wnames.index(name) for name in names]
            dataset.load_data(field_indices=ids)
            source_staggered = bool(dataset.metadata['staggered'])
            # This profile explicitly exports ordinary values, never CT faces.
            dataset.metadata = dict(dataset.metadata, staggered=False)
            loaded = time.perf_counter()
            header = dataset.write_datfile(str(scratch))
            saved = time.perf_counter()
            expected = array_hash(dataset.blocks())
            hashed = time.perf_counter()
            del dataset
        else:
            metadata = get_metadata(args.file)[0]
            source_staggered = bool(metadata['staggered'])
            with simesh.open_amrvac(args.file, memory_limit=limit) as source:
                fields = simesh.read_fields(source, names, memory_limit=limit)
            loaded = time.perf_counter()
            header = simesh.write_amrvac(scratch, fields, metadata=metadata, memory_limit=limit)
            saved = time.perf_counter()
            expected = array_hash(np.moveaxis(fields.values, -1, 1))
            hashed = time.perf_counter()
            del fields
        gc.collect()
        reopened = open_dataset(str(scratch))
        reopened.load_data()
        finished = time.perf_counter()
        actual = array_hash(reopened.blocks())
        assert actual == expected
        assert reopened.wnames == names
        assert not reopened.metadata['staggered']
        digest = hashlib.sha256()
        with scratch.open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024**2), b''):
                digest.update(block)
        record = dict(flavor=args.flavor, import_seconds=imported-start,
                      read_seconds=loaded-imported, export_seconds=saved-loaded,
                      reopen_seconds=finished-hashed, hash_seconds=hashed-saved,
                      startup_to_result_seconds=finished-start-(hashed-saved),
                      input_staggered=source_staggered, exported_fields=names,
                      leaf_count=int(header['nleafs']), file_bytes=scratch.stat().st_size,
                      value_hash=actual, file_hash=digest.hexdigest(),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      ordinary_only=True, roundtrip_exact=True)
        args.output.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    finally:
        if scratch.exists():
            scratch.unlink()


if __name__ == '__main__':
    main()
