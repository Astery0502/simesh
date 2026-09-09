"""Export AMRVAC fields to NumPy memory maps or uniform binary VTK."""

import argparse
from pathlib import Path
import numpy as np
import simesh as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='New NumPy output directory or destination .vtk file.')
    parser.add_argument('--format', choices=('numpy', 'vtk'), default='numpy')
    parser.add_argument('--resolution', nargs=3, type=int, required=True)
    parser.add_argument('--fields', nargs='+')
    parser.add_argument('--interpolation', choices=('zero', 'native', 'linear'), default='zero')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--scheme', choices=('exact-phase',))
    parser.add_argument('--support-capacity', type=int, default=128)
    parser.add_argument('--tile-rows', type=int, default=64)
    args = parser.parse_args()
    if (args.interpolation == "linear") != (args.scheme is not None):
        parser.error("Use --scheme exact-phase exactly when --interpolation is linear.")
    shape = tuple(args.resolution)
    options = dict(fields=args.fields, interpolation=args.interpolation,
                   batch_size=args.batch_size, scheme=args.scheme,
                   support_capacity=args.support_capacity, tile_rows=args.tile_rows)
    if args.format == 'vtk':
        path = sm.export_uniform_vtk(args.snapshot, args.output, shape, **options)
        print(f'Exported {shape} cells to {path}')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    with sm.open_amrvac(args.snapshot) as source:
        count = len(source.field_ids(args.fields))
        values = np.lib.format.open_memmap(args.output/'values.npy', mode='w+',
                                           dtype=np.float64, shape=(*shape, count))
        valid = np.lib.format.open_memmap(args.output/'valid.npy', mode='w+',
                                          dtype=np.bool_, shape=shape)
        result = sm.export_uniform(source, shape, output=(values, valid), **options)
        values.flush()
        valid.flush()
        np.savez(args.output/'geometry.npz', lower=result.lower, upper=result.upper,
                 names=[f.name for f in result.definitions],
                 units=[f.units for f in result.definitions],
                 interpretations=[f.interpretation for f in result.definitions],
                 interpolation=args.interpolation, scheme=args.scheme or "interior")
    print(f'Exported {shape} cells to {args.output}')


if __name__ == '__main__':
    main()
