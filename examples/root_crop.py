"""Export a rectangular group of AMR root subtrees without resampling.

Run from the repository root:
    .venv/bin/python examples/root_crop.py --output example-output/root-crop

The new output directory contains a synthetic mixed-level input, equivalent
file/Fields crops, and a separate description of the crop's provenance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import simesh as sm


def run(output):
    output.mkdir(parents=True)
    mesh = sm.mesh_from_forest((4, 3, 2), np.array(([False]+[True]*9)*12),
                              lower=(-2., -3., 1.), upper=(6., 3., 5.), block_shape=(4, 4, 4))
    names = ('rho', 'b1', 'b2', 'b3')
    values = np.empty((mesh.leaf_count, len(names), *mesh.block_shape))
    local = np.indices(mesh.block_shape) + .5
    for leaf in range(mesh.leaf_count):
        x, y, z = (mesh.bounds[leaf, 0, :, None, None, None]
                   + local*mesh.spacing[leaf, :, None, None, None])
        values[leaf] = np.array([10+x, y+z, x+z, x+y])
    header = {
        'datfile_version': 5, 'ndim': 3, 'ndir': 3, 'geometry': 'Cartesian_3D',
        'periodic': [False]*3, 'staggered': False,
        'xmin': mesh.lower, 'xmax': mesh.upper,
        'domain_nx': mesh.root_shape*np.array(mesh.block_shape), 'block_nx': mesh.block_shape,
        'time': 2.5, 'it': 17, 'physics_type': 'mhd',
        'n_par': 1, 'params': [5/3], 'param_names': ['gamma'],
        'snapshotnext': 1, 'slicenext': 0, 'collapsenext': 0,
    }
    original = output/'original.dat'
    with sm.source_from_arrays(mesh, values, names, copy=False) as source:
        sm.write_amrvac(original, sm.read_fields(source), metadata=header)

    box = ((1, 1, 0), (4, 3, 2))
    chosen = ('b3', 'rho')
    direct = output/'direct.dat'
    sm.crop_amrvac(original, direct, root_bounds=box, fields=chosen)

    with sm.open_amrvac(original) as source:
        selection = sm.select_roots(source.mesh, box)
        fields = sm.read_fields(source, chosen, region=selection)
        metadata = source.metadata
    # Owned Fields survive Source closure and retain original Mesh geometry.
    detached = output/'fields.dat'
    sm.write_amrvac(detached, fields, metadata=metadata, root_bounds=box)
    assert direct.read_bytes() == detached.read_bytes()
    before = sm.volume_integral(fields, 'rho')
    with sm.open_amrvac(direct) as source:
        cropped = sm.read_fields(source)
        after = sm.volume_integral(cropped, 'rho')
        np.testing.assert_array_equal(cropped.values.view(np.uint64), fields.values.view(np.uint64))
        np.testing.assert_allclose(after.value, before.value, rtol=1e-14)
        assert source.metadata.time == metadata.time
        print(f'Original/cropped leaves: {mesh.leaf_count}/{source.mesh.leaf_count}')
        print(f'Crop root shape: {tuple(source.mesh.root_shape)}; maximum level: {source.mesh.forest.max_level}')
    description = {
        'source': str(original), 'root_bounds_zero_based_half_open': box,
        'fields': chosen, 'field_units': 'code', 'coordinate_units': 'code',
        'values': 'synthetic cell-centered samples',
        'boundary_policy': 'No original exterior halo; new file edges bound the cropped domain.',
        'restart_certified': False,
    }
    (output/'crop-info.json').write_text(json.dumps(description, indent=2)+'\n')
    print(f'Density volume integral: {after.value:g}; native values and file paths agree')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('example-output/root-crop'))
    run(parser.parse_args().output)
