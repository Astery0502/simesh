"""Vendor and verify AIA RGB controls from a pinned SunPy release (network required)."""
import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '5915092bf6b790bf792b50dd6814f320b809cc61'
BASE = f'https://raw.githubusercontent.com/sunpy/sunpy/{COMMIT}/'
FILES = ['sunpy/visualization/colormaps/color_tables.py',
         'sunpy/visualization/colormaps/data/idl_3.csv', 'LICENSE.rst']


def main():
    payload = {path: urlopen(BASE + path).read() for path in FILES}
    table = np.loadtxt(io.BytesIO(payload[FILES[1]]), delimiter=',')
    # Evaluate only the inspected upstream numerical constructors; the unit
    # multiplier is one because our public selector is integer Angstroms.
    tree = ast.parse(payload[FILES[0]].decode())
    names = {'create_cdict', '_cmap_from_rgb', 'create_aia_wave_dict'}
    functions = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    scope = dict(np=np, colors=colors, u=SimpleNamespace(angstrom=1), get_idl3=lambda: table.copy())
    exec(compile(functions, FILES[0], 'exec'), scope)
    waves = scope['create_aia_wave_dict']()
    arrays = {}
    for wave, channels in waves.items():
        # Normalize each channel before stacking to preserve upstream float32 rounding.
        rgb = np.column_stack([channel / 255.0 for channel in channels]).astype(np.float64)
        original = scope['_cmap_from_rgb'](*channels, f'sdoaia{wave}')
        local = colors.LinearSegmentedColormap.from_list(f'sdoaia{wave}', rgb, N=256)
        np.testing.assert_array_equal(original(np.linspace(0, 1, 4097)), local(np.linspace(0, 1, 4097)))
        arrays[str(wave)] = rgb
    destination = ROOT / 'src/simesh/_data'
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / 'aia_colormaps.npz', **arrays)
    (ROOT / 'LICENSES/SunPy-BSD-2-Clause.txt').write_bytes(payload['LICENSE.rst'])
    manifest = dict(project='SunPy', version='7.0.0', commit=COMMIT, license='BSD-2-Clause',
        ancestry='SSWIDL SDO/AIA aia_lct.pro, Karel Schrijver, 2010-04-12',
        sources=[dict(url=BASE+name, sha256=hashlib.sha256(value).hexdigest()) for name, value in payload.items()],
        representation='256 normalized RGB control points per channel, preserving upstream floating-point values',
        wavelengths_angstrom=sorted(waves),
        verification='4097 scalar samples per channel match the pinned upstream LinearSegmentedColormap exactly')
    (destination / 'aia_colormaps.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Vendored and verified', len(arrays), 'AIA colormaps against SunPy', COMMIT)


if __name__ == '__main__':
    main()
