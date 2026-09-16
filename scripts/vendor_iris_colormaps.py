"""Vendor IRIS spectral FUV/NUV palettes from pinned SunPy (network required)."""
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '5915092bf6b790bf792b50dd6814f320b809cc61'
URL = f'https://raw.githubusercontent.com/sunpy/sunpy/{COMMIT}/sunpy/visualization/colormaps/color_tables.py'


def main():
    source = urlopen(URL).read()
    tree = ast.parse(source.decode())
    names = {'create_cdict', '_cmap_from_rgb', 'iris_sji_color_table'}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np, colors=colors)
    exec(compile(tree, URL, 'exec'), scope)
    tables = {}
    for channel in ('FUV', 'NUV'):
        upstream = scope['iris_sji_color_table'](channel)
        rgb = np.column_stack([np.array(upstream._segmentdata[name])[:, 1] for name in ('red', 'green', 'blue')])
        local = colors.LinearSegmentedColormap.from_list(f'irissji{channel}', rgb, N=256)
        np.testing.assert_array_equal(local(np.linspace(0, 1, 4097)), upstream(np.linspace(0, 1, 4097)))
        tables[channel] = rgb
    out = ROOT / 'src/simesh/_data'
    np.savez_compressed(out / 'iris_colormaps.npz', **tables)
    (out / 'iris_colormaps.json').write_text(json.dumps(dict(project='SunPy',version='7.0.0',commit=COMMIT,
        license='BSD-2-Clause',source_url=URL,source_sha256=hashlib.sha256(source).hexdigest(),
        channels=list(tables),convention='IRIS spectral FUV/NUV colors, also exposed under irissjiFUV/irissjiNUV',
        documentation='https://irispy.readthedocs.io/en/stable/tutorial/data_idiosyncrasies.html',
        verification='4097 scalar samples per channel equal the upstream colormap'),indent=2)+'\n')
    print('Verified and saved IRIS FUV/NUV colormaps')


if __name__ == '__main__':
    main()
