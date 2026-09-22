"""Render text-free README panels from retained WENO509 exploratory products."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource, ListedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def canvas(projection=None):
    """Use a common transparent canvas; README Markdown supplies all labels."""
    fig = plt.figure(figsize=(6, 4), facecolor='none')
    ax = fig.add_axes([.025, .025, .95, .95], projection=projection, facecolor='none')
    ax.set_axis_off()
    return fig, ax


def save(fig, output):
    fig.savefig(output, dpi=150, transparent=True)
    plt.close(fig)


def main(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(source/'adaptive/amr-slice.npz') as data:
        blocks = data['blocks']*10
        levels = data['levels'].reshape(len(blocks), -1)
        assert np.all(levels == levels[:, :1])
        levels = levels[:, 0]
        lower, upper = data['lower']*10, data['upper']*10
    fig, ax = canvas()
    polys = np.stack([blocks[:, [0,2]], blocks[:, [1,2]],
                      blocks[:, [1,3]], blocks[:, [0,3]]], axis=1)
    ax.add_collection(PolyCollection(polys, array=levels,
                      cmap=ListedColormap(['#183b58', '#246978', '#46aaa7']),
                      norm=Normalize(levels.min(), levels.max()),
                      edgecolors='#6de0df', linewidths=.28))
    ax.set(xlim=(lower[1], upper[1]), ylim=(lower[2], upper[2]), aspect='equal')
    save(fig, output_dir/'adaptive-mesh.png')

    with np.load(source/'isosurface-j.npz') as data:
        vertices, faces, level = data['vertices']*10, data['faces'], float(data['level'])
    fig, ax = canvas(projection='3d')
    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors='#ed9b53',
        linewidths=0, shade=True, lightsource=LightSource(315, 40), rasterized=True))
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    ax.set_box_aspect(upper-lower, zoom=1.3)
    ax.view_init(elev=30, azim=45)
    ax.set_proj_type('ortho')
    save(fig, output_dir/'current-structure.png')

    summary = json.loads((source/'summary.json').read_text())
    with np.load(source/'los-oblique.npz') as data:
        values = np.ma.masked_where(~data['valid'], data['image']).T
        extent = data['extent']*10
    fig, ax = canvas()
    palette = plt.get_cmap('inferno').copy()
    # Only invalid pixels are transparent; valid low intensities keep their color.
    palette.set_bad((0, 0, 0, 0))
    ax.imshow(values, origin='lower', extent=extent, cmap=palette,
              norm=Normalize(0, summary['rendering']['los_color_limits'][1]), interpolation='nearest')
    save(fig, output_dir/'synthetic-emission.png')

    files = ['adaptive/amr-slice.npz', 'isosurface-j.npz', 'los-oblique.npz', 'summary.json']
    manifest = dict(inputs={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
        source_figures=['weno509-slices-amr.png: AMR leaf-block structure',
                        'weno509-emission-isosurfaces-lines.png: current isosurface and oblique emission'],
        panels=['adaptive-mesh.png', 'current-structure.png', 'synthetic-emission.png'],
        presentation='Transparent panels without text or decorative backgrounds; Markdown provides labels',
        length_unit_Mm=10, current_isosurface_A_m2=level,
        isosurface_method='Native curl resampled to a uniform volume, then marching cubes',
        emission_model='Isothermal 1 MK, optically thin AIA 171; illustrative solar normalization',
        camera=dict(azimuth_deg=45, elevation_deg=30),
        description='Existing non-manuscript exploratory panels replotted for the README')
    (output_dir/'amr-showcase.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/assets/readme'))
    args = parser.parse_args()
    main(args.source, args.output_dir)
