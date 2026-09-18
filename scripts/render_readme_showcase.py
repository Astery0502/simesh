"""Compose the README showcase from retained WENO509 exploratory products."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource, ListedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def main(source, output):
    background, card, muted = '#0b1220', '#111e30', '#9cb0c6'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': '#eef5ff', 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig = plt.figure(figsize=(15, 6), facecolor=background)
    fig.text(.035, .91, 'Explore the simulation. Keep the adaptive mesh.',
             fontsize=23, weight='bold')
    fig.text(.035, .855, 'Native AMR geometry  /  Three-dimensional structures  /  Synthetic views',
             fontsize=11, color=muted)
    starts = [.025, .35, .675]
    titles = ['Adaptive mesh', 'Current structure', 'Synthetic emission']
    accents = ['#54d4cd', '#ffba76', '#f293c4']
    for i, (x, title, color) in enumerate(zip(starts, titles, accents)):
        fig.add_artist(FancyBboxPatch((x, .135), .30, .65, boxstyle='round,pad=0.008,rounding_size=0.015',
                       transform=fig.transFigure, facecolor=card, edgecolor='#22364c', zorder=0))
        fig.text(x+.014, .732, f'0{i+1}', color=color, fontsize=10, weight='bold')
        fig.text(x+.051, .73, title, fontsize=15, weight='bold')
    with np.load(source/'adaptive/amr-slice.npz') as data:
        blocks = data['blocks']*10
        levels = data['levels'].reshape(len(blocks), -1)
        assert np.all(levels == levels[:, :1])
        levels = levels[:, 0]
        lower, upper = data['lower']*10, data['upper']*10
    ax = fig.add_axes([.055, .275, .245, .385], facecolor=card)
    polys = np.stack([blocks[:, [0,2]], blocks[:, [1,2]],
                      blocks[:, [1,3]], blocks[:, [0,3]]], axis=1)
    ax.add_collection(PolyCollection(polys, array=levels,
                      cmap=ListedColormap(['#183b58', '#246978', '#46aaa7']),
                      norm=Normalize(levels.min(), levels.max()),
                      edgecolors='#6de0df', linewidths=.28))
    ax.set(xlim=(lower[1], upper[1]), ylim=(lower[2], upper[2]),
           aspect='equal', xlabel='y [Mm]', ylabel='z [Mm]')
    ax.tick_params(labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_color('#3a5268')
    fig.text(.039, .19, f'{len(blocks):,} native leaf blocks  ·  x = 0', color=muted, fontsize=10)
    with np.load(source/'isosurface-j.npz') as data:
        vertices, faces, level = data['vertices']*10, data['faces'], float(data['level'])
    ax = fig.add_axes([.36, .235, .28, .455], projection='3d', facecolor=card)
    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors='#ed9b53',
        linewidths=0, shade=True, lightsource=LightSource(315, 40), rasterized=True))
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
    ax.set_box_aspect(upper-lower, zoom=1.15)
    ax.view_init(elev=30, azim=45)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    fig.text(.364, .19, r'$|J|$ isosurface  ·  99th percentile', color=muted, fontsize=10)
    summary = json.loads((source/'summary.json').read_text())
    with np.load(source/'los-oblique.npz') as data:
        values = np.ma.masked_where(~data['valid'], data['image']).T
        extent = data['extent']*10
    ax = fig.add_axes([.703, .265, .245, .42], facecolor=card)
    palette = plt.get_cmap('inferno').copy()
    palette.set_bad(card)
    ax.imshow(values, origin='lower', extent=extent, cmap=palette,
              norm=Normalize(0, summary['rendering']['los_color_limits'][1]), interpolation='nearest')
    ax.set_axis_off()
    fig.text(.689, .19, 'AIA 171 Å  ·  Oblique view  ·  1 MK', color=muted, fontsize=10)
    fig.text(.035, .06, 'WENO509 exploratory results  •  One dataset, complementary analysis geometries',
             fontsize=10, color=muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=background)
    plt.close(fig)
    files = ['adaptive/amr-slice.npz', 'isosurface-j.npz', 'los-oblique.npz', 'summary.json']
    manifest = dict(inputs={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
        source_figures=['weno509-slices-amr.png: AMR leaf-block structure',
                        'weno509-emission-isosurfaces-lines.png: current isosurface and oblique emission'],
        length_unit_Mm=10, current_isosurface_A_m2=level,
        isosurface_method='Native curl resampled to a uniform volume, then marching cubes',
        emission_model='Isothermal 1 MK, optically thin AIA 171; illustrative solar normalization',
        camera=dict(azimuth_deg=45, elevation_deg=30),
        description='Existing non-manuscript exploratory panels replotted as a README composition')
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/assets/readme/amr-showcase.png'))
    args = parser.parse_args()
    main(args.source, args.output)
