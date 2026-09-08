"""Render measured thermal images; label manufactured physical inputs explicitly."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    root=Path('benchmark-results/analysis-core')
    data=np.load(root/'thermal-500-images.npz')
    keys=('axis','oblique','diagonal')
    labels=('Along z','20 degree oblique','Diagonal view')
    positive=np.concatenate([data[k][data[k]>0] for k in keys])
    low=max(float(np.percentile(positive,1)),1e-10)
    high=float(positive.max())
    fig,axes=plt.subplots(1,3,figsize=(13,4.9),layout='constrained',facecolor='#11151d')
    for ax,key,label in zip(axes,keys,labels):
        ax.set_facecolor('#11151d')
        values=np.ma.masked_less_equal(data[key].T,0.)
        im=ax.imshow(values,origin='lower',cmap='inferno',norm=LogNorm(low,high),interpolation='nearest')
        ax.set_title(label+' | 500 x 500 rays',color='white',fontsize=11,pad=10)
        ax.set_xlabel('Image pixel',color='#c1c8d6')
        ax.tick_params(colors='#c1c8d6',labelsize=8)
        for spine in ax.spines.values():spine.set_color('#394457')
    axes[0].set_ylabel('Image pixel',color='#c1c8d6')
    colorbar=fig.colorbar(im,ax=axes,shrink=.76,pad=.02)
    colorbar.set_label('DN / s / AIA pixel',color='#c1c8d6')
    colorbar.ax.tick_params(colors='#c1c8d6',labelsize=8)
    fig.suptitle('Historical AIA 171 | WENO AMR density + manufactured temperature\n'
                 'Demonstration normalization; not a physically validated snapshot image',
                 color='white',fontsize=13)
    fig.savefig(root/'thermal-500-projections.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(root/'thermal-500-projections.png')

if __name__=='__main__':main()
