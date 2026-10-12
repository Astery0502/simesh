"""Plot recorded enine measurements without rerunning the benchmark."""
from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _paths import data_directory, OUTPUT

DATA = data_directory() / 'figure4'
OUT = OUTPUT
REPS = ('amr', 'l5', 'l6')
WORKERS = np.array([1, 4, 8, 16, 32])
COLORS = ('#0072B2', '#D55E00', '#009E73')
LABELS = ('Native AMR', 'Uniform L5', 'Uniform L6')


def measured(path):
    data = json.loads(path.read_text())
    calls = [c for c in data['calls'] if c['repeat'] >= 0]
    times = [c['wall_seconds'] for c in calls]
    return (statistics.median(times), min(times), max(times),
            max(c['rss_peak_observed_call'] for c in calls) / 2**30)


endpoints = {rep: np.array([measured(DATA / f'results/{rep}-endpoints-w{w}.json')
                            for w in WORKERS]) for rep in REPS}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={'width_ratios': [1, 1]})
fig.subplots_adjust(left=.085, right=.91, top=.965, bottom=.17, wspace=.38)
ax = axes[0]
ax.plot(WORKERS, WORKERS, color='#888888', linestyle='--', linewidth=1.2, label='Ideal scaling')
for rep, color, label, marker, size, style in zip(
    REPS, COLORS, LABELS, ('o', 's', None), (6, 4.2, 0), ('-', '-', (0, (5, 3)))
):
    m = endpoints[rep]
    speed = m[0, 0] / m[:, 0]
    error = np.stack((speed - m[0, 0] / m[:, 2], m[0, 0] / m[:, 1] - speed))
    ax.errorbar(WORKERS, speed, yerr=error, color=color, marker=marker, markersize=size,
                markerfacecolor='white', markeredgewidth=.95, linestyle=style,
                linewidth=1.6, capsize=3, label=label)
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xticks(WORKERS, [str(w) for w in WORKERS])
ax.set_yticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
ax.set_ylim(.85, 48)
ax.set_xlabel('Workers')
ax.set_ylabel('Speedup over 1 worker')
ax.legend(fontsize=11, loc='lower right', frameon=False)

x = np.arange(3)
width = .34
ax = axes[1]
rss_axis = ax.twinx()
measures = np.array([endpoints[rep][-1] for rep in REPS])
times = measures[:, 0]
errors = np.stack((times-measures[:, 1], measures[:, 2]-times))
time_color, rss_color = '#577E9F', '#D8A451'
time_bars = ax.bar(x-width/2, times, width, color=time_color, label='Wall time',
                   yerr=errors, capsize=3, error_kw={'linewidth': 1})
rss_bars = rss_axis.bar(x+width/2, measures[:, 3], width,
                       color=rss_color, label='Peak RSS')
ax.set_xticks(x, ['Native AMR', 'Uniform L5', 'Uniform L6'])
ax.set_ylabel('Wall time [s]', color=time_color)
rss_axis.set_ylabel('Peak process RSS [GiB]', color='#9B6F22', fontsize=13)
ax.set_ylim(0, 3)
rss_axis.set_ylim(0, 2)
ax.set_yticks(np.arange(0, 3.1, .5))
rss_axis.set_yticks(np.arange(0, 2.1, .5))
ax.set_axisbelow(True)
rss_axis.spines['right'].set_visible(True)
rss_axis.spines['right'].set_color('#9B6F22')
rss_axis.spines[['left', 'bottom']].set_visible(False)
rss_axis.tick_params(axis='y', labelsize=11, colors='#9B6F22')
ax.legend([time_bars, rss_bars], ['Wall time', 'Peak RSS'],
          frameon=False, fontsize=11, loc='upper right')
for ax, label in zip(axes, ('(a)', '(b)')):
    ax.spines[['left', 'bottom']].set_color('#7a8790')
    ax.grid(axis='y', color='#a1adb5', linewidth=.5, alpha=.25)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontsize(13)
    ax.text(.025, .96, label, transform=ax.transAxes,
            ha='left', va='top', fontsize=17,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.7, pad=.8), zorder=9)

axes[1].spines['left'].set_color(time_color)
axes[1].tick_params(axis='y', colors=time_color)

for extension in ('png', 'pdf', 'svg'):
    path = OUT / f'figure4-performance.{extension}'
    fig.savefig(path, dpi=220, facecolor='white')
    print(path)
plt.close(fig)
