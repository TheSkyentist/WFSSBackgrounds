#! /usr/bin/env python

# Import packages
import glob
import numpy as np
from itertools import product
from matplotlib import pyplot
from multiprocessing import Pool

# Import astropy functions
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

# Silence warnings
import warnings

warnings.filterwarnings('ignore')


# Get median
def get_median(gf, ext):

    # # For Testing
    # if ext == '_crds':
    #     return np.ones((2, 2040))
    # else:
    #     return np.zeros((2, 2040))

    # Get subbed products
    subbed = np.load(f'{gf}{ext}/subbed.npy')

    # Get masked products
    masks = np.load(f'{gf}{ext}/masks.npy')

    # Get background strenghts
    if (ext == '_crds') and ('CLEAR' in gf):
        heights = np.reshape(
            [np.genfromtxt(f)[1] for f in sorted(glob.glob(f'{gf}_custom/*.info'))],
            (len(masks), 1, 1),
        )
        subbed -= heights
    else:
        heights = np.reshape(
            [np.genfromtxt(f)[1] for f in sorted(glob.glob(f'{gf}{ext}/*.info'))],
            (len(masks), 1, 1),
        )

    # Iterate over dispersion direction
    return np.array(
        [
            sigma_clipped_stats(
                (subbed / heights), sigma=1, mask=masks, axis=(0, k + 1)
            )[1]
            * 100
            for k in range(2)
        ]
    )


# Get medians
def get_medians(gf):
    N = len(Table.read(f'{gf}/{gf}.fits'))
    return N, {ext: get_median(gf, ext) for ext in ['_custom', '_crds']}


# Filter grism combos
gfs = [
    '-'.join(p)
    for p in product(['CLEAR', 'GR150C', 'GR150R'], ['F115W', 'F150W', 'F200W'])
]

# Colors for CRDS and custom
colors = ['#E20134', '#008DF9']

# Percentile positions
xs = np.arange(-2, 3) * 0.175 + 0.5
labels = [r'min', r'$-2\sigma$', r'$\tilde r$', r'$+2\sigma$', r'max']

# Get medians for each filter grism combo
with Pool(len(gfs)) as p:
    result = p.map(get_medians, gfs)
medians = {gf: result[i] for i, gf in enumerate(gfs)}

# Create figure
size = 4
fig = pyplot.figure(figsize=(0.95 * size * 3, 3 * size))
gs = fig.add_gridspec(3, 3, hspace=0.1, wspace=0)

# Iterate over grisms
for j, grism in enumerate(['CLEAR', 'GR150C', 'GR150R']):
    # Set y-label
    ax = fig.add_subplot(gs[j, 0])
    ax.text(
        -0.175,
        0.5,
        grism,
        fontsize=25,
        ha='center',
        va='center',
        transform=ax.transAxes,
        rotation=90,
    )
    ax.axis('off')

    # Iterate over filters
    for i, filt in enumerate(['F115W', 'F150W', 'F200W']):
        # Get grism and filter
        gf = f'{grism}-{filt}'

        # Iterate over axes
        for k in range(2):
            # Get subgridspec
            subgs = gs[j, i].subgridspec(2, 1, hspace=0)

            # Get axis
            ax = fig.add_subplot(subgs[k])

            # Set axis parameters
            ax.set(
                xlim=(0, 2048),
                ylim=(-5, 5),
                xticks=[0, 500, 1000, 1500, 2000],
                yticks=[-5, 0, 5],
            )
            if j == 2 and k == 1:
                ax.set(xticklabels=[0, 500, 1000, 1500, ''])
            else:
                ax.set(xticklabels=[])
            if i == 0:
                ax.set(yticklabels=[-5, 0, 5])
            else:
                ax.set(yticklabels=[])
            if i == 2:
                label = 'Row' if k else 'Column'
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(label)

            # Set font parameters
            font_args = dict(
                transform=ax.transAxes, fontsize=13, ha='center', va='center'
            )

            # Plot percentile labels
            for x, label in zip(xs, labels):
                ax.text(x, 0.375, label, **font_args)

            # Get median
            N, medians_dict = medians[gf]

            # Iterate over CRDS and custom
            for m, ext in enumerate(['_custom', '_crds']):
                # Get median
                median = medians_dict[ext][k]

                # Plot
                ax.plot(
                    np.arange(4, 2044),
                    median,
                    color=colors[m],
                    ds='steps-mid',
                    alpha=0.75,
                    solid_joinstyle='miter',
                    linewidth=1,
                )

                # Get median and rms (rounded strings)
                percentiles = [
                    rf'${p:.2f}$' for p in np.nanpercentile(median, [0, 5, 50, 95, 100])
                ]

                # List sigma range
                for x, percentile in zip(xs, percentiles):
                    y = 0.125 * (m + 1)
                    ax.text(x, y, percentile, c=colors[m], alpha=0.75, **font_args)

                # Set title
                if j == 0 and k == 0:
                    ax.set_title(f'{filt}', fontsize=25)

# Set overall labels
fig.supxlabel(r'$\textbf{Distance [px]}$', y=0.05, fontsize=25)
fig.supylabel(r'$\textbf{Median Residual (Percent of Background)}$', fontsize=25)

# Set overall legend
fig.legend(
    handles=[
        pyplot.Line2D(
            [],
            [],
            color=colors[i],
            markeredgecolor='none',
            label=ext,
            alpha=0.75,
            ls='none',
            marker='o',
            ms=30,
        )
        for i, ext in enumerate([r'\textbf{Emperical}', r'\textbf{CRDS}'])
    ],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.965),
    fontsize=25,
    ncols=2,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.1,
)

# Save figure
fig.savefig('compareSub.pdf')
pyplot.close(fig)
