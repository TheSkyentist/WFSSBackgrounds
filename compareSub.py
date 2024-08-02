#! /usr/bin/env python

# Import packages
import glob
import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool

# Import astropy functions
from astropy.stats import sigma_clipped_stats

# Silence warnings
import warnings

warnings.filterwarnings('ignore')


def makeFig(grism):
    # Create figure
    size = 4
    fig = pyplot.figure(figsize=(0.95 * size * 3, 2 * size))
    gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0)

    # Iterate over filts
    for i, filt in enumerate(['F115W', 'F150W', 'F200W']):
        # Get grism and filter
        gf = f'{grism}-{filt}'

        # Iterate over extensions
        for j, ext in enumerate(['_custom', '_crds']):
            # Get subbed products
            subbed = np.load(f'{gf}{ext}/subbed.npy')

            # Get masked products
            masks = np.load(f'{gf}{ext}/masks.npy')

            # Get background strenghts
            if (ext == '_crds') and (grism == 'CLEAR'):
                heights = np.reshape(
                    [
                        np.genfromtxt(f)[1]
                        for f in sorted(glob.glob(f'{gf}_custom/*.info'))
                    ],
                    (len(masks), 1, 1),
                )
                subbed -= heights
            else:
                heights = np.reshape(
                    [
                        np.genfromtxt(f)[1]
                        for f in sorted(glob.glob(f'{gf}{ext}/*.info'))
                    ],
                    (len(masks), 1, 1),
                )

            print(f'Loaded data for {gf}{ext}')

            # Get subgridspec
            subgs = gs[j, i].subgridspec(2, 1, hspace=0)

            # Iterate over dispersion direction
            for k in range(2):
                # Get axis
                axis = k + 1  # (i + k) % 2 + 1

                # Get summary statistics
                # medians = np.nanmedian((subbed / heights), axis=axis)
                _, medians, _ = sigma_clipped_stats(
                    (subbed / heights), sigma=1, mask=masks, axis=axis
                )
                medians *= 100  # Convert to percent

                # Get summary of summary statistics
                # median = np.nanmedian(medians,axis=0)
                # stddev = np.nanstd(medians,axis=0)
                _, median, stddev = sigma_clipped_stats(medians, sigma=1, axis=0)

                # Plot
                ax = fig.add_subplot(subgs[k])
                ax.plot(
                    np.arange(4, 2044),
                    median,
                    color='k',
                    ds='steps-mid',
                    alpha=1,
                    solid_joinstyle='miter',
                    linewidth=1,
                )
                ax.set(
                    xlim=(0, 2048),
                    ylim=(-5, 5),
                    xticks=[0, 500, 1000, 1500, 2000],
                    xticklabels=[0, 500, 1000, 1500, ''],
                )

                # Get median and rms (rounded strings)
                labels = [
                    r'$-2\sigma$',
                    r'$-1\sigma$',
                    r'$\mu$',
                    r'$+1\sigma$',
                    r'$+2\sigma$',
                ]
                percentiles = [
                    fr'${p:.2f}$' for p in np.nanpercentile(median, [5, 16, 50, 84, 95])
                ]

                # List sigma range
                xs = np.arange(-2, 3) * 0.15 + 0.5
                t, a = ax.transAxes, 'center'
                for x, label, percentile in zip(xs, labels, percentiles):
                    ax.text(x, 0.25, label, transform=t, fontsize=13, ha=a, va=a)
                    ax.text(x, 0.125, percentile, transform=t, fontsize=13, ha=a, va=a)

                # Set title
                if j == 0 and k == 0:
                    ax.set_title(f'{filt} $(N={len(masks)})$', fontsize=25, y=1.025)
                    # ax.set_title(f'{filt} $(N=)$',fontsize=25,y=1.025)

                # Set y-label
                if i != 0:
                    ax.set_yticklabels([])
                if i == 2:
                    label = 'Row' if k else 'Column'
                    # Set label on right side
                    ax.set_ylabel(label)
                    ax.yaxis.set_label_position('right')

                if k != 1 or j != 1:
                    ax.set(xticklabels=[])

    # Create titles
    fig.suptitle(r'\textbf{' + grism + '}', fontsize=30)
    fig.supxlabel(r'$\textbf{Distance [px]}$', fontsize=25)

    # Set overall labels
    ax = fig.add_subplot(gs[:, :])
    ax.text(
        -0.12,
        0.5,
        r'$\textbf{Residual (Percent of Background)}$',
        fontsize=25,
        ha='center',
        va='center',
        transform=ax.transAxes,
        rotation=90,
    )
    ax.axis('off')
    for i, label in enumerate(['Emperical', 'CRDS']):
        ax = fig.add_subplot(gs[i, 0])
        ax.text(
            -0.2,
            0.5,
            label,
            fontsize=25,
            ha='center',
            va='center',
            transform=ax.transAxes,
            rotation=90,
        )
        ax.axis('off')

    # Axis parameters
    # ax.set(xlim=(4,2044),ylim=(-0.1,0.1))

    # Labels
    # for j,f in enumerate(filters):
    #     axes[j,0].set_ylabel(f)
    # for i in range(4):
    #     axes[-1,i].set_xlabel(f"$\{'perp' if i%2 else 'parallel'}$ Dispersion [px]")

    # Save figure
    fig.savefig(f'compareSub-{grism}.pdf')
    pyplot.close(fig)

    print(f'Saved figure for {grism}')


if __name__ == '__main__':
    # Iterate over grisms
    with Pool(3) as p:
        p.map(makeFig, ['CLEAR', 'GR150C', 'GR150R'])
