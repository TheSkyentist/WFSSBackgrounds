#! /usr/bin/env python

# Import packages
import numpy as np
from matplotlib import pyplot, lines

# Import astropy functions
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic

# Silence warnings
import warnings

warnings.filterwarnings('ignore')

# Colors
colors = [
    ['#FF6E3A', '#E20134', '#A40122'],
    ['#00C2F9', '#008DF9', '#8400CD'],
    ['#FFB2FD', '#FF5AAF', '#9F0162'],

]

# Create figure
size = 6
fig, axes = pyplot.subplots(1, 3, figsize=(3 * size, size), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)

# Iterate over grisms
for i, g in enumerate(['CLEAR', 'GR150C', 'GR150R']):
    ax = axes[i]
    # Iterate over filters
    handles = []
    for j, f in enumerate(['F115W', 'F150W', 'F200W']):
        # Get grism filter
        gf = f'{g}-{f}'
        c = colors[i][j]

        # Get products
        prods = Table.read(f'{gf}/{gf}.fits')
        pfiles = [f'{gf}/{f}' for f in prods['productFilename']]

        # Get absolute ecliptic latitude
        keys = ['TARG_RA', 'TARG_DEC']
        htable = Table(
            [[fits.getval(p, k, 'PRIMARY') for p in pfiles] for k in keys], names=keys
        )
        elev = np.abs(
            SkyCoord(htable['TARG_RA'], htable['TARG_DEC'], unit='deg')
            .transform_to(GeocentricMeanEcliptic)
            .lat.deg
        )

        # Get background heights
        heights = np.array(
            [
                np.genfromtxt(
                    f.replace('/', '_custom/').replace(
                        '.fits', '_flatfield_1_sky_background.info'
                    )
                )[1]
                for f in pfiles
            ]
        )

        # Scatter plot
        ax.scatter(elev, heights, c=c, alpha=0.25, edgecolor='none')

        # Sort into bins
        binsize = 15
        bins = np.arange(0, 90 + 1, binsize)
        abins = np.digitize(elev, bins)

        # Remove outliers
        abins[
            np.logical_or.reduce(
                [
                    np.logical_and(elev > 15, heights > 1.4),
                    np.logical_and(elev > 80, heights > 0.8),
                    np.logical_and(elev > 80, heights < 0.1),
                ]
            )
        ] = 0

        # Compute average
        y = np.array([np.mean(heights[abins == i]) for i in range(1, len(bins))])
        yerr = np.array([np.std(heights[abins == i]) for i in range(1, len(bins))])

        # Plot average
        x = bins[:-1] + binsize / 2
        # ax.errorbar(x, y, yerr=yerr, xerr=binsize / 2, ls='none', c=c, capsize=5, capthick=2)
        # ax.scatter(x, y, fc='none', ec=c, s=60)

        # Empty marker for legend
        handles.append(lines.Line2D([], [], color=c, ls='none', marker='o', label=f))

    # Plot regions around LMC and M33
    if g == 'GR150C':
        ax.fill_between([18, 20.5], 1.4, 2.4, fc='none', ls='--', ec='gray')
        ax.text(19.25, 1.32, 'LMC', ha='center', va='center', fontsize=20, c='gray')
    ax.fill_between([83.5, 87.5], 0, 1.425, fc='none', ls='--', ec='gray')
    ax.text(83, 1.48, 'M33', ha='center', va='center', fontsize=20, c='gray')

    # Set limits
    ax.set(
        xlim=[0, 90],
        xticks=np.arange(0, 90, 10),
        xlabel='Absolute Ecliptic Latitude [deg]',
    )
    if i == 0:
        ax.set(ylim=[0, 2.5], ylabel=r'Average Background [DN\,s$^{-1}$]')
    ax.set_title(f'{g}')
    ax.legend(
        handles=handles,
        fontsize=25,
        columnspacing=1,
        labelspacing=0.1,
        markerscale=2,
        handletextpad=0.1,
    )

# Save figure
fig.savefig('zodi.pdf')
pyplot.close(fig)
