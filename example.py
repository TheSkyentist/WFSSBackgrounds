#! /usr/bin/env python

# Import packages
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib
from matplotlib import pyplot, gridspec, colormaps as cm

matplotlib.use('PDF')

# Colormap
cmap = cm.get_cmap('coolwarm_r')
cmap.set_bad('k')

# Grisms & Filters
grisms = ['CLEAR', 'GR150C', 'GR150R']
filts = ['F115W', 'F150W', 'F200W']

# Dictionary of example files
suffix = '_00001_nis_rate_flatfield.fits'
obsids = {
    'CLEAR-F115W': 'jw03383360001_35201',
    'CLEAR-F150W': 'jw03383356001_27201',
    'CLEAR-F200W': 'jw03383352001_03201',
    'GR150C-F115W': 'jw03383368001_49201',
    'GR150C-F150W': 'jw03383366001_45201',
    'GR150C-F200W': 'jw03383364001_41201',
    'GR150R-F115W': 'jw03383369001_51201',
    'GR150R-F150W': 'jw03383367001_47201',
    'GR150R-F200W': 'jw03383365001_43201',
}

# Create figure
size = 4
fig = pyplot.figure(figsize=(3 * size, 3 * size))
fig.subplots_adjust(wspace=1 / 40, hspace=1 / 40)
gs = gridspec.GridSpec(3, 3, figure=fig)

# Non-reference region
nonref = (slice(4, -4), slice(4, -4))

# Iterate over filts
for gf, obsid in obsids.items():
    # Get data
    file = f'{obsid}{suffix}'
    rate = f'{gf}_crds/{file}'
    sci = fits.getdata(rate, 'SCI')[nonref]

    # Load mask
    masks = np.load(f'{gf}_custom/masks.npy')

    # Load table
    prods = Table.read(f'{gf}/{gf}.fits')

    # Get exact mask
    mask = masks[prods['productFilename'] == file.replace('_flatfield', '')][0]
    sci[mask] = np.nan

    # If clear, subbtract the median
    if 'CLEAR' in gf:
        # Subtract median
        median = np.median(sci[np.invert(mask)])
        sci -= median

        # Normalize to median
        sci /= median

    else:
        # Load sky_background level
        median = np.genfromtxt(
            f'{gf}_crds/{file.replace('.fits', '_1_sky_background.info')}'
        )[1]

        # Normalize to median
        sci /= median

    # Get index into figure
    g, f = gf.split('-')
    i, j = grisms.index(g), filts.index(f)

    # Create axes
    ax = fig.add_subplot(gs[i, j])

    # Plot image
    cim = ax.imshow(sci * 100, cmap=cmap, clim=(-5, 5))
    ax.axis('off')

    # Add title
    if i == 0:
        ax.set_title(f)


# Set labels
for i, label in enumerate(grisms):
    ax = fig.add_subplot(gs[i, 0])
    ax.text(
        -0.075,
        0.5,
        label,
        fontsize=30,
        ha='center',
        va='center',
        transform=ax.transAxes,
        rotation=90,
    )
    ax.axis('off')

# Create colorbars
ax = fig.add_subplot(gs[:, :])
fig.colorbar(
    cim,
    ax=ax,
    ticks=range(-5, 6),
    shrink=0.9975,
    aspect=25,
    anchor=(1.55, 0.5),
    label='Percent of Background',
)
# Set colorbar ticks

ax.axis('off')

# Save figure
# fig.savefig('example.pdf')
fig.savefig('example.png', dpi=300)
pyplot.close(fig)
