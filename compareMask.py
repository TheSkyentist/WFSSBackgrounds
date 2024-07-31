#! /usr/bin/env python

# Import packages
import numpy as np
from astropy.io import fits
from matplotlib import pyplot, colors, colormaps as cm

# Create figure
size = 8
fig, axes = pyplot.subplots(1, 3, figsize=(size*3, size), sharey=True, sharex=True)
pyplot.subplots_adjust(wspace=0.0)

# Colormap
cm_lin = cm.get_cmap('gray_r')
cm_lin.set_bad('w')
cm_mask= colors.ListedColormap(["#E20134"])

# Load data
f = 'jw01571172001_02201_00001_nis_rate_flatfield.fits'
exts = ['', '_crds', '_custom']
titles = (
    'No Background Subtraction',
    'CRDS Background Subtraction',
    'Custom Background Subtraction',
)

nonref = (slice(4, -4), slice(4, -4))

# Iterate over files
for i, ext in enumerate(exts):
    # Load data
    im = fits.getdata(f'GR150R-F200W{ext}/{f}', 'SCI')[nonref]
    dq = fits.getdata(f'GR150R-F200W{ext}/{f}', 'DQ')[nonref]
    im[dq != 0] = np.nan
    mask = np.load(f'GR150R-F200W{ext}/masks.npy')[26]
    mask[dq != 0] = 0

    # Get limit
    clim = [0.4, 0.9] if i == 0 else [-0.1, 0.2]

    # Normalize image to clim
    im = (im - clim[0])/(clim[1] - clim[0])

    # Make image into RGB
    im = np.array(cm_lin(im))
    im[mask] = np.array([226,1,52,255])/255

    # Plot Image
    axes[i].imshow(im)
    
    axes[i].axis('off')

    # Get medians
    # _, med, _ = sigma_clipped_stats(im, mask=mask, sigma=1, axis=1)
    # axes[1, i].plot(np.arange(4, 2044), med, ds='steps-mid', color='k')
    # axes[1, i].set(ylim=clim, xlim=(0, 2040))

    # _, med, _ = sigma_clipped_stats(im, mask=mask, sigma=1, axis=0)
    # axes[2, i].plot(np.arange(4, 2044), med, ds='steps-mid', color='k')
    # axes[2, i].set(ylim=clim, xlim=(0, 2040))

    # Axis title
    axes[i].set_title(titles[i])

    # Label axes
    # if i == 0:
    #     # axes[0, i].set_ylabel('Y [pix]')
    #     axes[1, i].set_ylabel('Median (Row) [DN/s]')
    #     axes[2, i].set_ylabel('Median (Col) [DN/s]')
    # axes[i, 2].set_xlabel('Distance [pix]')

fig.savefig('compareMask.pdf')
pyplot.close(fig)
