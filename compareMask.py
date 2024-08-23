#! /usr/bin/env python

# Import packages
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot, colors, colormaps as cm

# Create figure
size = 7
fig, axes = pyplot.subplots(
    2, 3, figsize=(3 * size, 2 * size), sharey=True, sharex=True
)
pyplot.subplots_adjust(wspace=0.0, hspace=0.05)

# Colormap
cm_lin = cm.get_cmap('gray_r')
cm_lin.set_bad('w')
cm_mask = colors.ListedColormap(['#E20134'])

# Load data
gfs, fs = (
    ['CLEAR-F200W', 'GR150R-F200W'],
    [
        'jw01571173001_04201_00001_nis_rate_flatfield.fits',
        'jw01571172001_02201_00001_nis_rate_flatfield.fits',
    ],
)

# Extensions
exts = ['', '_crds', '_custom']
titles = (
    r'\textbf{Flat$\boldsymbol{-}$Fielded Rate File}',
    r'\textbf{CRDS Background Sub.}',
    r'\textbf{Custom Background Sub.}',
)


nonref = (slice(4, -4), slice(4, -4))

# Iterate over grisms:
for j, (gf, f) in enumerate(zip(gfs, fs)):
    # Get mask index
    prods = Table.read(f'{gf}/{gf}.fits')
    index = np.argwhere(prods['obs_id'] == f[:29])[0][0]

    # Iterate over files
    for i, ext in enumerate(exts):
        # Load data
        im = fits.getdata(f'{gf}{ext}/{f}', 'SCI')[nonref]
        dq = fits.getdata(f'{gf}{ext}/{f}', 'DQ')[nonref]
        im[dq != 0] = np.nan

        # Get mask
        mask = np.load(f'{gf}{ext}/masks.npy')[index]
        mask[dq != 0] = 0

        # Subtract median from clear
        if (ext == '_crds') and ('CLEAR' in gf):
            height = np.genfromtxt(
                f'{gf}_custom/{f.replace('.fits','_1_sky_background.info')}'
            )[1]
            im -= height

        # Get limit
        if i == 0 and j == 0:
            clim = [0.7, 0.9]
        elif i == 0:
            clim = [0.4, 0.9]
        else:
            clim = [-0.1, 0.2]

        # Normalize image to clim
        im = (im - clim[0]) / (clim[1] - clim[0])

        # Make image into RGB
        im = np.array(cm_lin(im))
        if i != 0:
            im[mask] = np.array([226, 1, 52, 255]) / 255

        # Plot Image
        axes[j, i].imshow(im)
        axes[j, i].axis('off')

        # Axis title
        if j == 0:
            axes[j, i].set_title(titles[i])

    # Set labels
    axes[j,0].text(
        -0.04,
        0.5,
        gf.replace('-', '$-$'),
        fontsize=30,
        ha='center',
        va='center',
        transform=axes[j,0].transAxes,
        rotation=90,
    )

fig.savefig('compareMask.pdf')
pyplot.close(fig)
