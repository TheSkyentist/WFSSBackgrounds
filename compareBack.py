#! /usr/bin/env python

# Import packages
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from jwst.flatfield import FlatFieldStep
from matplotlib import pyplot, gridspec, colormaps as cm

# Colormap
cm_lin = cm.get_cmap('gray')
cm_lin.set_bad('red')
cm_diff = cm.get_cmap('coolwarm')
cm_diff.set_bad('black')

# Initialize Pipeline Step
flat_field = FlatFieldStep()

for grism in ['CLEAR', 'GR150C', 'GR150R']:

    # Create figure
    size = 4
    fig = pyplot.figure(figsize=(3 * size, 3 * size))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Iterate over filts
    for i,filt in enumerate(['F115W', 'F150W', 'F200W']):
        gf = f'{grism}-{filt}'

        # Get products
        prods = Table.read(f'{gf}/{gf}.fits')
        pfiles = [f'{gf}/{f}' for f in prods['productFilename']]

        # Get header information
        keys = ['READPATT', 'SUBARRAY']
        htable = Table(
            [[fits.getval(p, k, 'PRIMARY') for p in pfiles] for k in keys], names=keys
        )
        tab = prods[
            np.logical_and(htable['READPATT'] == 'NIS', htable['SUBARRAY'] == 'FULL')
        ]

        # Find files
        file = f'nis-{filt.lower()}-{grism.lower()}_skyflat.fits'
        fcustom = os.path.join('wfssbackgrounds/', file)
        fcrds = flat_field.get_reference_file(
            os.path.join(f'{grism}-{filt}', tab['productFilename'][0]), 'wfssbkg'
        )

        # Open files
        custom = fits.getdata(fcustom, 'SCI')
        if fcrds == 'N/A':
            crds = np.zeros((2048, 2048))
        else:
            crds = fits.getdata(fcrds, 'SCI')

        # Flat field CRDS
        flat_file = flat_field.get_reference_file(
            os.path.join(f'{grism}-{filt}', tab['productFilename'][0]), 'flat'
        )
        fl = fits.getdata(flat_file)
        bad = (fl < 0.6) | (fl > 1.3)
        fl[bad] = 1
        crds /= fl
        crds /= np.nanmedian(crds)
        bad |= ~np.isfinite(crds)
        crds[bad] = 1

        # Plot
        axcustom = fig.add_subplot(gs[0, i])
        axcustom.imshow(custom, cmap=cm_lin, clim=[0.6, 1.3])
        axcustom.axis('off')
        axcustom.set_title(f'{filt} $(N={len(tab)})$', fontsize=25)

        # Plot
        axcrds = fig.add_subplot(gs[1, i])
        im1 = axcrds.imshow(crds, cmap=cm_lin, clim=[0.6, 1.3])
        axcrds.axis('off')

        # Plot
        axdiff = fig.add_subplot(gs[2, i])
        im2 = axdiff.imshow(custom - crds, cmap=cm_diff, clim=[-0.05, 0.05])
        axdiff.axis('off')

        # if i == 0:
        #     axcustom.set_ylabel('Emperical',fontsize=25)
        #     axcrds.set_ylabel('CRDS',fontsize=25)
        #     axdiff.set_ylabel(r'Emperical $-$ CRDS',fontsize=25)

    # Create titles
    fig.suptitle(r'\textbf{'+grism+'}', y = 0.95, fontsize=30)

    # Set labels
    for i, label in enumerate(
        [r'\textbf{Empirical}', r'\textbf{CRDS}', r'\textbf{Empirical $-$ CRDS}']
    ):
        ax = fig.add_subplot(gs[i, 0])
        ax.text(
            -0.075,
            0.5,
            label,
            ha='center',
            va='center',
            transform=ax.transAxes,
            rotation=90,
        )
        ax.axis('off')

    # Create colorbars
    ax = fig.add_subplot(gs[0:2, :])
    fig.colorbar(im1, ax=ax, shrink=0.9, aspect=20, anchor=(1.4, 0.5))
    ax.axis('off')
    ax = fig.add_subplot(gs[2, :])
    fig.colorbar(im2, ax=ax, shrink=0.9, aspect=10, anchor=(1.4, 0.5))
    ax.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)

    # Save figure
    fig.savefig(f'compareBack-{grism}.pdf')
    pyplot.close(fig)