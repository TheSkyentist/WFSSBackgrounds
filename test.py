#! /usr/bin/env python

# Import packages
import numpy as np
from astropy.io import fits
from matplotlib import pyplot,colormaps as cm
from astropy.stats import sigma_clipped_stats

# Create figure
fig,axes = pyplot.subplots(3,3,figsize=(30,30))
pyplot.subplots_adjust(wspace=0,hspace=0)

# Colormap
cm_lin = cm.get_cmap('gray_r')
cm_lin.set_bad('red')

# Load data
f = 'jw04681407001_04201_00002_nis_rate_flatfield.fits'
exts = ['','_crds','_custom']

nonref = (slice(4,-4),slice(4,-4))

# Iterate over files
for i,ext in enumerate(exts):

    # Load data
    im = fits.getdata(f'GR150R-F200W{ext}/{f}','SCI')[nonref]
    mask = np.load(f'GR150R-F200W{ext}/masks.npy')[-1]
    im[mask] = np.nan

    # Get limit
    clim = [0.4,0.9] if i == 0 else [-0.1,0.2]

    # Plot
    axes[0,i].imshow(im,cmap=cm_lin,clim=clim)
    axes[0,i].axis('off')

    # Get medians
    _,med,_ = sigma_clipped_stats(im,mask=mask,sigma=1,axis=1)
    axes[1,i].plot(np.arange(4,2044),med,ds='steps-mid')
    axes[1,i].set(ylim=clim,xlim=(0,2040))

    _,med,_ = sigma_clipped_stats(im,mask=mask,sigma=1,axis=0)
    axes[2,i].plot(np.arange(4,2044),med,ds='steps-mid')
    axes[2,i].set(ylim=clim,xlim=(0,2040))

fig.savefig('test.pdf')
pyplot.close(fig)