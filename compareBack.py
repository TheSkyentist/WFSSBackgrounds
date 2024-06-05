#! /usr/bin/env python

# Import packages
import os
import glob
import tqdm
import numpy as np
from astropy.io import fits
from itertools import product
from astropy.table import Table
from jwst.flatfield import FlatFieldStep
from matplotlib import pyplot,gridspec,colormaps as cm

# Colormap
cm_lin = cm.get_cmap('gray_r')
cm_lin.set_bad('red')
cm_diff = cm.get_cmap('coolwarm')
cm_diff.set_bad('black')

# Grism filters
gfs = ['-'.join(p) for p in product(['GR150C','GR150R'],['F115W','F150W','F200W'])]

# Initialize Pipeline Step
flat_field = FlatFieldStep()

# Create figure
size = 4
fig = pyplot.figure(figsize=(size*len(gfs),3*size))
gs = gridspec.GridSpec(3,len(gfs),figure=fig)

# Iterate over filts
for i,gf in enumerate(tqdm.tqdm(gfs)):

    grism,filt = gf.split('-')

    # Get products
    prods = Table.read(f'{gf}/{gf}.fits')
    pfiles = [f"{gf}/{f}" for f in prods['productFilename']]

    # Get header information
    keys = ['READPATT','SUBARRAY']
    htable = Table([[fits.getval(p,k,'PRIMARY') for p in pfiles] for k in keys],names=keys)
    tab = prods[np.logical_and(htable['READPATT']=='NIS',htable['SUBARRAY']=='FULL')]

    # Find files
    file = f'nis-{filt.lower()}-{grism.lower()}_skyflat.fits'
    fcustom = os.path.join('wfssbackgrounds/',file)
    fcrds = flat_field.get_reference_file(
        os.path.join(f'{grism}-{filt}',tab['productFilename'][0]),'wfssbkg'
    )

    # Open files
    custom = fits.getdata(fcustom,'SCI')
    # custom[fits.getdata(fcustom,'DQ') != 0] = np.nan
    crds = fits.getdata(fcrds,'SCI')
    # crds[fits.getdata(fcrds,'DQ') != 0] = np.nan

    # Flat field CRDS
    flat_file = flat_field.get_reference_file(
        os.path.join(f'{grism}-{filt}',tab['productFilename'][0]),'flat'
    )
    fl = fits.getdata(flat_file)
    bad = (fl < 0.6) | (fl > 1.3)
    fl[bad] = 1
    crds /= fl
    crds /= np.nanmedian(crds)
    bad |= ~np.isfinite(crds)
    crds[bad] = 1

    # Plot
    axcustom = fig.add_subplot(gs[0,i])
    axcustom.imshow(custom,cmap=cm_lin,clim=[0.6,1.3])
    axcustom.axis('off')
    axcustom.set_title(f'{filt} $(N={len(tab)})$',fontsize=25)

    # Plot
    axcrds = fig.add_subplot(gs[1,i])
    im1 = axcrds.imshow(crds,cmap=cm_lin,clim=[0.6,1.3])
    axcrds.axis('off')

    # Plot
    axdiff = fig.add_subplot(gs[2,i])
    im2 = axdiff.imshow(custom-crds,cmap=cm_diff,clim=[-0.1,0.1])
    axdiff.axis('off')

    # if i == 0:
    #     axcustom.set_ylabel('Emperical',fontsize=25)
    #     axcrds.set_ylabel('CRDS',fontsize=25)
    #     axdiff.set_ylabel(r'Emperical $-$ CRDS',fontsize=25)

# Create titles
ax = fig.add_subplot(gs[:,0:3])
ax.set_title(r'\textbf{GR150C}',y=1.05,fontsize=30)
ax.axis('off')
ax = fig.add_subplot(gs[:,3:])
ax.set_title(r'\textbf{GR150R}',y=1.05,fontsize=30)
ax.axis('off')

# Set labels
for i,l in enumerate(['Empirical','CRDS',r'Empirical $-$ CRDS']):
    ax = fig.add_subplot(gs[i,0])
    ax.text(-0.075,0.5,l,ha='center',va='center',transform=ax.transAxes,rotation=90)
    ax.axis('off')

# Create colorbars
ax = fig.add_subplot(gs[0:2,:])
fig.colorbar(im1,ax=ax,shrink=0.9,aspect=20,anchor=(1.2,0.5))
ax.axis('off')
ax = fig.add_subplot(gs[2,:])
fig.colorbar(im2,ax=ax,shrink=0.9,aspect=10,anchor=(1.2,0.5))
ax.axis('off')
fig.subplots_adjust(wspace=0,hspace=0)

# Save figure
fig.savefig('compareBack.pdf')
pyplot.close(fig)