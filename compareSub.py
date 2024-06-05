#! /usr/bin/env python

# Import packages
import os
import glob
import numpy as np
from tqdm import tqdm
from itertools import product
from matplotlib import pyplot,colormaps as cm

# Import astropy functions
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

# Silence warnings
# import warnings
# warnings.filterwarnings('ignore')

# Grism filters
gfs = ['-'.join(p) for p in product(['GR150C','GR150R'],['F115W','F150W','F200W'])]

# Create figure
size = 4
fig = pyplot.figure(figsize=(size*len(gfs),2*size))
gs = fig.add_gridspec(2,len(gfs),hspace=0.1,wspace=0)

# Median Region
medreg = (slice(128,-128),slice(128,-128))

# Outside reference regions
nonref = (slice(4,-4),slice(4,-4))

# Iterate over filts
for i,gf in enumerate(gfs):

    # Get grism and filter
    grism,filt = gf.split('-')

    # Iterate over extensions
    for j,ext in enumerate(['_custom','_crds']):

        # Get subbed products
        subbed = np.load(f'{gf}{ext}/subbed.npy')

        # Get masked products
        masks = np.load(f'{gf}{ext}/masks.npy')

        # Get background strenghts
        heights = np.reshape([np.genfromtxt(f)[1] for f in sorted(glob.glob(f'{gf}{ext}/*.info'))],(len(subbed),1,1))

        print(f'Loaded data for {gf}{ext}')

        # Get subgridspec
        subgs = gs[j,i].subgridspec(2,1,hspace=0)

        # Iterate over dispersion direction
        for k in range(2): 

            # Get axis 
            axis = (i//3 + k) % 2 + 1

            # Get summary statistics
            # medians = np.nanmedian((subbed/heights),axis=axis)
            _,medians,_ = sigma_clipped_stats((subbed/heights),sigma=1,mask=masks,axis=axis)

            # Get summary of summary statistics
            # median = np.nanmedian(medians,axis=0)
            # stddev = np.nanstd(medians,axis=0)
            _,median,stddev = sigma_clipped_stats(medians,sigma=1,axis=0)

            # Plot
            ax = fig.add_subplot(subgs[k])
            ax.fill_between(np.arange(4,2044),median-3*stddev,median+3*stddev,color='k',alpha=0.1,joinstyle='miter',interpolate=False,ec='none',step='mid')
            ax.plot(np.arange(4,2044),median,color='k',ds='steps-mid',alpha=1,solid_joinstyle='miter',linewidth=1)
            ax.set(xlim=(4,2044),ylim=(-0.05,0.05),xticklabels=[])

            # Set title
            if j == 0 and k == 0:
                ax.set_title(f'{filt} $(N={len(subbed)})$',fontsize=25)

            # Set y-label
            if i == 0:
                label = r'$\perp$' if k else r'$\parallel$'
                ax.set_ylabel(label)
            else:
                ax.set_yticklabels([])
    
# Create titles
ax = fig.add_subplot(gs[:,0:3])
ax.set_title(r'\textbf{GR150C}',y=1.05,fontsize=30)
ax.axis('off')
ax = fig.add_subplot(gs[:,3:])
ax.set_title(r'\textbf{GR150R}',y=1.05,fontsize=30)
ax.axis('off')

# Set labels
for i,l in enumerate([r'\textbf{Empirical}',r'\textbf{CRDS}',r'\textbf{Empirical $-$ CRDS}'][:2]):
    ax = fig.add_subplot(gs[i,0])
    ax.text(-0.5,0.5,l,ha='center',va='center',transform=ax.transAxes,rotation=90)
    ax.axis('off')

# Axis parameters
# ax.set(xlim=(4,2044),ylim=(-0.1,0.1))

# Labels
# for j,f in enumerate(filters):
#     axes[j,0].set_ylabel(f)
# for i in range(4):
#     axes[-1,i].set_xlabel(f"$\{'perp' if i%2 else 'parallel'}$ Dispersion [px]")

# Save figure
fig.savefig('compareSub.pdf')
pyplot.close(fig)