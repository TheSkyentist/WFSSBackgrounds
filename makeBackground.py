#! /usr/bin/env python

# Python Packages
import os
import warnings
import argparse
import numpy as np
from maskfill import maskfill
from itertools import product
from multiprocessing import Pool

# Astropy Packages
from astropy.stats import sigma_clipped_stats
# from astropy.convolution import Tophat2DKernel,interpolate_replace_nans

# Ignore RuntimeWarnings
warnings.simplefilter("ignore", category=RuntimeWarning)

# Grism filters
filts = ['-'.join(p) for p in product(['GR150C','GR150R'],['F115W','F150W','F200W'])]

# Median measurement region
medreg = (slice(128,-128),slice(128,-128))

# Define background function
def makeAvgBackground(filt):

    print(f'Creating background for {filt}')

    # Load Arrays
    if os.path.exists(f'{filt}_custom/masks.npy'):
        print('Using custom masks')
        masks = np.load(f'{filt}_custom/masks.npy')
    else:
        masks = np.load(f'{filt}/masks.npy')
    flatted = np.load(f'{filt}/flatted.npy')

    # Normalize
    _,norms,_ = sigma_clipped_stats(flatted[:,*medreg],sigma=1,mask=masks[:,*medreg],axis=(1,2))
    bkgs = flatted/norms[:,None,None]

    # Median
    _,median,_ = sigma_clipped_stats(bkgs,mask=masks,sigma_lower=1,sigma_upper=1,axis=0)
    
    # Fill in NaNs
    bkg,_ = maskfill(median,np.isnan(median),smooth=False)
    # bkg = interpolate_replace_nans(median,kernel=Tophat2DKernel(radius=7))

    # Save background
    np.save(f'{filt}/background.npy',bkg)
    print(f'Created background for {filt}')

# Main function
if __name__ == '__main__':

    # Multiprocess
    pool = Pool(processes=len(filts))
    pool.map_async(makeAvgBackground,filts,chunksize=1)
    pool.close()
    pool.join()