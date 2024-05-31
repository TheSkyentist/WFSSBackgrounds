#! /usr/bin/env python

# Python Packages
import numpy as np
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool

# Image Processing Packages
from sep import Background
from photutils.segmentation import detect_sources, detect_threshold

# Astropy Packages
from astropy.io import fits
from astropy.table import Table

# JWST Pipeline
from jwst.flatfield import FlatFieldStep
flat_field = FlatFieldStep()

# Grism filters
filts = ['-'.join(p) for p in product(['GR150C','GR150R'],['F115W','F150W','F200W'])]

# Non reference pixels (in FULL mode)
nonref = (slice(4,-4),slice(4,-4))

# Define background function
def makeSourceMask(filt):

    # Get products
    prods = Table.read(f'{filt}/{filt}.fits')
    pfiles = [f"{filt}/{f}" for f in prods['productFilename']]

    # Get header information
    keys = ['READPATT','SUBARRAY']
    htable = Table([[fits.getval(p,k,'PRIMARY') for p in pfiles] for k in keys],names=keys)

    # Restrict to NIS and FULL
    rate = np.array(pfiles)[np.logical_and(htable['READPATT']=='NIS',htable['SUBARRAY']=='FULL')]

    # Make arrays
    masks = np.zeros(shape=(len(rate),2040,2040),dtype=bool)
    flatted = np.zeros(shape=(len(rate),2040,2040),dtype=float)
    for i,r in tqdm(enumerate(rate),total=len(rate)):

        # Flat-field
        dm = flat_field.call(r)
        dm.save(r.replace('.fits','_flatfield.fits'))
        im = dm.data[nonref].astype(np.float32)
        err = dm.err[nonref].astype(np.float32)
        dq = dm.dq[nonref]

        # Mask data
        mask = dq > 0
        im[mask] = np.nan
        flatted[i] = im

        # Rudimentary background subtraction
        bkg = Background(im,mask=mask).back()

        # Detect
        thresh = detect_threshold(im-bkg,nsigma=1,background=bkg,error=err,mask=mask)
        seg = detect_sources(im-bkg,thresh,npixels=3,connectivity=8)

        # Mask detected sources
        mask = np.logical_or(mask,seg.data>0)
        masks[i] = mask

    # Save mask to pickle
    np.save(f'{filt}/masks.npy',masks)
    np.save(f'{filt}/flatted.npy',flatted)

# Main function
if __name__ == '__main__':

    # Multiprocess
    pool = Pool(processes=len(filts))
    pool.map_async(makeSourceMask,filts,chunksize=1)
    pool.close()
    pool.join()