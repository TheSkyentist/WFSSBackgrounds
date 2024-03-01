#! /usr/bin/env python

# Import packages
import glob
import argparse
import numpy as np
from tqdm import tqdm
from sep import Background
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Pool
from jwst.flatfield import FlatFieldStep
from astropy.stats import sigma_clipped_stats
from skimage.restoration import denoise_tv_chambolle
from photutils.segmentation import detect_sources, detect_threshold
from astropy.convolution import Tophat2DKernel, interpolate_replace_nans
    
# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('-d','--dontFlat',action="store_true",help="Don't flat-field the data")
args = parser.parse_args()
dontFlat = args.dontFlat

# Grism filters
filts = ['GR150C-F115W','GR150C-F150W','GR150C-F200W','GR150R-F115W','GR150R-F150W','GR150R-F200W']

# Non reference pixels (in FULL mode)
nonref = (slice(4,-4),slice(4,-4))

# Median measurement region
medreg = (slice(128,-128),slice(128,-128))

# Initialize Pipeline Step
flat_field = FlatFieldStep()

# Define background function
def extractBackground(filt):

    # Get products
    out = [f"{filt}/{f}" for f in Table.read(f'{filt}/{filt}.fits')['productFilename'].tolist()]
    out = [o for o in out if fits.getval(o,'SUBARRAY','PRIMARY') == 'FULL']

    # Make arrays
    bkgs = np.zeros(shape=(len(out),2040,2040),dtype=np.float32)
    masks = np.zeros(shape=(len(out),2040,2040),dtype=bool)
    for i,o in tqdm(enumerate(out),total=len(out)):

        # Open file  flat-field
        with fits.open(o) as hdul:
            im = hdul['SCI'].data[nonref].astype(np.float32)
            err = hdul['ERR'].data[nonref].astype(np.float32)
            dq = hdul['DQ'].data[nonref]

        # Mask data
        mask = dq > 0
        im[mask] = np.nan

        # Rudimentary background subtraction
        bkg = Background(im,mask=mask).back()

        # Detect
        thresh = detect_threshold(im-bkg,nsigma=1,background=bkg,error=err,mask=mask)
        seg = detect_sources(im-bkg,thresh,npixels=3,connectivity=8)

        # Mask detected sources
        mask = np.logical_or(mask,seg.data>0)
        im[mask] = np.nan

        # Find median to normalize
        _,norm,_ = sigma_clipped_stats(im[medreg],sigma=1,mask=mask[medreg])

        # Keep track of medianed-out background and mask
        bkgs[i] = im/norm
        masks[i] = mask

    # Mask and median
    _,median,_ = sigma_clipped_stats(bkgs,mask=masks,sigma_lower=1,sigma_upper=3,axis=0)
    bkg = interpolate_replace_nans(median,kernel=Tophat2DKernel(radius=7))

    # Denoise (only if we flat-fielded)
    den = denoise_tv_chambolle(bkg)

    # Create WFSS Background
    out = np.zeros(shape=(2048,2048),dtype=den.dtype)
    out[nonref] = den # Reference pixels to zero

    # If don't flat, undo Flat-Field
    if dontFlat:
        flat_file = flat_field.get_reference_file(o,'flat')
        out *= fits.getdata(flat_file)

    # Open CRDS WFSS background file
    hdul = fits.open(flat_field.get_reference_file(o, 'wfssbkg'))
    
    # Update data
    hdul[1].data = out
    hdul[0].header['QDATE'] = fits.getval(f'{filt}/{filt}.fits','QDATE','PRIMARY')

    # Record if we have flat-fielded
    if not dontFlat: hdul[0].header['FIXFLAT'] = True

    # Save to file
    g,f = filt.split('-')
    hdul.writeto(f'wfssbackgrounds/nis-{f.lower()}-{g.lower()}_skyflat.fits',overwrite=True)

# Main function
if __name__ == '__main__':

    # Multiprocess
    pool = Pool(processes=len(filts))
    pool.map_async(extractBackground,filts,chunksize=1)
    pool.close()
    pool.join()