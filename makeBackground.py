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
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, detect_threshold
from astropy.convolution import Tophat2DKernel, interpolate_replace_nans

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('-d','--dontFlat',action="store_true",help="Don't flat-field the data")
args = parser.parse_args()
dontFlat = args.dontFlat

# Non reference pixels (in FULL mode)
reg = (slice(4,-4),slice(4,-4))

# Initialize Pipeline Step
if not dontFlat:
    from jwst.flatfield import FlatFieldStep
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

        # Open file and optionally flat-field
        if dontFlat:
            hdul = fits.open(o)
            im = hdul['SCI'].data[reg].astype(np.float32)
            err = hdul['ERR'].data[reg].astype(np.float32)
            dq = hdul['DQ'].data[reg]
        else:
            dm = flat_field.process(o)
            im = dm.data[reg].astype(np.float32)
            err = dm.err[reg].astype(np.float32)
            dq = dm.dq[reg]

        # Mask data
        mask = dq > 0
        im[mask] = np.nan

        # Rudimentary background subtraction
        bkg = Background(im,mask=mask).back()

        # Detect
        thresh = detect_threshold(im-bkg,nsigma=1.0,background=bkg,error=err,mask=mask)
        seg = detect_sources(im-bkg,thresh,npixels=3,connectivity=8)

        # Mask detected sources
        mask = np.logical_or(mask,seg.data>0)
        im[mask] = np.nan

        # Find median
        _,median,_ = sigma_clipped_stats(im,mask=mask)

        # Keep track of medianed-out background and mask
        bkgs[i] = im/median
        masks[i] = mask

    # Mask and median
    _,median,_ = sigma_clipped_stats(bkgs,mask=masks,axis=0)
    bkg = interpolate_replace_nans(median,kernel=Tophat2DKernel(radius=7))

    # Create WFSS Background
    out = np.zeros(shape=(2048,2048),dtype=bkg.dtype)
    out[reg] = bkg # Reference pixels to zero

    # Set header information
    g,f = filt.split('-')
    hdu = fits.PrimaryHDU(out)
    hdu.header = fits.getheader(f'{filt}/{filt}.fits')
    hdu.header['PUPIL'] = f
    hdu.header['FILTER'] = g

    # Record if we have flat-fielded
    if not dontFlat:
        hdu.header['FIXFLAT'] = True

    # Save to file
    hdu.writeto(f'wfssbackgrounds/nis-{f.lower()}-{g.lower()}_skyflat.fits',overwrite=True)

# Main function
if __name__ == '__main__':

    # Multiprocess
    pool = Pool(processes=12)
    pool.map_async(extractBackground,sorted(glob.glob('GR*')),chunksize=1)
    pool.close()
    pool.join()



