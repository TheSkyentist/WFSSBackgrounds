#! /usr/bin/env python

# Import packages
import glob
import numpy as np
from tqdm import tqdm
from sep import Background
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Pool
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, detect_threshold
from astropy.convolution import Tophat2DKernel, interpolate_replace_nans

# Define background function
def extractBackground(filt):

    # Get products
    out = [f"{filt}/{f.replace('rate','out')}" for f in Table.read(f'{filt}/{filt}.fits')['productFilename'].tolist()]
    out = [o for o in out if fits.getval(o,'SUBARRAY','PRIMARY') == 'FULL']

    # Make arrays
    bkgs = np.zeros(shape=(len(out),2040,2040),dtype=np.float32)
    masks = np.zeros(shape=(len(out),2040,2040),dtype=bool)
    for i,o in tqdm(enumerate(out),total=len(out)):

        # Get data (extract non-reference pixels)
        hdul = fits.open(o)
        im = hdul['SCI'].data[4:-4,4:-4].astype(np.float32)
        err = hdul['ERR'].data[4:-4,4:-4].astype(np.float32)
        dq = hdul['DQ'].data[4:-4,4:-4]
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

        # Find weighted mean of the image
        _,median,_ = sigma_clipped_stats(im,mask=mask)

        # Keep track of medianed-out background and mask
        bkgs[i] = im/median
        masks[i] = mask

    # Mask and median
    _,median,_ = sigma_clipped_stats(bkgs,mask=masks,axis=0)
    bkg = interpolate_replace_nans(median,kernel=Tophat2DKernel(radius=7))

    # Create WFSS Background
    out = np.zeros(shape=(2048,2048),dtype=bkg.dtype)
    out[4:-4,4:-4] = bkg # Reference pixels to zero

    # Set header information
    g,f = filt.split('-')
    hdu = fits.PrimaryHDU(out)
    hdu.header = fits.getheader(f'{filt}/{filt}.fits')
    hdu.header['PUPIL'] = f
    hdu.header['FILTER'] = g

    # Save to file
    hdu.writeto(f'wfssbackgrounds/nis-{f.lower()}-{g.lower()}_skyflat.fits',overwrite=True)

# Main function
if __name__ == '__main__':

    # Multiprocess
    pool = Pool(processes=12)
    pool.map_async(extractBackground,sorted(glob.glob('GR*')),chunksize=1)
    pool.close()
    pool.join()



