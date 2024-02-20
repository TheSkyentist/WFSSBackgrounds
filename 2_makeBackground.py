#! /usr/bin/env python

# Import packages
import glob
import numpy as np
from tqdm import tqdm
from sep import Background
from astropy.io import fits
from jwst import datamodels
from astropy.table import Table
from multiprocessing import Pool
from photutils.segmentation import detect_sources, detect_threshold
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.convolution import Tophat2DKernel, convolve, interpolate_replace_nans

# JWST FWHMs
filter_fwhm = dict(F090W=1.33,F115W=1.42,F140M=1.39,F150W=1.39,F158M=1.44,F200W=1.61,F277W=1.73,F356W=1.99,F380M=2.11,F430M=2.28,F444W=2.24,F480M=2.54)

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

        # Get PSF kernel
        # fwhm = filter_fwhm[filt]
        # size = np.floor(fwhm)
        # if (size % 2 == 0): size += 1.
        # sigma = gaussian_fwhm_to_sigma*fwhm
        # kernel = Gaussian2DKernel(sigma, x_size=size, y_size=size)

        # Convolve and detect
        thresh = detect_threshold(im-bkg,nsigma=1.0,background=bkg,error=err,mask=mask)
        # imconv = convolve(im-bkg,kernel,normalize_kernel=True,preserve_nan=True)
        seg = detect_sources(im-bkg,thresh,npixels=3,connectivity=8)

        # Mask detected sources
        mask = np.logical_or(mask,seg.data>0)
        im[mask] = np.nan

        # Find weighted mean of the image
        _,median,_ = sigma_clipped_stats(im,mask=mask)

        # Keep track of medianed-out background
        bkgs[i] = im/median
        masks[i] = mask

        # # Plot images
        # from matplotlib import pyplot,colormaps as cm
        # cmap = cm.get_cmap('gray')
        # cmap.set_bad('red')
        # ims = [im,bkg,nosource,nosource2]
        # fig,ax = pyplot.subplots(1,len(ims),figsize=(5*len(ims),5))
        # fig.subplots_adjust(wspace=0,hspace=0)
        # for i,ax in zip(ims,ax):
        #     ax.imshow(i,cmap=cmap,clim=np.nanpercentile(i,[1,99]))
        #     ax.axis('off')
        # fig.savefig('test.pdf')
        # pyplot.close(fig)

    # Mask and median
    _,median,_ = sigma_clipped_stats(bkgs,mask=masks,axis=0)
    bkg = interpolate_replace_nans(median,kernel=Tophat2DKernel(radius=7))

    # Save
    out = np.zeros(shape=(2048,2048),dtype=bkg.dtype)
    out[4:-4,4:-4] = bkg
    g,f = filt.lower().split('-')
    fits.PrimaryHDU(out).writeto(f'wfssbackgrounds/nis-{f}-{g}_skyflat.fits',overwrite=True)

# Process
pool = Pool(processes=8)
pool.map_async(extractBackground,sorted(glob.glob('GR*')),chunksize=1)
pool.close()
pool.join()



