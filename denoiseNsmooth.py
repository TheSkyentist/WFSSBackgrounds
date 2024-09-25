#! /usr/bin/env python

# Python Packages
import os
import numpy as np
from itertools import product
from multiprocessing import Pool
from datetime import datetime, timezone

# Image Processing Packages
from maskfill import maskfill
from skimage.restoration import estimate_sigma, denoise_nl_means

# Astropy Packages
from astropy.io import fits
from astropy.table import Table

# JWST Pipeline Packages
from jwst.flatfield import FlatFieldStep

# Ignore RuntimeWarnings
# import warnings
# warnings.simplefilter("ignore", category=RuntimeWarning)

# Grism filters
filts = [
    '-'.join(p)
    for p in product(['CLEAR', 'GR150C', 'GR150R'], ['F115W', 'F150W', 'F200W'])
]

# Outside reference regions
nonref = (slice(4, -4), slice(4, -4))


# Define background function
def denoiseNsmooth(filt):
    print(f'Denoising and smoothing {filt}')

    # Get a valid product
    p = Table.read(f'{filt}/{filt}.fits')['productFilename'][0]

    # Find CRDS WFSS background
    ref = 'flat' if 'CLEAR' in filt else 'wfssbkg'
    flat_field = FlatFieldStep()
    wfssbkg = fits.open(flat_field.get_reference_file(f'{filt}/{p}', ref))

    # Remove DATE keyword for direct
    if 'CLEAR' in filt:
        del wfssbkg['PRIMARY'].header['DATE']

    # Get DQ array
    dq = wfssbkg['DQ'].data[nonref]

    # Load arrays
    bkg = np.load(f'{filt}/background.npy')
    sigma_est = estimate_sigma(bkg)

    # Denoise and smooth
    den = denoise_nl_means(
        bkg,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
        h=0.8 * sigma_est,
        sigma=sigma_est,
        preserve_range=True,
    )
    den, _ = maskfill(den, dq > 0, smooth=False)

    # Create WFSS Background
    out = np.zeros(shape=(2048, 2048), dtype=den.dtype)
    out[nonref] = den  # Reference pixels to zero

    # Reverse the flat-field
    ratehdul = fits.open(f'{filt}/{p}')
    ratehdul['SCI'].data = out
    unflat = flat_field.call(ratehdul, inverse=True).data

    # Fix Header Values
    header = wfssbkg['PRIMARY'].header
    qdate = fits.getval(f'{filt}/{filt}.fits', 'QDATE', 'PRIMARY')
    header['QDATE'] = qdate
    # header['USEAFTER'] = qdate
    for deletekey in ['HISTORY', 'PEDIGREE', 'USEAFTER']:
        del header[deletekey]
    header['DATE'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    header['AUTHOR'] = 'Raphael Hviding'
    header['DATAMODL'] = 'ImgBkgModel' if 'CLEAR' in filt else 'WfssBkgModel'
    header['REFTYPE'] = 'IMGBKG' if 'CLEAR' in filt else 'WFSSBKG'
    header['DESCRIP'] = 'Emperical Background'

    # Place the unflat fielded background into HDUL
    wfssbkg['SCI'].data = unflat

    # Save to file
    g, f = filt.split('-')
    filename = f'nis-{f.lower()}-{g.lower()}_skyflat_unflat.fits'
    header['FILENAME'] = filename
    wfssbkg.writeto(os.path.join('wfssbackgrounds',filename), overwrite=True)

    # Reverse the flat-field on the error
    ratehdul['SCI'].data = wfssbkg['ERR'].data
    unflat_err = flat_field.call(ratehdul, inverse=True).data

    # Save the flat-fielded background
    wfssbkg['SCI'].data = out
    wfssbkg['ERR'].data = unflat_err
    header['FIXFLAT'] = True

    # Save to file
    filename = f'nis-{f.lower()}-{g.lower()}_skyflat.fits'
    header['FILENAME'] = filename
    wfssbkg.writeto(os.path.join('wfssbackgrounds',filename), overwrite=True)
    print(f'Denoised and smoothed {filt}')


# Main function
if __name__ == '__main__':
    # Multiprocess
    with Pool(processes=len(filts)) as pool:
        pool.map(denoiseNsmooth, filts, chunksize=1)
