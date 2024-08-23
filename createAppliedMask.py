#! /usr/bin/env python

# Import packages
import numpy as np
from tqdm import tqdm
from itertools import product

# Astropy
from astropy.io import fits
from astropy.table import Table

# Multiprocessing
from multiprocessing import Pool, cpu_count

# Image Processing Packages
from sep import Background
from photutils.segmentation import detect_sources, detect_threshold

# List of filter grism pairs
filts = [
    ''.join(p)
    for p in product(
        ['CLEAR-', 'GR150C-', 'GR150R-'],
        ['F115W', 'F150W', 'F200W'],
        ['_crds', '_custom'],
    )
]

nonref = (slice(4, -4), slice(4, -4))

def makeMask(file):

    # Load data
    hdul = fits.open(file)
    im = hdul['SCI'].data[nonref].astype(np.float32)
    err = hdul['ERR'].data[nonref].astype(np.float32)
    dq = hdul['DQ'].data[nonref].astype(np.float32)
    mask = dq > 0

    # Create new mask
    bkg = Background(im, mask=mask).back()

    # Detect
    im = hdul['SCI'].data[nonref]
    thresh = detect_threshold(im, nsigma=1, background=bkg, error=err, mask=mask)
    seg = detect_sources(im, thresh, npixels=10, connectivity=8)

    # Mask detected sources
    mask = np.logical_or(mask, seg.data > 0)
    im[mask] = np.nan

    return mask, im


# Background Step
def createAppliedMask(filt):
    # Get products
    prods = Table.read(f"{filt.split('_')[0]}/{filt.split('_')[0]}.fits")
    files = [
        f"{filt}/{f.replace('.fits','_flatfield.fits')}"
        for f in prods['productFilename']
    ]

    # Make arrays
    masks = np.zeros(shape=(len(files), 2040, 2040), dtype=bool)
    subbed = np.zeros(shape=(len(files), 2040, 2040), dtype=float)

    # Multipricess over files
    with Pool(processes=cpu_count()) as pool:
        for i, (mask, sub) in enumerate(tqdm(pool.imap(makeMask, files), total=len(files), desc=filt)):
            masks[i] = mask
            subbed[i] = sub

    # Save
    np.save(f'{filt}/masks.npy', masks)
    np.save(f'{filt}/subbed.npy', subbed)
    print(f'Created source mask for {filt}')

if __name__ == '__main__':
    for filt in filts:
        createAppliedMask(filt)
