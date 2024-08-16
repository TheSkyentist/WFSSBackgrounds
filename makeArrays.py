#! /usr/bin/env python

# Python Packages
import shutil
import numpy as np
from tqdm import tqdm
from os.path import isdir
from itertools import product
from multiprocessing import Pool, cpu_count

# Image Processing Packages
from sep import Background
from photutils.segmentation import detect_sources, detect_threshold

# Astropy Packages
from astropy.table import Table

# JWST Pipeline
from jwst.flatfield import FlatFieldStep

flat_field = FlatFieldStep()

# Grism filters
filts = [
    '-'.join(p)
    for p in product(['CLEAR', 'GR150C', 'GR150R'], ['F115W', 'F150W', 'F200W'])
]

# Non reference pixels (in FULL mode)
nonref = (slice(4, -4), slice(4, -4))


def makeArray(r):
    # Flat-field
    dm = flat_field.call(r)
    dm.save(r.replace('.fits', '_flatfield.fits'))
    im = dm.data[nonref].astype(np.float32)
    err = dm.err[nonref].astype(np.float32)
    dq = dm.dq[nonref]

    # Mask data
    mask = dq > 0
    im[mask] = np.nan

    # Rudimentary background subtraction
    bkg = Background(im, mask=mask).back()

    # Detect
    thresh = detect_threshold(im, nsigma=2, background=bkg, error=err, mask=mask)
    seg = detect_sources(im, thresh, npixels=10, connectivity=8)

    # Mask detected sources
    mask = np.logical_or(mask, seg.data > 0)

    return im, mask


# Define background function
def makeSourceMask(filt):
    # Get products
    prods = Table.read(f'{filt}/{filt}.fits')
    rate = [f'{filt}/{f}' for f in prods['productFilename']]

    # Make arrays
    masks = np.zeros(shape=(len(rate), 2040, 2040), dtype=bool)
    flatted = np.zeros(shape=(len(rate), 2040, 2040), dtype=float)

    # Multipricess
    with Pool(processes=cpu_count()) as pool:
        for i, (im, mask) in tqdm(
            enumerate(pool.imap(makeArray, rate)), total=len(rate)
        ):
            masks[i] = mask
            flatted[i] = im

    # Save mask to pickle
    np.save(f'{filt}/masks.npy', masks)
    np.save(f'{filt}/flatted.npy', flatted)


# Main function
if __name__ == '__main__':
    # Delete old directories
    with Pool(processes=cpu_count()) as pool:
        pool.map(
            shutil.rmtree,
            [
                f'{filt}{ext}'
                for filt in filts
                for ext in ['_custom', '_crds']
                if isdir(f'{filt}{ext}')
            ],
        )

    for filt in filts:
        makeSourceMask(filt)
        print(f'Created mask for {filt}')
