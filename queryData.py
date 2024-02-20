#! /usr/bin/env python

# Import packages
import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone

# Astropy packages
import astropy.units as u
from astropy.io import fits
from astroquery.gaia import Gaia
from astropy.table import vstack
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord

# Query all data
all_obs = Observations.query_criteria(instrument_name="NIRISS/WFSS",obs_collection="JWST",dataRights="PUBLIC",intentType='science')
all_obs = all_obs[np.unique(all_obs['obs_id'],return_index=True)[1]]

# Limiting star magnitude
maglim = 9.5

# Query from GAIA lite
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source_lite"

# Iterate over filters and grisms
for filt in sorted(np.unique(all_obs['filters'])):

    print(filt)

    # Restrict to filter
    obs = all_obs[all_obs['filters'] == filt]
    coords = SkyCoord(ra=obs['s_ra'], dec=obs['s_dec'], unit=u.degree, frame='icrs')

    # Query GAIA
    gaia = [Gaia.cone_search(c, radius=2*u.arcmin).get_results() for c in tqdm(coords)]

    # Find bright stars
    good = np.array([g['phot_g_mean_mag'].min() > maglim for g in gaia])
    obs = obs[good]

    # Get products
    tables = [Observations.get_product_list(o) for o in tqdm(obs)]
    allprods = []
    for t in tables:
        good = np.logical_and(
            t['productType']=='SCIENCE',
            t['productSubGroupDescription']=='RATE'
            )
        t = t[good]
        t['prvversion'] = t['prvversion'].astype(str)
        allprods.append(t)
    allprods = vstack(allprods)
    prods = allprods[np.unique(allprods['dataURI'],return_index=True)[1]]

    # Replace filter name
    filt = filt.replace(';','-')

    # Save product list
    prods.write(filt+'/'+filt+'.fits',overwrite=True)

    # Update date in header
    hdul = fits.open(filt+'/'+filt+'.fits',mode='update')
    time = datetime.now(timezone.utc).strftime('%Y-%M-%dT%H:%M:%S.%f')[:-3]
    hdul[0].header['QDATE'] = (time, 'Date of MAST query for this file (UTC)')
    hdul.flush()
    hdul.close()
    
    # Download products
    if not os.path.isdir(filt):
        os.mkdir(filt)
    Observations.download_products(prods,download_dir=filt,flat=True)


