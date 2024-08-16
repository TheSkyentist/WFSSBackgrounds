#! /usr/bin/env python

# Import packages
import os
import numpy as np
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool
from datetime import datetime, timezone

# Astropy packages
import astropy.units as u
from astropy.io import fits
from astroquery.gaia import Gaia
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack, join


# Query from GAIA lite
def gquery(c, maglim=9.5, radius=1 / 30):
    return Gaia.launch_job(
        f"""
            SELECT TOP 1 g.phot_g_mean_mag
            FROM gaiadr3.gaia_source_lite as g
            WHERE 
                CONTAINS(
                    POINT('ICRS',g.ra,g.dec),
                    CIRCLE('ICRS',{c.ra.deg},{c.dec.deg},{radius})
                ) = 1
                AND (g.phot_g_mean_mag < {maglim})
        """
    ).get_results()


if __name__ == '__main__':
    # Record time of query
    time = datetime.now(timezone.utc).strftime('%Y-%M-%dT%H:%M:%S.%f')[:-3]

    # Query all data
    all_obs = Observations.query_criteria(
        instrument_name='NIRISS*',
        obs_collection='JWST',
        dataRights='PUBLIC',
        intentType='science',
        calib_level=2,
    )
    all_obs = all_obs[np.unique(all_obs['obs_id'], return_index=True)[1]]

    # Restrict to unique RA and dec
    coords, inverse = np.unique(all_obs['s_ra', 's_dec'], return_inverse=True)

    # Restrict to filter
    coords = SkyCoord(
        ra=coords['s_ra'], dec=coords['s_dec'], unit=u.degree, frame='icrs'
    )

    # Query GAIA
    print('Querying GAIA, removing contaminated obervations')
    with Pool(10) as pool:
        gaia = list(tqdm(pool.imap(gquery, coords), total=len(coords)))

    # Check if a bright star is nearby
    good = np.array([len(g) == 0 for g in gaia])
    all_obs = all_obs[good[inverse]]

    # Create product filter grism combos
    filts = [
        ';'.join(p)
        for p in product(['CLEAR', 'GR150C', 'GR150R'], ['F115W', 'F150W', 'F200W'])
    ]

    # Iterate over filters and grisms
    for filt in filts:
        # Restrict to filter
        obs = all_obs[all_obs['filters'] == filt]

        # Get products
        allprods = Observations.get_product_list(obs)
        good = np.logical_and.reduce(
            [
                allprods['filters'] == filt,
                allprods['productType'] == 'SCIENCE',
                allprods['productSubGroupDescription'] == 'RATE',
            ]
        )
        allprods = allprods[good]
        prods = allprods[np.unique(allprods['dataURI'], return_index=True)[1]]

        # Replace filter name
        filt = filt.replace(';', '-')

        # Save product list
        if not os.path.isdir(filt):
            os.mkdir(filt)

        # Get products that are not already downloaded
        todo = np.setdiff1d(prods['productFilename'], os.listdir(filt))

        # Download products
        if len(todo) > 0:
            Observations.download_products(
                join(prods, Table([todo])), download_dir=filt, flat=True
            )

        # Limit list to NIS and FULL
        pfiles = [f'{filt}/{f}' for f in prods['productFilename']]

        # Get header information
        keys = ['READPATT', 'SUBARRAY', 'EFFINTTM']
        htable = Table(
            [[fits.getval(p, k, 'PRIMARY') for p in pfiles] for k in keys], names=keys
        )

        # Restrict to NIS and FULL
        hstack([prods, htable])[
            np.logical_and(htable['READPATT'] == 'NIS', htable['SUBARRAY'] == 'FULL')
        ].write(f'{filt}/{filt}.fits', overwrite=True)

        # Update date in header
        hdul = fits.open(filt + '/' + filt + '.fits', mode='update')
        hdul[0].header['QDATE'] = (time, 'Date of MAST query for this file (UTC)')
        hdul.flush()
        hdul.close()
