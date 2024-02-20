#! /usr/bin/env python

# Import packages
import glob
from astropy.table import Table
from multiprocessing import Pool
from jwst.assign_wcs import AssignWcsStep
from jwst.flatfield import FlatFieldStep

# Initialize steps
assign_wcs = AssignWcsStep()
flat_field = FlatFieldStep()

def process(r):
    # out = assign_wcs.process(r)
    out = flat_field.process(r)
    out.write(r.replace('rate','out'),overwrite=True)

# Iterate over filters and grisms
for filt in sorted(glob.glob('GR*')):

    # Get products
    rate = [f'{filt}/{f}' for f in Table.read(f'{filt}/{filt}.fits')['productFilename'].tolist()]

    # Process
    pool = Pool(processes=10)
    pool.map_async(process,rate)
    pool.close()
    pool.join()