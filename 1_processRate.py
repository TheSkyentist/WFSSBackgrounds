#! /usr/bin/env python

# Import packages
import glob
from astropy.table import Table
from multiprocessing import Pool
from jwst.flatfield import FlatFieldStep

# Initialize steps
flat_field = FlatFieldStep()

# Apply pipeline steps 
def process(r):

    # Flat-field and write to file
    out = flat_field.process(r)
    out.write(r.replace('rate','out'),overwrite=True)

# Main function
if __name__ == '__main__':

    # Iterate over filters and grisms
    for filt in sorted(glob.glob('GR*')):

        # Get products in grism
        rate = [f'{filt}/{f}' for f in Table.read(f'{filt}/{filt}.fits')['productFilename'].tolist()]

        # Multiprocess
            pool = Pool(processes=10)
            pool.map_async(process,rate)
            pool.close()
            pool.join()