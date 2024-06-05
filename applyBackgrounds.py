#! /usr/bin/env python

# Import packages
import os
import glob
import shutil
from tqdm import tqdm
from itertools import product

# Multiprocessing
from multiprocessing import cpu_count,Pool
from threadpoolctl import threadpool_limits

# Astropy
from astropy.io import fits
from astropy.table import Table

# Grizli
from grizli.prep import visit_grism_sky

# List of filter grism pairs
filts = ['-'.join(p) for p in product(['GR150C','GR150R'],['F115W','F150W','F200W'])]

# Number of threads
processes = 30
chunksize = 1

# WFSS Background
@threadpool_limits.wrap(limits=16, user_api='blas')
def wfssBack(f):

    # Grism dictionary
    grism = {
        'product':os.path.basename(f).strip('.fits'),
        'files':[f]
    }

    # Run
    visit_grism_sky(grism,column_average=False,verbose=False)
    
    # Delete extraneous files
    # for ext in ['_1_sky_background.info','_column.png']:
    #     os.remove(os.path.basename(f).replace('.fits',ext))
    # if os.path.exists(f.replace('.fits','_column.dat')):
    #     os.remove(f.replace('.fits','_column.dat'))

    return


# Background Step
def applyBackgrounds(filt):

    # Get products
    files = [f"{filt}/{f.replace('.fits','_flatfield.fits')}" for f in Table.read(f"{filt}/{filt}.fits")['productFilename']]

    # Delete and Create Old Direcotries
    exts = ['_crds','_custom']
    for ext in exts:
        if os.path.exists(f"{filt}{ext}"):
            shutil.rmtree(f"{filt}{ext}")
        os.mkdir(f"{filt}{ext}")

    # Copy files to new folder
    print(f"Copying {filt} data to new directories")
    for f in tqdm(files):
        shutil.copy(f,f"{filt}_crds")
        shutil.copy(f,f"{filt}_custom")

    # Get base filenames
    files = [os.path.basename(f) for f in files]

    # Delete SkyFlats 
    for b in glob.glob('/home/hviding/Projects/py-packages/grizli-conf/CONF/nis*skyflat.fits'): os.remove(b)

    # Run WFSS Background with CRDS flats
    print(f"Running WFSS Background on {filt} data (CRDS)")
    os.chdir(f"{filt}_crds")
    for f in tqdm(files): wfssBack(f)
    # wfssBack(files[0]) # Run first file so grizli can create the background
    # with Pool(processes) as pool: results = list(tqdm(pool.imap(wfssBack, files[1:], chunksize=chunksize), total=len(files)))
    os.chdir('..')

    # Copy custom SkyFlats
    for b in glob.glob('wfssbackgrounds/nis*skyflat.fits'):
        shutil.copy(b,'/home/hviding/Projects/py-packages/grizli-conf/CONF/')

    # Run WFSS Background with custom flats
    print(f"Running WFSS Background on {filt} data (Custom)")
    os.chdir(f"{filt}_custom")
    for f in tqdm(files): wfssBack(f)
    # with Pool(processes) as pool: results = list(tqdm(pool.imap(wfssBack, files, chunksize=chunksize), total=len(files)))
    os.chdir('..')

    return

if __name__ == "__main__":

    # Iterate over filters
    for filt in filts:
        applyBackgrounds(filt)
        print()