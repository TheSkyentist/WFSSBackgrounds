#! /usr/bin/env python

# Import packages
import numpy as np
from tqdm import tqdm

# Astropy
from astropy.io import fits
from astropy.table import Table

# Bayesian Inference
import jax
from jax import random, numpy as jnp
from numpyro import sample, infer, distributions as dist

# Set random key
rng = random.PRNGKey(0)

# Non reference pixels (in FULL mode)
nonref = (slice(4, -4), slice(4, -4))

# Get ratefiles
filt = 'CLEAR-F115W'
prods = Table.read(f'{filt}/{filt}.fits')
rate = [f'{filt}/{f}' for f in prods['productFilename']]

# Initalize arrays
X = np.zeros(shape=(len(rate), 2040, 2040), dtype=np.float32)
E = np.zeros(shape=(len(rate), 2040, 2040), dtype=np.float32)
M = np.invert(np.load(f'{filt}/masks.npy'))

# Fill arrays
for i, r in enumerate(tqdm(rate)):
    # Open fits file
    hdul = fits.open(r)

    # Get image data
    X[i] = hdul['SCI'].data[nonref]

    # Weight matrix
    E[i] = hdul['ERR'].data[nonref]

# Fill error arrays
E[np.invert(M)] = 1

# Limit to where we have on datapoint
good = M.sum(0) > 0

## Small size for testing
# good[:] = False
# good[1018:-1018, 1018:-1018] = True


# Convert arrays to JAXs
X, E, M = jnp.array(X[:, good]), jnp.array(E[:, good]), jnp.array(M[:, good])

# Get starting values
ext = '_flatfield_1_sky_background.info'
h_init = jnp.array(
    [
        np.genfromtxt(f'{filt}_custom/{f.replace('.fits',ext)}')[1]
        for f in prods['productFilename']
    ]
)

# Initial Background
B_init = jnp.ones(shape=(2040, 2040), dtype=np.float32)[good]

# Initial Flat-Field
F_init = jnp.array(
    fits.getdata(
        '/data/beegfs/astro-storage/groups/jwst/common/crds_cache/jwst_ops/references/jwst/niriss/jwst_niriss_flat_0261.fits',
        'SCI',
    )[nonref]
)[good]


# Model
def model(X, M, E):
    # Get dimensions
    n, x = X.shape

    # Sample H (n-dimensional vector)
    h = sample('h', dist.Uniform(0, 3).expand([n]))

    # Sample B and F (x-dimensional vectors)
    B = sample('B', dist.Uniform(0.9, 1.1).expand([x]))
    # F = sample('F', dist.Uniform(0.2, 1.6).expand([x]))

    # Normalize B and F
    B_norm = B / B.mean()
    # F_norm = F / F.mean()

    # Calculate the predicted values
    pred = h[:, None] * B_norm * F_init

    # Calculate likelyhood and mask
    L = dist.Normal(pred, E).mask(M)

    # Sample the data
    sample('X', L, obs=X)


# Initial strategy
init_strategy = infer.init_to_value(values={'h': h_init, 'B': B_init, 'F': F_init})

# Create the NUTS kernel
kernel = infer.NUTS(model, init_strategy=init_strategy)

# Batch size and number
N = 2

# Keep track of output
samples = [None] * N

# Initialize MCMC sampler
mcmc = infer.MCMC(
    kernel,
    num_warmup=100,
    num_samples=100,  # Limited by GPU memory
    num_chains=1,
)

# Iterate over the batches
for i in range(N):
    # Run the sampler
    mcmc.run(rng, X=X, M=M, E=E)

    # store the samples omn the CPU
    samples[i] = jax.device_put(mcmc.get_samples(), jax.devices('cpu')[0])

    # Reset the sampler
    rng = mcmc.last_state.rng_key
    mcmc.post_warmup_state = mcmc.last_state

# Concatenate and save the samples
all_samples = {
    k: jnp.concatenate([sample[k] for sample in samples]) for k in samples[-1].keys()
}

# Save the samples
for k, v in all_samples.items():
    # Reshape prime values
    if k in ['B', 'F']:
        # Normalize arrays
        v = v / v.mean(1)[:, None]

        # Initialize empty array
        mean = np.zeros((2040, 2040), dtype=np.float32)
        std = np.zeros((2040, 2040), dtype=np.float32)

        # Fill array with mean and std
        mean[good] = v.mean(0)
        std[good] = v.std(0)

        # Fix key
        k.replace('_prime', '')
    else:
        # Mean and std
        mean = v.mean(0)
        std = v.std(0)

    # Save to fits
    fits.PrimaryHDU(mean).writeto(f'{filt}/{k}-mean.fits', overwrite=True)
    fits.PrimaryHDU(std).writeto(f'{filt}/{k}-std.fits', overwrite=True)
