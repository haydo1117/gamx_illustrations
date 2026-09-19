import numpy as np
import matplotlib.pyplot as plt
import polars as pl

## set dispersion
n = 10000 # number of samples
phi_low = 1.2 # low dispersion
phi_high = 2.0 # high dispersion

# Create the default random number generator
rng = np.random.default_rng()

# Define shape (k) and scale (theta) parameters
shape_low = 1/phi_low
shape_high = 1/phi_high

# Simulate 1,000 gamma random variables for each dispersion level
samples_low = rng.gamma(shape_low, size = n)
samples_high = rng.gamma(shape_high, size = n)

## data
x = rng.uniform(1000.0, 1000000.0, n)
eta = 0.3 * np.log(x)
mu = np.exp(eta) * 100.0

## df
df = pl.DataFrame(
    {
        'x': x,
        'eta': eta,
        'mu': mu,
        'phi_low': mu * samples_low * phi_low,
        'phi_high': mu * samples_high * phi_high,
    }
)
## export
df.write_parquet('data/log_gamma_unif_x.parquet')
